#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_WIDTH 16
#define WARMUP_RUNS 5
#define MEASURE_RUNS 30
#define L2_BUFFER_MB 128

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

struct Stats {
    double median_ms;
    double mean_ms;
    double stddev_ms;
    double ci_low_ms;
    double ci_high_ms;
    double cv_percent;
    int kept;
    int total;
    double gflops;
    double gos;
};

__global__ void matMulNaive(const float* M, const float* N, float* P,
                            int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int t = 0; t < k; ++t) {
            sum += M[row * k + t] * N[t * n + col];
        }
        P[row * n + col] = sum;
    }
}

__global__ void matMulTiled(const float* M, const float* N, float* P,
                            int m, int k, int n) {
    __shared__ float Ms[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Ns[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (k + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int tile = 0; tile < numTiles; ++tile) {
        int m_col = tile * TILE_WIDTH + threadIdx.x;
        int n_row = tile * TILE_WIDTH + threadIdx.y;

        if (row < m && m_col < k) {
            Ms[threadIdx.y][threadIdx.x] = M[row * k + m_col];
        } else {
            Ms[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (n_row < k && col < n) {
            Ns[threadIdx.y][threadIdx.x] = N[n_row * n + col];
        } else {
            Ns[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += Ms[threadIdx.y][i] * Ns[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        P[row * n + col] = sum;
    }
}

__global__ void flushL2Kernel(float* buffer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buffer[idx] = buffer[idx] + 1.0f;
    }
}

void initMatrix(float* A, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void matMulCPU(const float* M, const float* N, float* P, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int t = 0; t < k; ++t) {
                sum += M[i * k + t] * N[t * n + j];
            }
            P[i * n + j] = sum;
        }
    }
}

float maxAbsError(const float* A, const float* B, size_t size) {
    float maxErr = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float e = fabsf(A[i] - B[i]);
        if (e > maxErr) maxErr = e;
    }
    return maxErr;
}

double medianOf(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return 0.0;
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double studentT95TwoSided(int df) {
    // Valeurs t_(0.975, df) utiles pour n <= 30, puis approx normale
    static const double table[] = {
        0.0,
        12.706, 4.303, 3.182, 2.776, 2.571,
        2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060,
        2.056, 2.052, 2.048, 2.045, 2.042
    };
    if (df <= 0) return 0.0;
    if (df <= 30) return table[df];
    if (df <= 40) return 2.021;
    if (df <= 60) return 2.000;
    if (df <= 120) return 1.980;
    return 1.960;
}

Stats computeStats(const std::vector<double>& rawTimesMs, int m, int k, int n) {
    Stats s{};
    s.total = (int)rawTimesMs.size();

    double med = medianOf(rawTimesMs);

    std::vector<double> absDev(rawTimesMs.size());
    for (size_t i = 0; i < rawTimesMs.size(); ++i) {
        absDev[i] = fabs(rawTimesMs[i] - med);
    }
    double mad = medianOf(absDev);

    std::vector<double> filtered;
    if (mad == 0.0) {
        filtered = rawTimesMs;
    } else {
        for (double x : rawTimesMs) {
            if (fabs(x - med) <= 3.0 * mad) {
                filtered.push_back(x);
            }
        }
        if (filtered.empty()) {
            filtered = rawTimesMs;
        }
    }

    s.kept = (int)filtered.size();
    s.median_ms = medianOf(filtered);

    double mean = std::accumulate(filtered.begin(), filtered.end(), 0.0) / filtered.size();
    s.mean_ms = mean;

    double var = 0.0;
    if (filtered.size() > 1) {
        for (double x : filtered) {
            double d = x - mean;
            var += d * d;
        }
        var /= (filtered.size() - 1);
    }
    s.stddev_ms = std::sqrt(var);

    double tcrit = studentT95TwoSided((int)filtered.size() - 1);
    double margin = 0.0;
    if (!filtered.empty()) {
        margin = tcrit * s.stddev_ms / std::sqrt((double)filtered.size());
    }

    s.ci_low_ms = mean - margin;
    s.ci_high_ms = mean + margin;
    s.cv_percent = (mean != 0.0) ? (s.stddev_ms / mean * 100.0) : 0.0;

    double median_s = s.median_ms / 1000.0;
    double flops = 2.0 * (double)m * (double)n * (double)k;
    double bytes = 4.0 * ((double)m * k + (double)k * n + (double)m * n);

    s.gflops = flops / (median_s * 1e9);
    s.gos = bytes / (median_s * 1e9);

    return s;
}

float verifyKernel(void (*cpuRef)(const float*, const float*, float*, int, int, int),
                   bool tiled, int m, int k, int n) {
    size_t sizeM = (size_t)m * k;
    size_t sizeN = (size_t)k * n;
    size_t sizeP = (size_t)m * n;

    std::vector<float> h_M(sizeM);
    std::vector<float> h_N(sizeN);
    std::vector<float> h_P_cpu(sizeP);
    std::vector<float> h_P_gpu(sizeP);

    initMatrix(h_M.data(), sizeM);
    initMatrix(h_N.data(), sizeN);

    float *d_M, *d_N, *d_P;
    CUDA_CHECK(cudaMalloc(&d_M, sizeM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_N, sizeN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P, sizeP * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_M, h_M.data(), sizeM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N.data(), sizeN * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (tiled) {
        matMulTiled<<<grid, block>>>(d_M, d_N, d_P, m, k, n);
    } else {
        matMulNaive<<<grid, block>>>(d_M, d_N, d_P, m, k, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_P_gpu.data(), d_P, sizeP * sizeof(float), cudaMemcpyDeviceToHost));

    cpuRef(h_M.data(), h_N.data(), h_P_cpu.data(), m, k, n);

    float err = maxAbsError(h_P_gpu.data(), h_P_cpu.data(), sizeP);

    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));

    return err;
}

Stats benchmarkKernel(bool tiled, int m, int k, int n, float* d_flushBuf, int flushCount) {
    size_t sizeM = (size_t)m * k;
    size_t sizeN = (size_t)k * n;
    size_t sizeP = (size_t)m * n;

    std::vector<float> h_M(sizeM);
    std::vector<float> h_N(sizeN);

    float *d_M, *d_N, *d_P;
    CUDA_CHECK(cudaMalloc(&d_M, sizeM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_N, sizeN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P, sizeP * sizeof(float)));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float sink = 0.0f;
    std::vector<double> timesMs;
    timesMs.reserve(MEASURE_RUNS);

    int flushBlock = 256;
    int flushGrid = (flushCount + flushBlock - 1) / flushBlock;

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        initMatrix(h_M.data(), sizeM);
        initMatrix(h_N.data(), sizeN);

        CUDA_CHECK(cudaMemcpy(d_M, h_M.data(), sizeM * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_N, h_N.data(), sizeN * sizeof(float), cudaMemcpyHostToDevice));

        flushL2Kernel<<<flushGrid, flushBlock>>>(d_flushBuf, flushCount);
        CUDA_CHECK(cudaGetLastError());

        if (tiled) {
            matMulTiled<<<grid, block>>>(d_M, d_N, d_P, m, k, n);
        } else {
            matMulNaive<<<grid, block>>>(d_M, d_N, d_P, m, k, n);
        }
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&sink, d_P, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < MEASURE_RUNS; ++i) {
        initMatrix(h_M.data(), sizeM);
        initMatrix(h_N.data(), sizeN);

        CUDA_CHECK(cudaMemcpy(d_M, h_M.data(), sizeM * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_N, h_N.data(), sizeN * sizeof(float), cudaMemcpyHostToDevice));

        flushL2Kernel<<<flushGrid, flushBlock>>>(d_flushBuf, flushCount);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(start));
        if (tiled) {
            matMulTiled<<<grid, block>>>(d_M, d_N, d_P, m, k, n);
        } else {
            matMulNaive<<<grid, block>>>(d_M, d_N, d_P, m, k, n);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        CUDA_CHECK(cudaMemcpy(&sink, d_P, sizeof(float), cudaMemcpyDeviceToHost));
        timesMs.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));

    volatile float keepSink = sink;
    (void)keepSink;

    return computeStats(timesMs, m, k, n);
}

void runConfig(int m, int k, int n, float* d_flushBuf, int flushCount) {
    printf("=== Config: M(%dx%d) * N(%dx%d) = P(%dx%d) ===\n", m, k, k, n, m, n);
    printf("Verification:\n");

    float errNaive = verifyKernel(matMulCPU, false, m, k, n);
    float errTiled = verifyKernel(matMulCPU, true, m, k, n);

    printf("Naif -> Err max: %.6f\n", errNaive);
    printf("Tuile -> Err max: %.6f\n", errTiled);

    Stats naive = benchmarkKernel(false, m, k, n, d_flushBuf, flushCount);
    Stats tiled = benchmarkKernel(true, m, k, n, d_flushBuf, flushCount);

    printf("Naif:\n");
    printf("Mediane: %.4f ms IC 95%%: [%.4f, %.4f]\n",
           naive.median_ms, naive.ci_low_ms, naive.ci_high_ms);
    printf("GFLOP/s: %.1f Go/s: %.1f CV: %.2f%% N: %d/%d\n",
           naive.gflops, naive.gos, naive.cv_percent, naive.kept, naive.total);

    printf("Tuile:\n");
    printf("Mediane: %.4f ms IC 95%%: [%.4f, %.4f]\n",
           tiled.median_ms, tiled.ci_low_ms, tiled.ci_high_ms);
    printf("GFLOP/s: %.1f Go/s: %.1f CV: %.2f%% N: %d/%d\n",
           tiled.gflops, tiled.gos, tiled.cv_percent, tiled.kept, tiled.total);

    double speedup = naive.median_ms / tiled.median_ms;
    printf("Acceleration: %.2fx\n", speedup);
}

int main() {
    srand(42);

    printf("=== Protocole ===\n");
    printf("Prechauffages: %d\n", WARMUP_RUNS);
    printf("Mesures: %d\n", MEASURE_RUNS);
    printf("Buffer L2: %d Mo\n", L2_BUFFER_MB);

    size_t flushBytes = (size_t)L2_BUFFER_MB * 1024 * 1024;
    int flushCount = (int)(flushBytes / sizeof(float));

    float* d_flushBuf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flushBuf, flushCount * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_flushBuf, 0, flushCount * sizeof(float)));

    runConfig(512, 512, 512, d_flushBuf, flushCount);
    runConfig(1024, 1024, 1024, d_flushBuf, flushCount);
    runConfig(2048, 2048, 2048, d_flushBuf, flushCount);
    runConfig(4096, 4096, 4096, d_flushBuf, flushCount);
    runConfig(2048, 1024, 512, d_flushBuf, flushCount);
    runConfig(1024, 4096, 512, d_flushBuf, flushCount);
    runConfig(1000, 1500, 2000, d_flushBuf, flushCount);

    CUDA_CHECK(cudaFree(d_flushBuf));
    return 0;
}