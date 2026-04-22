#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <omp.h>

// 说明：
// 1) 本实现采用“分块 LU + 部分选主元”；
// 2) A 为行优先存储，可原地被破坏；
// 3) 只在更新尾随矩阵时进行 OpenMP 并行；
// 4) 最后用前后代入求解 Ax=b，将解回写到 b。

// 交换 A 的两行（行优先存储），用于行置换
static inline void swap_rows(double *A, int n, int r1, int r2) {
    if (r1 == r2) return;
    double *row1 = A + (size_t)r1 * n;
    double *row2 = A + (size_t)r2 * n;
    for (int j = 0; j < n; ++j) {
        double tmp = row1[j];
        row1[j] = row2[j];
        row2[j] = tmp;
    }
}

// 求解 Ax = b（x 写回到 b）
void my_solver(int n, double *A, double *b) {
    if (n <= 0 || A == nullptr || b == nullptr) return;

    // 记录每一列的主元行位置（行置换）
    int *ipiv = (int *)malloc((size_t)n * sizeof(int));
    if (!ipiv) {
        fprintf(stderr, "Pivot allocation failed.\n");
        return;
    }
    for (int i = 0; i < n; ++i) ipiv[i] = i;

    // 分块大小（影响 cache 命中和并行效率，可调参）
    const int BS = 64;
    const double PIV_TOL = 1e-14;

    for (int k = 0; k < n; k += BS) {
        int kb = std::min(BS, n - k);

        // 面板分解：对当前块列 [k, k+kb) 做非分块 LU，并进行部分选主元
        for (int j = k; j < k + kb; ++j) {
            int pivot = j;
            double max_val = std::fabs(A[(size_t)j * n + j]);
            for (int i = j + 1; i < n; ++i) {
                double val = std::fabs(A[(size_t)i * n + j]);
                if (val > max_val) {
                    max_val = val;
                    pivot = i;
                }
            }

            // 主元过小视为奇异矩阵
            if (max_val < PIV_TOL) {
                printf("LU factorization failed: coefficient matrix is singular.\n");
                free(ipiv);
                return;
            }

            ipiv[j] = pivot;
            // 行交换：把主元行换到当前行
            if (pivot != j) swap_rows(A, n, j, pivot);

            double diag = A[(size_t)j * n + j];
            for (int i = j + 1; i < n; ++i) {
                double *rowi = A + (size_t)i * n;
                // L(i,j) = A(i,j) / A(j,j)
                rowi[j] /= diag;
                double lij = rowi[j];
                // 只更新面板内部的上三角区域
                for (int col = j + 1; col < k + kb; ++col) {
                    rowi[col] -= lij * A[(size_t)j * n + col];
                }
            }
        }

        int k2 = k + kb;
        int cols = n - k2;
        if (cols > 0) {
            // 计算 U12：对块行做前代（L11^{-1} * A(k:k2, k2:n)）
            for (int i = k; i < k2; ++i) {
                double *rowi = A + (size_t)i * n;
                double *rowi_right = rowi + k2;
                for (int t = k; t < i; ++t) {
                    double lij = rowi[t];
                    double *rowt = A + (size_t)t * n + k2;
                    #pragma omp simd
                    for (int j = 0; j < cols; ++j) {
                        rowi_right[j] -= lij * rowt[j];
                    }
                }
            }

            // 更新尾随子矩阵：A22 -= L21 * U12
            // 这是主计算量所在，用 OpenMP 并行
            #pragma omp parallel for schedule(static)
            for (int i = k2; i < n; ++i) {
                double *rowi = A + (size_t)i * n;
                double *rowi_right = rowi + k2;
                for (int t = k; t < k2; ++t) {
                    double a = rowi[t];
                    double *rowt = A + (size_t)t * n + k2;
                    #pragma omp simd
                    for (int j = 0; j < cols; ++j) {
                        rowi_right[j] -= a * rowt[j];
                    }
                }
            }
        }
    }

    // 把分解时的行置换应用到 b
    for (int i = 0; i < n; ++i) {
        if (ipiv[i] != i) {
            double tmp = b[i];
            b[i] = b[ipiv[i]];
            b[ipiv[i]] = tmp;
        }
    }

    // 前代：解 L * y = b（L 为单位下三角）
    for (int i = 0; i < n; ++i) {
        double sum = b[i];
        double *rowi = A + (size_t)i * n;
        for (int j = 0; j < i; ++j) {
            sum -= rowi[j] * b[j];
        }
        b[i] = sum;
    }

    // 回代：解 U * x = y
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        double *rowi = A + (size_t)i * n;
        for (int j = i + 1; j < n; ++j) {
            sum -= rowi[j] * b[j];
        }
        b[i] = sum / rowi[i];
    }

    // 释放资源
    free(ipiv);
}
