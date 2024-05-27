#include <iostream>
#include <omp.h>
#include <windows.h>
#include <emmintrin.h>
#include <immintrin.h>
#include<cmath>
#include<algorithm>

using namespace std;
const int N = 1000;
float A[N][N];
float a[N][N];
const int numthreads = 8;
int i, j, k;
int blockSize = 64;

void serial() {
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1;

        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}
void serial_SSE() {
    for (int k = 0; k < N; k++) {
        __m128 t1, t2;
        int j = k + 1;
        t1 = _mm_set_ps1(A[k][k]);  // Initialize t1

        for (; j <= N - 4; j += 4) {
            t2 = _mm_loadu_ps(A[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(A[k] + j, t2);
        }

        for (; j < N; j++) {
            A[k][j] /= A[k][k];
        }

        A[k][k] = 1.0f;

        for (int i = k + 1; i < N; i++) {
            t1 = _mm_set_ps1(A[i][k]);

            int j = k + 1;
            for (; j <= N - 4; j += 4) {
                t2 = _mm_loadu_ps(A[k] + j);
                __m128 t3 = _mm_loadu_ps(A[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_storeu_ps(A[i] + j, t3);
            }

            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }

            A[i][k] = 0.0f;
        }
    }
}

void m_reset(float m[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            m[i][j] = 0.0;
        }
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; ++j) {
            m[i][j] = static_cast<float>(rand() % 1000);  // Generate a random float between 0 and 1000
        }
    }
    for (int k = 0; k < N; k++) {  // Traverse each row as the base row
        for (int i = k + 1; i < N; i++) {  // Traverse each row below the base row
            for (int j = 0; j < N; j++) {  // Traverse each column
                m[i][j] += m[k][j];  // Add the element value of the base row to the corresponding element of the current row
            }
        }
    }
}

void backup(float src[N][N], float dst[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            dst[i][j] = src[i][j];
}

void restore(float src[N][N], float dst[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            dst[i][j] = src[i][j];
        }
    }
}
double measureFunction(void(*func)(), int numRuns) {
    LARGE_INTEGER start, end, freq;
    QueryPerformanceFrequency(&freq);

    double totalExecutionTime = 0.0;

    for (int i = 0; i < numRuns; ++i) {

        QueryPerformanceCounter(&start);

        func();

        QueryPerformanceCounter(&end);
        totalExecutionTime += static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
        restore(a, A);
    }

    return totalExecutionTime / numRuns; // Average execution time
}


void printMatrix(const float matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}
void openmpStatic_block() {
    int i, j, k;
    int blockSize = 64;  // 假设块大小为64
#pragma omp parallel num_threads(numthreads) shared(A) private(i, j, k)
    {
        for (int kk = 0; kk < N; kk += blockSize) {
            for (int jj = 0; jj < N; jj += blockSize) {
                for (int k = kk; k < min(kk + blockSize, N); ++k) {
#pragma omp single nowait
                    {
                        float tmp = A[k][k];
                        for (j = k + 1; j < N; ++j) {
                            A[k][j] = A[k][j] / tmp;
                        }
                        A[k][k] = 1.0;
                    }
#pragma omp barrier  // 确保所有线程都完成了上面的single块
#pragma omp for schedule(static)
                    for (i = k + 1; i < N; ++i) {
                        float tmp = A[i][k];
                        for (j = max(k + 1, jj); j < min(jj + blockSize, N); ++j) {
                            A[i][j] = A[i][j] - tmp * A[k][j];
                        }
                        A[i][k] = 0.0;
                    }
                }
            }
        }
    }
}


void openmpStatic_row() {
#pragma omp parallel num_threads(numthreads) shared(A) private(i, j, k)
    for (int k = 0; k < N; ++k) {
#pragma omp single
        {
            float tmp = A[k][k];
            for (int j = k + 1; j < N; ++j) {
                A[k][j] = A[k][j] / tmp;
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(static, 1)
        for (int i = k + 1; i < N; ++i) {
            float tmp = A[i][k];
            for (int j = k + 1; j < N; ++j) {
                A[i][j] = A[i][j] - tmp * A[k][j];
            }
            A[i][k] = 0.0;
        }  
    }
}
void openmpStatic_col() {
#pragma omp parallel num_threads(numthreads)
   
        for (int k = 0; k < N; ++k) {
#pragma omp single
            {
                float tmp = A[k][k];
                for (j = k + 1; j < N; ++j) {
                    A[k][j] = A[k][j] / tmp;
                }
                A[k][k] = 1.0;
            }
#pragma omp for schedule(static, 1)
            for (j = k + 1; j < N; ++j) {
                for (i = k + 1; i < N; ++i) {
                    float tmp = A[i][k];
                    A[i][j] = A[i][j] - tmp * A[k][j];
                }
            }
        }
#pragma omp single
        for (i = 1; i < N; i++) {
            for (j = 0; j < i; j++) {
                A[i][j] = 0.0;
            }
        }
    }
void openmpStatic_row_SSE() {
#pragma omp parallel num_threads(numthreads) shared(A) private(i, j, k)
    for (int k = 0; k < N; ++k) {
        __m128 t1, t2;
#pragma omp single
        {
            t1 = _mm_set_ps1(A[k][k]);
            int j = k + 1;
            for (; j <= N - 4; j += 4) {
                t2 = _mm_loadu_ps(A[k] + j);
                t2 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(A[k] + j, t2);
            }
            for (; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(static, 1)
        for (int i = k + 1; i < N; i++) {
            float tmp = A[i][k];
            t1 = _mm_set_ps1(tmp);
            int j = k + 1;
            for (; j <= N - 4; j += 4) {  
                t2 = _mm_loadu_ps(A[k] + j);
                __m128 t3 = _mm_loadu_ps(A[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_storeu_ps(A[i] + j, t3);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
}

void openmpDynamic_SSE() {
    int i, j, k;
#pragma omp parallel num_threads(numthreads) shared(A) private(i, j, k)
    for (k = 0; k < N; ++k) {
        __m128 t1, t2;
#pragma omp single
        {
            t1 = _mm_set_ps1(A[k][k]);
            int j = k + 1;
            for (; j <= N - 4; j += 4) {
                t2 = _mm_loadu_ps(A[k] + j);
                t2 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(A[k] + j, t2);
            }
            for (; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(dynamic, 1)
        for (i = k + 1; i < N; i++) {
            float tmp = A[i][k];
            t1 = _mm_set_ps1(tmp);
            int j = k + 1;
            for (; j <= N - 4; j += 4) {  // j + 4 <= N, this will include the extra elements not a multiple of 4
                t2 = _mm_loadu_ps(A[k] + j);
                __m128 t3 = _mm_loadu_ps(A[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_storeu_ps(A[i] + j, t3);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
}

void openmpDynamic_row() {
    int i, j, k;
#pragma omp parallel num_threads(numthreads) shared(A) private(i, j, k)
    for (k = 0; k < N; ++k) {
#pragma omp single
        {
            float tmp = A[k][k];
            for (int j = k + 1; j < N; ++j) {
                A[k][j] = A[k][j] / tmp;
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(dynamic, 1)
        for (i = k + 1; i < N; ++i) {
            float tmp = A[i][k];
            for (int j = k + 1; j < N; ++j) {
                A[i][j] = A[i][j] - tmp * A[k][j];
            }
            A[i][k] = 0.0;
        }  
    }
}
void openmpDynamic_col() {
    int i, j, k;
#pragma omp parallel num_threads(numthreads) shared(A) private(i, j, k)
    for (k = 0; k < N; ++k) {
#pragma omp single
        {
            float tmp = A[k][k];
            for (int j = k + 1; j < N; ++j) {
                A[k][j] = A[k][j] / tmp;
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(dynamic, 1)
        for (j = k + 1; j < N; ++j) {
            for (i = k + 1; i < N; ++i) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }  
#pragma omp for schedule(dynamic, 1)
        for (i = k + 1; i < N; ++i) {
            A[i][k] = 0.0;
        }
    }
}
void openmpDynamic_block() {
    int i, j, k;
    int blockSize = 64;  // 假设块大小为64
#pragma omp parallel num_threads(numthreads) shared(A) private(i, j, k)
    {
        for (int kk = 0; kk < N; kk += blockSize) {
            for (int jj = 0; jj < N; jj += blockSize) {
                for (int k = kk; k < min(kk + blockSize, N); ++k) {
#pragma omp single nowait
                    {
                        float tmp = A[k][k];
                        for (j = k + 1; j < N; ++j) {
                            A[k][j] = A[k][j] / tmp;
                        }
                        A[k][k] = 1.0;
                    }
#pragma omp barrier  
#pragma omp for schedule(dynamic)
                    for (i = k + 1; i < N; ++i) {
                        float tmp = A[i][k];
                        for (j = max(k + 1, jj); j < min(jj + blockSize, N); ++j) {
                            A[i][j] = A[i][j] - tmp * A[k][j];
                        }
                        A[i][k] = 0.0;
                    }
                }
            }
        }
    }
}

void openmpGuided_row() {
    int i, j, k;
#pragma omp parallel num_threads(numthreads)shared(A) private(i, j, k)
        for (int k = 0; k < N; ++k) {
#pragma omp single
            {
                float tmp = A[k][k];
                for (j = k + 1; j < N; ++j) {
                    A[k][j] = A[k][j] / tmp;
                }
                A[k][k] = 1.0;
            }
#pragma omp for schedule(guided, 1)
            for (i = k + 1; i < N; ++i) {
                float tmp = A[i][k];
                for (j = k + 1; j < N; ++j) {
                    A[i][j] = A[i][j] - tmp * A[k][j];
                }
                A[i][k] = 0.0;
            }
        }
    }
void openmpGuided_col() {
#pragma omp parallel num_threads(numthreads) shared(A) private(i, j, k)
    for (int k = 0; k < N; ++k) {
#pragma omp single
        {
            float tmp = A[k][k];
            for (j = k + 1; j < N; ++j) {
                A[k][j] = A[k][j] / tmp;
            }
            A[k][k] = 1.0;
        }

#pragma omp for schedule(guided, 1)
        for (j = k + 1; j < N; ++j) {
            for (i = k + 1; i < N; ++i) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
        }  
#pragma omp for schedule(guided, 1)
        for (i = k + 1; i < N; ++i) {
            A[i][k] = 0.0;
        }
    }
}
void openmpGuided_block() {
    int i, j, k;
    int blockSize = 64;  // 假设块大小为64
#pragma omp parallel num_threads(numthreads) shared(A) private(i, j, k)
    {
        for (int kk = 0; kk < N; kk += blockSize) {
            for (int jj = 0; jj < N; jj += blockSize) {
                for (int k = kk; k < min(kk + blockSize, N); ++k) {
#pragma omp single nowait
                    {
                        float tmp = A[k][k];
                        for (j = k + 1; j < N; ++j) {
                            A[k][j] = A[k][j] / tmp;
                        }
                        A[k][k] = 1.0;
                    }
#pragma omp barrier  
#pragma omp for schedule(guided)
                    for (i = k + 1; i < N; ++i) {
                        float tmp = A[i][k];
                        for (j = max(k + 1, jj); j < min(jj + blockSize, N); ++j) {
                            A[i][j] = A[i][j] - tmp * A[k][j];
                        }
                        A[i][k] = 0.0;
                    }
                }
            }
        }
    }
}
void openmpGuided_SSE() {
#pragma omp parallel num_threads(numthreads)shared(A) private(i, j, k)

    for (int k = 0; k < N; ++k) {
        __m128 t1, t2;
#pragma omp single
        {
            t1 = _mm_set_ps1(A[k][k]);
            int j = k + 1;
            for (; j <= N - 4; j += 4) {
                t2 = _mm_loadu_ps(A[k] + j);
                t2 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(A[k] + j, t2);
            }
            for (; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(guided, 1)
        for (int i = k + 1; i < N; i++) {
            float tmp = A[i][k];
            t1 = _mm_set_ps1(tmp);
            int j = k + 1;
            for (; j <= N - 4; j += 4) {  // j + 4 <= N, this will include the extra elements not a multiple of 4
                t2 = _mm_loadu_ps(A[k] + j);
                __m128 t3 = _mm_loadu_ps(A[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_storeu_ps(A[i] + j, t3);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
}
void openmpGuide_barrier() {
    int i, j, k;
#pragma omp parallel num_threads(numthreads)shared(A) private(i, j, k)
        for (int k = 0; k < N; ++k) {
#pragma omp single
            {
                float tmp = A[k][k];
                for (int j = k + 1; j < N; ++j) {
                    A[k][j] = A[k][j] / tmp;
                }
                A[k][k] = 1.0;
            }
#pragma omp barrier
#pragma omp for schedule(guided, 1) nowait
            for (int i = k + 1; i < N; ++i) {
                float tmp = A[i][k];
                for (int j = k + 1; j < N; ++j) {
                    A[i][j] = A[i][j] - tmp * A[k][j];
                }
                A[i][k] = 0.0;
            }
#pragma omp barrier
        }
    }

void openmpGuide_critical() {
    int i, j, k;
#pragma omp parallel num_threads(numthreads)shared(A) private(i, j, k)
    {
        for (int k = 0; k < N; ++k) {

#pragma omp single
            {
                float tmp = A[k][k];
                for (int j = k + 1; j < N; ++j) {
                    A[k][j] = A[k][j] / tmp;
                }
                A[k][k] = 1.0;
            }


#pragma omp barrier


#pragma omp for schedule(guided, 1) nowait
            for (int i = k + 1; i < N; ++i) {
                float tmp;
#pragma omp critical
                {
                    tmp = A[i][k];
                }
                for (int j = k + 1; j < N; ++j) {
                    A[i][j] = A[i][j] - tmp * A[k][j];
                }
                A[i][k] = 0.0;
            }

#pragma omp barrier
        }
    }
}


void openmpGuided_inner1() {
    for (int k = 0; k < N; ++k) {
       
        {
            float tmp = A[k][k];
            for (int j = k + 1; j < N; ++j) {
                A[k][j] = A[k][j] / tmp;
            }
            A[k][k] = 1.0;
        }
#pragma omp parallel for num_threads(numthreads) schedule(guided,1)
        for (int i = k + 1; i < N; ++i) {
            float tmp = A[i][k];
            for (int j = k + 1; j < N; ++j) {
                A[i][j] = A[i][j] - tmp * A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
}
void openmpGuided_row1() {
    int i, j, k;
#pragma omp parallel num_threads(numthreads) shared(A) private(i, j, k)
    {
        for (k = 0; k < N; ++k) {
#pragma omp single
            {
                float tmp = A[k][k];
                for (j = k + 1; j < N; ++j) {
                    A[k][j] = A[k][j] / tmp;
                }
                A[k][k] = 1.0;
            }
#pragma omp barrier  
            for (i = k + 1; i < N; ++i) {
                float tmp = A[i][k];
#pragma omp parallel for schedule(guided, 1)
                for (j = k + 1; j < N; ++j) {
                    A[i][j] = A[i][j] - tmp * A[k][j];
                }
                A[i][k] = 0.0;
            }
        }
    }
}
void openmpGuided_inner2() {
    for (int k = 0; k < N; ++k) {
        {
            float tmp = A[k][k];
            for (int j = k + 1; j < N; ++j) {
                A[k][j] = A[k][j] / tmp;
            }
            A[k][k] = 1.0;
        }

        for (int i = k + 1; i < N; ++i) {
            float tmp = A[i][k];
#pragma omp parallel for num_threads(numthreads) schedule(guided, 1) private(j)
            for (int j = k + 1; j < N; ++j) {
                A[i][j] = A[i][j] - tmp * A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
}

void openmp_SIMD1()
{
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(8) private(i, j, k, tmp) shared(A, N)
    for (k = 0; k < N; k++)
    {
#pragma omp single
        {
            tmp = A[k][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                A[k][j] = A[k][j] / tmp;
            }
            A[k][k] = 1.0;
        }
#pragma omp for schedule(guided,1)
        for (i = k + 1; i < N; i++)
        {
            tmp = A[i][k];
#pragma omp simd aligned(matrix : 16) simdlen(4)
            for (j = k + 1; j < N; j++)
            {
                A[i][j] = A[i][j] - tmp * A[k][j];
            }
            A[i][k] = 0.0;
        }
    }
}
void openmpGuided_barrier() {
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(numthreads) private(i, j, k, tmp)shared(A, N)
    {
        for (int k = 0; k < N; ++k) {
            // 串行部分，可以尝试并行化
#pragma omp single
            {
                float tmp = A[k][k];
                for (int j = k + 1; j < N; ++j) {
                    A[k][j] = A[k][j] / tmp;
                }
                A[k][k] = 1.0;
            }

            // 使用显式屏障同步所有线程
#pragma omp barrier

            // 并行部分，使用行划分
#pragma omp for schedule(guided, 1) nowait
            for (int i = k + 1; i < N; ++i) {
                float tmp = A[i][k];
                for (int j = k + 1; j < N; ++j) {
                    A[i][j] = A[i][j] - tmp * A[k][j];
                }
                A[i][k] = 0.0;
            }

            // 使用显式屏障确保所有线程在进入下一行处理前同步
#pragma omp barrier
        }
    }
}
void openmpGuided_critical() {
    int i, j, k;
    float tmp;
#pragma omp parallel num_threads(numthreads), private(i, j, k, tmp)shared(A, N)
    {
        for (int k = 0; k < N; ++k) {
            // 串行部分，可以尝试并行化
#pragma omp single
            {
                float tmp = A[k][k];
                for (int j = k + 1; j < N; ++j) {
                    A[k][j] = A[k][j] / tmp;
                }
                A[k][k] = 1.0;
            }

            // 使用显式屏障同步所有线程
#pragma omp barrier

            // 并行部分，使用行划分
#pragma omp for schedule(guided, 1) nowait
            for (int i = k + 1; i < N; ++i) {
                float tmp;
#pragma omp critical
                {
                    tmp = A[i][k];
                }
                for (int j = k + 1; j < N; ++j) {
                    A[i][j] = A[i][j] - tmp * A[k][j];
                }
                A[i][k] = 0.0;
            }

            // 使用显式屏障确保所有线程在进入下一行处理前同步
#pragma omp barrier
        }
    }
}


int main() {
    m_reset(A);
    backup(A, a);
    int numRuns = 20;
    cout << "serial:" << measureFunction(serial, numRuns) << " ms" << endl;
    cout << "serial_SSE:" << measureFunction(serial_SSE, numRuns) << " ms" << endl;
    //cout << "openmpStatic_block:" << measureFunction(openmpStatic_block, numRuns) << " ms" << endl;
    //cout << "openmpStatic_row:" << measureFunction(openmpStatic_row, numRuns) << " ms" << endl;
    //cout << "openmpStatic_row_SSE:" << measureFunction(openmpStatic_row_SSE, numRuns) << " ms" << endl;
    //cout << "openmpStatic_col:" << measureFunction(openmpStatic_col, numRuns) << " ms" << endl;
    //cout << "openmpDynamic_block:" << measureFunction(openmpDynamic_block, numRuns) << " ms" << endl;
    //cout << "openmpDynamic_row:" << measureFunction(openmpDynamic_row, numRuns) << " ms" << endl;
    //cout << "openmpDynamic_SSE:" << measureFunction(openmpDynamic_SSE, numRuns) << " ms" << endl;
    cout << "openmpGuided_block:" << measureFunction(openmpGuided_block, numRuns) << " ms" << endl;
    //cout << "openmpGuided_row:" << measureFunction(openmpGuided_row, numRuns) << " ms" << endl;
    //cout << "openmpGuided_inner1:" << measureFunction(openmpGuided_inner1, numRuns) << " ms" << endl;
    //cout << "openmpGuided_row1:" << measureFunction(openmpGuided_row1, numRuns) << " ms" << endl;
    //cout << "openmpGuided_inner2:" << measureFunction(openmpGuided_inner2, numRuns) << " ms" << endl;
    //cout << "openmpGuided_barrier:" << measureFunction(openmpGuided_barrier, numRuns) << " ms" << endl;
    //cout << "openmpGuided_critical:" << measureFunction(openmpGuided_critical, numRuns) << " ms" << endl;
    //cout << "openmpStatic_col:" << measureFunction(openmpStatic_col, numRuns) << " ms" << endl;
    //cout << "openmpDynamic_col:" << measureFunction(openmpDynamic_col, numRuns) << " ms" << endl;
    //cout << "openmpGuided_col:" << measureFunction(openmpGuided_col, numRuns) << " ms" << endl;
    //cout << "openmpGuide_SSE:" << measureFunction(openmpGuided_SSE, numRuns) << " ms" << endl;


}