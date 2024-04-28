#include <iostream>
#include <windows.h>
#include <cstdlib> 
#include <ctime>   
#include <immintrin.h> //AVX、AVX2
#include<string>

using namespace std;
const int N =500;
alignas(32) float A[N][N];
alignas(32) float a[N][N];
alignas(32) float b[N];
alignas(32) float x[N];
long long head, tail, freq;//计时变量

void gauss(float A[N][N], float b[N], float x[N]) {
    int n = N;
    //Gaussian elimination
    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i < n; ++i) {
            float factor = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution process
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; --i) {
        float sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
}

void unalignedAVX(float A[N][N], float b[N], float x[N]) {
    int n = N;

    // Gaussian elimination
    for (int k = 0; k < n; ++k) {
        int j;
        __m256 t1 = _mm256_set1_ps(A[k][k]);
        for (j = k + 1; j + 8 <= N; j += 8) {
            __m256 t2 = _mm256_loadu_ps(&A[k][j]);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(&A[k][j], t2);
        }
        for (; j < N; j++) { 
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0f;
        for (int i = k + 1; i < n; ++i) {
            __m256 factor = _mm256_set1_ps(A[i][k] / A[k][k]);
            for (j = k; j + 8 <= n; j += 8) {
                __m256 row_k = _mm256_loadu_ps(&A[k][j]);
                __m256 row_i = _mm256_loadu_ps(&A[i][j]);
                __m256 prod = _mm256_mul_ps(factor, row_k);
                row_i = _mm256_sub_ps(row_i, prod);
                _mm256_storeu_ps(&A[i][j], row_i);
            }
            for (; j < n; ++j) { 
                A[i][j] -= A[i][k] * A[k][j];
            }
            
            b[i] -= b[k] * A[i][k];
        }
    }

    // Back substitution
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; --i) {
        __m256 sum_vec = _mm256_setzero_ps(); 
        float sum = b[i];
        int j;
        for (j = i + 1; j + 8 <= n; j += 8) {
            __m256 a_vec = _mm256_loadu_ps(&A[i][j]);
            __m256 x_vec = _mm256_loadu_ps(&x[j]);
            __m256 prod = _mm256_mul_ps(a_vec, x_vec);
            sum_vec = _mm256_add_ps(sum_vec, prod);
        }
        for (; j < n; ++j) { 
            sum -= A[i][j] * x[j];
        }
        __m256 hsum = _mm256_hadd_ps(sum_vec, sum_vec);
        __m128 hsum_high = _mm256_extractf128_ps(hsum, 1);  
        __m128 hsum_low = _mm256_castps256_ps128(hsum);     
        __m128 hsum_final = _mm_add_ps(hsum_low, hsum_high); 
        hsum_final = _mm_hadd_ps(hsum_final, hsum_final);   
        float final_result = _mm_cvtss_f32(hsum_final);     

        sum -= final_result; 

        x[i] = sum / A[i][i]; 
    }
}


void unalignedAVXelimination(float A[N][N], float b[N], float x[N]) {
    int n = N;
    for (int k = 0; k < n; ++k) {
        int j;
        __m256 t1 = _mm256_set1_ps(A[k][k]);
        for (j = k + 1; j + 8 <= N; j += 8) {
            __m256 t2 = _mm256_loadu_ps(&A[k][j]);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(&A[k][j], t2);
        }
        for (; j < N; j++) { 
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0f;
        for (int i = k + 1; i < n; ++i) {
            __m256 factor = _mm256_set1_ps(A[i][k] / A[k][k]);
            for (j = k; j + 8 <= n; j += 8) {
                __m256 row_k = _mm256_loadu_ps(&A[k][j]);
                __m256 row_i = _mm256_loadu_ps(&A[i][j]);
                __m256 prod = _mm256_mul_ps(factor, row_k);
                row_i = _mm256_sub_ps(row_i, prod);
                _mm256_storeu_ps(&A[i][j], row_i);
            }
            for (; j < n; ++j) { 
                A[i][j] -= A[i][k] * A[k][j];
            }
           
            b[i] -= b[k] * A[i][k];
        }
    }
    // Back substitution process
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; --i) {
        float sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * x[j];
        }
        x[i] = sum / A[i][i];
    }
}

void unalignedAVXback(float A[N][N], float b[N], float x[N]) {
    int n = N;
    //Gaussian elimination
    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i < n; ++i) {
            float factor = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    // Back substitution
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; --i) {
        __m256 sum_vec = _mm256_setzero_ps(); 
        float sum = b[i];
        int j;
        for (j = i + 1; j + 8 <= n; j += 8) {
            __m256 a_vec = _mm256_loadu_ps(&A[i][j]);
            __m256 x_vec = _mm256_loadu_ps(&x[j]);
            __m256 prod = _mm256_mul_ps(a_vec, x_vec);
            sum_vec = _mm256_add_ps(sum_vec, prod);
        }
        for (; j < n; ++j) { 
            sum -= A[i][j] * x[j];
        }
        __m256 hsum = _mm256_hadd_ps(sum_vec, sum_vec);
        __m128 hsum_high = _mm256_extractf128_ps(hsum, 1);  
        __m128 hsum_low = _mm256_castps256_ps128(hsum);     
        __m128 hsum_final = _mm_add_ps(hsum_low, hsum_high); 
        hsum_final = _mm_hadd_ps(hsum_final, hsum_final);   
        float final_result = _mm_cvtss_f32(hsum_final);     

        sum -= final_result; 

        x[i] = sum / A[i][i]; 
    }
}

void alignedAVX(float A[N][N], float b[N], float x[N]) {
    __m256 t1, t2;
    for (int k = 0; k < N; k++) {
        int j;
        if (A[k][k] == 0) continue;
        t1 = _mm256_set1_ps(A[k][k]);
        long long mkaddr = (long long)(&A[k][k]);
        int offset = (mkaddr % 32) / 4;
        int start = k + (8 - offset) % 8;
        int n = N - (N - start) % 8;
        if (start >= N) {
            start = N;
        }
        for (int i = k; i < start; i++) {
            A[k][i] = A[k][i] / A[k][k];
        }
        for (int i = n; i < N; i++) {
            A[k][i] = A[k][i] / A[k][k];
        }
        for (int j = start; j < n; j += 8) {
            t2 = _mm256_load_ps(A[k] + j);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_store_ps(A[k] + j, t2);
        }
        A[k][k] = 1.0f;
        for (int i = k + 1; i < N; ++i) {
            __m256 factor = _mm256_set1_ps(A[i][k] / A[k][k]);
            for (int j = k; j < start; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            for (int j = n; j < N; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            for (j = start; j +8<= N; j += 8) {
                __m256 row_k = _mm256_load_ps(&A[k][j]);
                __m256 row_i = _mm256_load_ps(&A[i][j]);
                __m256 prod = _mm256_mul_ps(factor, row_k);
                row_i = _mm256_sub_ps(row_i, prod);
                _mm256_store_ps(&A[i][j], row_i);
            }
            b[i] -= b[k] * A[i][k];
        }
    }
    x[N - 1] = b[N - 1] / A[N - 1][N - 1];
    for (int i = N - 2; i >= 0; --i) {
        __m256 sum_vec = _mm256_setzero_ps();
        float sum = b[i];
        long long aiaddr = (long long)(&A[i][i + 1]);
        int offset = (aiaddr % 32) / 4;
        int start = i + 1 + (8 - offset) % 8;
        int n = N - (N - start) % 8;
        if (start >= N) {
            start = N;
        }
        for (int j = i + 1; j < start; ++j) {
            sum -= A[i][j] * x[j];
        }
        for (int j = n; j < N; ++j) {
            sum -= A[i][j] * x[j];
        }
        for (int j = start; j + 8 <= n; j += 8) {
            __m256 a_vec = _mm256_load_ps(&A[i][j]);
            __m256 x_vec = _mm256_load_ps(&x[j]);
            __m256 prod = _mm256_mul_ps(a_vec, x_vec);
            sum_vec = _mm256_add_ps(sum_vec, prod);
        }
        //sum -= _mm256_reduce_add_ps(sum_vec);
        //x[i] = sum / A[i][i];
        __m256 sum_vec_perm = _mm256_permute2f128_ps(sum_vec, sum_vec, 1); 
        sum_vec = _mm256_add_ps(sum_vec, sum_vec_perm); 
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec); 
        sum_vec = _mm256_hadd_ps(sum_vec, sum_vec); 

       
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 1), _mm256_castps256_ps128(sum_vec));

        sum -= _mm_cvtss_f32(_mm_hadd_ps(sum128, sum128)); 
        sum -= _mm_cvtss_f32(_mm_hadd_ps(sum128, sum128)); 

        x[i] = sum / A[i][i];

    }
}


double measureFunction(void(*func)(float[][N], float[N], float[N]), int numRuns) {
    LARGE_INTEGER start, end, freq;
    QueryPerformanceFrequency(&freq);

    double totalExecutionTime = 0.0;

    for (int i = 0; i < numRuns; ++i) {
        QueryPerformanceCounter(&start);

        func(A, b, x);

        QueryPerformanceCounter(&end);
        totalExecutionTime += static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
    }

    return totalExecutionTime / numRuns; 
}


void m_reset(float m[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < i; ++j) {
            m[i][j] = 0.0;
        }
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; ++j) {
            m[i][j] = rand() % 1000; 
        }
    }
    for (int k = 0; k < N; k++) {  // 遍历每一行，作为基准行
        for (int i = k + 1; i < N; i++) {  // 遍历基准行之下的每一行
            for (int j = 0; j < N; j++) {  // 遍历每一列
                m[i][j] += m[k][j];  // 将基准行的元素值累加到当前行的对应元素上
            }
        }
    }
}
void initialize_b(float b[N]) {
    for (int i = 0; i < N; ++i) {
        b[i] = (float)(rand() % 1000); 
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

void printMatrix(const float matrix[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}


int main() {
    srand(time(NULL));
    m_reset(A);  
    initialize_b(b); 
    //printMatrix(A);
    backup(A, a);  
    int numExperiments = 1;
    float* backup_b = new float[N];  
    memcpy(backup_b, b, N * sizeof(float)); 

    //cout << "Timing gauss: " << measureFunction(gauss, numExperiments) << " ms" << endl;
    //restore(a, A);
    //memcpy(b, backup_b, N * sizeof(float)); // Restore b

    cout << "Timing unalignedAVX: " << measureFunction(unalignedAVX, numExperiments) << " ms" << endl;
    //restore(a, A);
    //memcpy(b, backup_b, N * sizeof(float)); // Restore b
    //////printMatrix(A);
    //cout << "Timing unalignedAVXdiv: " << measureFunction(unalignedAVXelimination, numExperiments) << " ms" << endl;
    //restore(a, A);  // Restore A to its initial state before the next function
    //memcpy(b, backup_b, N * sizeof(float)); // Restore b

    //cout << "Timing unalignedAVXelimination: " << measureFunction(unalignedAVXback, numExperiments) << " ms" << endl;
    //restore(a, A);
    //memcpy(b, backup_b, N * sizeof(float)); // Restore b

   //cout << "Timing alignedAVX: " << measureFunction(alignedAVX, numExperiments) << " ms" << endl;
    
    //printMatrix(A);
    /*for (int i = 0; i < N; ++i) {
        cout << x[i] << endl;
    }*/
    return 0;
}
