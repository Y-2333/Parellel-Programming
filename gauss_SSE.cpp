#include <iostream>
#include <windows.h>
#include <cstdlib> 
#include <ctime>   
#include <nmmintrin.h> //SSSE4.2

using namespace std;
const int N = 500;
alignas(16) float A[N][N];
alignas(16) float a[N][N];
alignas(16) float b[N];
alignas(16) float x[N];
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

void unalignedSSE(float A[N][N], float b[N], float x[N]) {
    int n = N;

    // Gaussian elimination
    for (int k = 0; k < n; ++k) {
        int j;
        __m128 t1 = _mm_set_ps1(A[k][k]);
        __m128 factor = _mm_setzero_ps();
        for (j = k + 1; j + 4 <= N; j += 4) {
            __m128 t2 = _mm_loadu_ps(&A[k][j]);
            factor = _mm_div_ps(t2, t1);
            _mm_storeu_ps(&A[k][j], factor);
        }
        for (; j < n; j++) { // Explicit tail handling
            float factor = A[k][j] / A[k][k];
            A[k][j] = factor;
            //A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0f;
        for (int i = k + 1; i < n; ++i) {
            //__m128 factor = _mm_set_ps1(A[i][k] / A[k][k]);
            for (j = k; j + 4 <= n; j += 4) {
                __m128 row_k = _mm_loadu_ps(&A[k][j]);
                __m128 row_i = _mm_loadu_ps(&A[i][j]);
                __m128 prod = _mm_mul_ps(factor, row_k);
                row_i = _mm_sub_ps(row_i, prod);
                _mm_storeu_ps(&A[i][j], row_i);
            }
            for (; j < n; ++j) { // Handle tail case
                A[i][j] -= A[i][k] * A[k][j];
            }
            // Update vector b
            b[i] -= b[k] * A[i][k];
        }
    }

    // Back substitution
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; --i) {
        __m128 sum_vec = _mm_setzero_ps(); // Initialize sum as zero
        float sum = b[i];
        int j;
        for (j = i + 1; j + 4 <= n; j += 4) {
            __m128 a_vec = _mm_loadu_ps(&A[i][j]);
            __m128 x_vec = _mm_loadu_ps(&x[j]);
            __m128 prod = _mm_mul_ps(a_vec, x_vec);
            sum_vec = _mm_add_ps(sum_vec, prod);
        }
        for (; j < n; ++j) { // Handle tail case
            sum -= A[i][j] * x[j];
        }
        sum -= _mm_cvtss_f32(_mm_hadd_ps(sum_vec, sum_vec)); // Horizontal add for sum_vec
        sum -= _mm_cvtss_f32(_mm_hadd_ps(sum_vec, sum_vec)); // Repeat to accumulate all elements
        x[i] = sum / A[i][i];
    }
}
void unalignedSSEdiv(float A[N][N], float b[N], float x[N]) {
    int n = N;

    // Gaussian elimination
    for (int k = 0; k < n; ++k) {
        int j;
        __m128 t1 = _mm_set_ps1(A[k][k]);
        __m128 factor = _mm_setzero_ps();
        for (j = k + 1; j + 4 <= N; j += 4) {
            __m128 t2 = _mm_loadu_ps(&A[k][j]);
            factor = _mm_div_ps(t2, t1);
            _mm_storeu_ps(&A[k][j], factor);
        }
        for (; j < n; j++) { // Explicit tail handling
            float factor = A[k][j] / A[k][k];
            A[k][j] = factor;
            //A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0f;
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            for (j = k ; j < n; j++) {
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

void unalignedSSEe(float A[N][N], float b[N], float x[N]) {
    int n = N;
    //Gaussian elimination
    for (int k = 0; k < n; ++k) {
        int j;
        for (int i = k + 1; i < n; ++i) {
            float factor = A[i][k] / A[k][k];
        }
        A[k][k] = 1.0f;
        for (int i = k+1; i < n; ++i) {
            __m128 factor = _mm_set1_ps(A[i][k]);
            for (j = k; j+4<= n; j += 4) {
                __m128 row_k = _mm_loadu_ps(&A[k][j]);
                __m128 row_i = _mm_loadu_ps(&A[i][j]);
                __m128 prod = _mm_mul_ps(factor, row_k);
                row_i = _mm_sub_ps(row_i, prod);
                _mm_storeu_ps(&A[i][j], row_i);
            }
            for (; j < n; ++j) { // Handle tail case
                A[i][j] -= A[i][k] * A[k][j];
            }
            // Update vector b
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
void unalignedSSEelimination(float A[N][N], float b[N], float x[N]) {
    int n = N;

    // Gaussian elimination
    for (int k = 0; k < n; ++k) {
        int j;
        __m128 t1 = _mm_set_ps1(A[k][k]);
        __m128 factor;
        for (j = k + 1; j + 4 <= N; j += 4) {
            __m128 t2 = _mm_loadu_ps(&A[k][j]);
            factor = _mm_div_ps(t2, t1);
            _mm_storeu_ps(&A[k][j], factor);
        }
        for (; j < n; j++) { // Explicit tail handling
            float factor = A[k][j] / A[k][k];
            A[k][j] = factor;
            //A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0f;
        for (int i = k + 1; i < n; ++i) {
            __m128 factor = _mm_set_ps1(A[i][k] / A[k][k]);
            for (j = k; j + 4 <= n; j += 4) {
                __m128 row_k = _mm_loadu_ps(&A[k][j]);
                __m128 row_i = _mm_loadu_ps(&A[i][j]);
                __m128 prod = _mm_mul_ps(factor, row_k);
                row_i = _mm_sub_ps(row_i, prod);
                _mm_storeu_ps(&A[i][j], row_i);
            }
            for (; j < n; ++j) { // Handle tail case
                A[i][j] -= A[i][k] * A[k][j];
            }
            // Update vector b
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

void unalignedSSEback(float A[N][N], float b[N], float x[N]) {
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
        __m128 sum_vec = _mm_setzero_ps(); // Initialize sum as zero
        float sum = b[i];
        int j;
        for (j = i + 1; j + 4 <= n; j += 4) {
            __m128 a_vec = _mm_loadu_ps(&A[i][j]);
            __m128 x_vec = _mm_loadu_ps(&x[j]);
            __m128 prod = _mm_mul_ps(a_vec, x_vec);
            sum_vec = _mm_add_ps(sum_vec, prod);
        }
        for (; j < n; ++j) { // Handle tail case
            sum -= A[i][j] * x[j];
        }
        sum -= _mm_cvtss_f32(_mm_hadd_ps(sum_vec, sum_vec)); 
        sum -= _mm_cvtss_f32(_mm_hadd_ps(sum_vec, sum_vec)); 
        x[i] = sum / A[i][i];
    }
}

void alignedSSE(float A[N][N], float b[N], float x[N]) {
    __m128 t1, t2;
    __m128 factor = _mm_setzero_ps();
    for (int k = 0; k < N; k++) {
        int j;
        if (A[k][k] == 0) continue;
        t1 = _mm_set_ps1(A[k][k]);
        long long mkaddr = (long long)(&A[k][k]);
        int offset = (mkaddr % 16) / 4;
        int start = k + (4 - offset) % 4;
        int n = N - (N - start) % 4;
        if (start >= N) {
            start = N;  
        }
        //处理头和尾部
            for (j = start; j + 4 <= N; j += 4) {
                __m128 t2 = _mm_loadu_ps(&A[k][j]);
                factor = _mm_div_ps(t2, t1);
                _mm_storeu_ps(&A[k][j], factor);
            }
            //for (; j < n; j++) { // Explicit tail handling
            //    float factor = A[k][j] / A[k][k];
            //    A[k][j] = factor;
            //    //A[k][j] = A[k][j] / A[k][k];
            //}
        for (int i = k; i < start; i++) {
            float factor = A[k][i] / A[k][k];
            A[k][i] = factor;
        }
        for (int i = n; i < N; i++) {
            float factor = A[k][i] / A[k][k];
            A[k][i] = factor;
        }
        /*for (int j = start; j < n; j += 4) {
            t2 = _mm_load_ps(A[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_store_ps(A[k] + j, t2);
        }*/
        A[k][k] = 1.0f;
        for (int i = k + 1; i < N; ++i) {
            __m128 factor = _mm_set_ps1(A[i][k] / A[k][k]);
            for (int j = k; j <start; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            for (int j = n; j < N; j++)
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            for (j = start; j+4<= N; j += 4) {
                __m128 row_k = _mm_load_ps(&A[k][j]);
                __m128 row_i = _mm_load_ps(&A[i][j]);
                __m128 prod = _mm_mul_ps(factor, row_k);
                row_i = _mm_sub_ps(row_i, prod);
                _mm_store_ps(&A[i][j], row_i);
            }
            //for (; j < N; ++j) { // Handle tail case
            //    A[i][j] -= A[i][k] * A[k][j];
            //}
            // Update vector b
            b[i] -= b[k] * A[i][k];
        }
    }
    x[N - 1] = b[N - 1] / A[N- 1][N - 1];
    for (int i = N - 2; i >= 0; --i) {
        __m128 sum_vec = _mm_setzero_ps(); 
        float sum = b[i];
        long long aiaddr = (long long)(&A[i][i + 1]);
        int offset = (aiaddr % 16) / 4;
        int start = i + 1 + (4 - offset) % 4;
        int n = N - (N - start) % 4;
        if (start >= N) {
            start = N;  
        }
        for (int j = i + 1; j <start; ++j) {
            sum -= A[i][j] * x[j];
        }
        for (int j = n; j < N; ++j) {
            sum -= A[i][j] * x[j];
        }
        
        for (int j = start; j + 4 <= n; j += 4) {
            __m128 a_vec = _mm_load_ps(&A[i][j]);
            __m128 x_vec = _mm_load_ps(&x[j]);
            __m128 prod = _mm_mul_ps(a_vec, x_vec);
            sum_vec = _mm_add_ps(sum_vec, prod);
        }


        sum -= _mm_cvtss_f32(_mm_hadd_ps(sum_vec, sum_vec)); // Horizontal add for sum_vec
        sum -= _mm_cvtss_f32(_mm_hadd_ps(sum_vec, sum_vec)); // Repeat to accumulate all elements
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

    return totalExecutionTime / numRuns; // Average execution time
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

    cout << "Timing unalignedsse: " << measureFunction(unalignedSSE, numExperiments) << " ms" << endl;
    //restore(a, A);
    //memcpy(b, backup_b, N * sizeof(float)); // Restore b

    ////cout << "Timing unalignedSSE: " << measureFunction(unalignedSSEdiv, numExperiments) << " ms" << endl;
    ////restore(a, A);
    ////memcpy(b, backup_b, N * sizeof(float)); // Restore b
    //////printMatrix(A);
    //cout << "Timing unalignedSSEEelimination: " << measureFunction(unalignedSSEelimination, numExperiments) << " ms" << endl;
    //restore(a, A);  // Restore A to its initial state before the next function
    //memcpy(b, backup_b, N * sizeof(float)); // Restore b

    //cout << "Timing unalignedSSEback: " << measureFunction(unalignedSSEback, numExperiments) << " ms" << endl;
    //restore(a, A);
    //memcpy(b, backup_b, N * sizeof(float)); // Restore b

    //cout << "Timing alignedsse: " << measureFunction(alignedSSE, numExperiments) << " ms" << endl;
    //printMatrix(A);
    //restore(a, A);
    //memcpy(b, backup_b, N * sizeof(float)); // Restore b
    ///*for (int i = 0; i < N; ++i) {
    //    cout << x[i] << endl;
    //}*/
    //cout << "Timing unalignedSSEdiv: " << measureFunction(unalignedSSEdiv, numExperiments) << " ms" << endl;
    //restore(a, A);
    //memcpy(b, backup_b, N * sizeof(float)); // Restore b
    //cout << "Timing unalignedSSEe: " << measureFunction(unalignedSSEe, numExperiments) << " ms" << endl;
    //restore(a, A);
    //memcpy(b, backup_b, N * sizeof(float)); // Restore b
    ////printMatrix(A);
    return 0;
}

