#include <iostream>
#include <windows.h>
#include <cstdlib>
#include <ctime>

using namespace std;
const int N = 5;
float A[N][N];
float a[N][N];
float b[N];
float x[N];

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

double measureFunction(void(*func)(float[][N], float[N], float[N]), int numRuns) {
    LARGE_INTEGER start, end, freq;
    QueryPerformanceFrequency(&freq);

    double totalExecutionTime = 0.0;

    for (int i = 0; i < numRuns; ++i) {
        QueryPerformanceCounter(&start);

        func(A,b,x);

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
    int numExperiments = 10;

    cout << "Timing serial: " << measureFunction(gauss, numExperiments) << " ms" << endl;
   // printMatrix(A);
    /*for (int i = 0; i < N; ++i) {
        cout << x[i] << endl;
    }*/
    return 0;
}
