#define _CRT_SECURE_NO_WARNINGS
#define HAVE_STRUCT_TIMESPEC
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <unordered_map>
#include <windows.h>
#include <pthread.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2

using namespace std;

const int maxsize = 3000; // 3000*32=96000
const int maxrow = 3000;
const int numBasis = 90000;
const int numThreads = 8; // Number of threads to be used

long long head, tail, freq;

map<int, int*> basisMap;
map<int, int> firstElementMap;
map<int, int*> resultMap;

fstream RowFile; // 被消元行
fstream BasisFile; // 消元子
int gRows[maxrow][maxsize];
int gBasis[numBasis][maxsize];

void reset() {
    memset(gRows, 0, sizeof(gRows));
    memset(gBasis, 0, sizeof(gBasis));
    if (RowFile.is_open()) {
        RowFile.close();
    }
    if (BasisFile.is_open()) {
        BasisFile.close();
    }
    RowFile.open("D:\\Groebner\\测试样例4 矩阵列数1011，非零消元子539，被消元行263\\被消元行.txt", ios::in | ios::out);
    BasisFile.open("D:\\Groebner\\测试样例4 矩阵列数1011，非零消元子539，被消元行263\\消元子.txt", ios::in | ios::out);
    basisMap.clear();
    firstElementMap.clear();
    resultMap.clear();
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
        reset();
    }

    return totalExecutionTime / numRuns;
}

void readBasis() {
    for (int i = 0; i < maxrow; i++) {
        if (BasisFile.eof()) {
            return;
        }
        string tmp;
        bool flag = false;
        int row = 0;
        getline(BasisFile, tmp);
        stringstream s(tmp);
        int pos;
        while (s >> pos) {
            if (!flag) {
                row = pos;
                flag = true;
                basisMap.insert(pair<int, int*>(row, gBasis[row]));
            }
            int index = pos / 32; // 基地址
            int offset = pos % 32; // 偏移量
            gBasis[row][index] = gBasis[row][index] | (1 << offset);
        }
        flag = false;
        row = 0;
    }
}

int readRows(int pos) {
    firstElementMap.clear();
    if (RowFile.is_open())
        RowFile.close();
    RowFile.open("D:\\Groebner\\测试样例4 矩阵列数1011，非零消元子539，被消元行263\\被消元行.txt", ios::in | ios::out);
    memset(gRows, 0, sizeof(gRows));
    string line;
    for (int i = 0; i < pos; i++) {
        getline(RowFile, line);
    }
    for (int i = pos; i < pos + maxsize; i++) {
        int tmp;
        getline(RowFile, line);
        if (line.empty()) {
            return i;
        }
        bool flag = false;
        stringstream s(line);
        while (s >> tmp) {
            if (!flag) {
                firstElementMap.insert(pair<int, int>(i - pos, tmp));
            }
            int index = tmp / 32;
            int offset = tmp % 32;
            gRows[i - pos][index] = gRows[i - pos][index] | (1 << offset);
            flag = true;
        }
    }
    return -1;
}

void updateFirstElement(int row) {
    bool flag = 0;
    for (int i = maxsize - 1; i >= 0; i--) {
        if (gRows[row][i] == 0)
            continue;
        else {
            if (!flag)
                flag = true;
            int pos = i * 32;
            int offset = 0;
            for (int k = 31; k >= 0; k--) {
                if (gRows[row][i] & (1 << k)) {
                    offset = k;
                    break;
                }
            }
            int newfirst = pos + offset;
            firstElementMap.erase(row);
            firstElementMap.insert(pair<int, int>(row, newfirst));
            break;
        }
    }
    if (!flag) {
        firstElementMap.erase(row);
    }
    return;
}
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
struct ThreadData {
    int thread_id;
    int start;
    int end;
};
void Special_Gauss_SERIAL() {
    int begin = 0;
    int flag;
    while (true) {
        flag = readRows(begin);
        int num = (flag == -1) ? maxsize : flag;
        for (int i = 0; i < num; i++) {
            while (firstElementMap.find(i) != firstElementMap.end()) {
                int first = firstElementMap.find(i)->second;
                if (basisMap.find(first) != basisMap.end()) {
                    int* basis = basisMap.find(first)->second;
                    for (int j = 0; j < maxsize; j++) {
                        gRows[i][j] = gRows[i][j] ^ basis[j];

                    }
                    updateFirstElement(i);
                }
                else {
                    for (int j = 0; j < maxsize; j++) {
                        gBasis[first][j] = gRows[i][j];
                    }
                    basisMap.insert(pair<int, int*>(first, gBasis[first]));
                    resultMap.insert(pair<int, int*>(first, gBasis[first]));
                    firstElementMap.erase(i);
                }
            }
        }
        if (flag == -1)
            begin += maxsize;
        else
            break;
    }
}


void* gaussEliminationThread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int start = data->start;
    int end = data->end;
    for (int i = start; i < end; i++) {
        while (true) {
            pthread_mutex_lock(&mutex);
            auto it = firstElementMap.find(i);
            if (it == firstElementMap.end()) {
                pthread_mutex_unlock(&mutex);
                break;
            }
            int first = it->second;
            auto basisIt = basisMap.find(first);
            if (basisIt != basisMap.end()) {
                int* basis = basisIt->second;
                pthread_mutex_unlock(&mutex);
                for (int j = 0; j < maxsize; j++) {
                    gRows[i][j] = gRows[i][j] ^ basis[j];
                }
                updateFirstElement(i);
            }
            else {
                for (int j = 0; j < maxsize; j++) {
                    gBasis[first][j] = gRows[i][j];
                }
                basisMap[first] = gBasis[first];
                resultMap[first] = gBasis[first];
                firstElementMap.erase(i);
                pthread_mutex_unlock(&mutex);
                break;
            }
        }
    }

    pthread_exit(nullptr);
    return 0;
}


void Special_Gauss_PARALLEL() {
    int begin = 0;
    int flag;
    const int NUM_THREADS = 8; 
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    while (true) {
        flag = readRows(begin);
        int num = (flag == -1) ? maxsize : flag;
        int chunk_size = num / NUM_THREADS;

        for (int i = 0; i < NUM_THREADS; i++) {
            thread_data[i].thread_id = i;
            thread_data[i].start = i * chunk_size;
            thread_data[i].end = (i == NUM_THREADS - 1) ? num : (i + 1) * chunk_size;
            pthread_create(&threads[i], nullptr, gaussEliminationThread, (void*)&thread_data[i]);
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], nullptr);
        }

        if (flag == -1)
            begin += maxsize;
        else
            break;
    }

    pthread_mutex_destroy(&mutex);
}

void* gaussEliminationThread_AVX(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int start = data->start;
    int end = data->end;
    for (int i = start; i < end; i++) {
        while (true) {
            pthread_mutex_lock(&mutex);
            auto it = firstElementMap.find(i);
            if (it == firstElementMap.end()) {
                pthread_mutex_unlock(&mutex);
                break;
            }
            int first = it->second;
            auto basisIt = basisMap.find(first);
            if (basisIt != basisMap.end()) {
                int* basis = basisIt->second;
                pthread_mutex_unlock(&mutex);

                // 使用SSE进行按位异或操作
                for (int j = 0; j < maxsize; j += 8) {
                    __m256i row_vec = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
                    __m256i basis_vec = _mm256_loadu_si256((__m256i*) & basis[j]);
                    __m256i result_vec = _mm256_xor_si256(row_vec, basis_vec);
                    _mm256_storeu_si256((__m256i*) & gRows[i][j], result_vec);
                }

                updateFirstElement(i);
            }
            else {
                for (int j = 0; j < maxsize; j++) {
                    gBasis[first][j] = gRows[i][j];
                }
                basisMap[first] = gBasis[first];
                resultMap[first] = gBasis[first];
                firstElementMap.erase(i);
                pthread_mutex_unlock(&mutex);
                break;
            }
        }
    }

    pthread_exit(nullptr);
    return nullptr;
}

void Special_Gauss_PARALLEL_AVX() {
    int begin = 0;
    int flag;
    const int NUM_THREADS = 8;
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    while (true) {
        flag = readRows(begin);
        int num = (flag == -1) ? maxsize : flag;
        int chunk_size = (num + NUM_THREADS - 1) / NUM_THREADS;  // 确保每个线程处理的范围不超出数组边界

        for (int i = 0; i < NUM_THREADS; i++) {
            thread_data[i].thread_id = i;
            thread_data[i].start = i * chunk_size;
            thread_data[i].end = min((i + 1) * chunk_size, num);
            pthread_create(&threads[i], nullptr, gaussEliminationThread_AVX, (void*)&thread_data[i]);
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], nullptr);
        }

        if (flag == -1)
            begin += maxsize;
        else
            break;
    }

    pthread_mutex_destroy(&mutex);
}

void writeResult(ofstream& out) {
    std::map<int, int*> sortedResultMap(resultMap.begin(), resultMap.end());

    for (auto it = sortedResultMap.rbegin(); it != sortedResultMap.rend(); it++) {
                int* result = it->second;
                int max = it->first / 32 + 1;
                for (int i = max; i >= 0; i--) {
                    if (result[i] == 0)
                        continue;
                    int pos = i * 32;
                    for (int k = 31; k >= 0; k--) {
                        if (result[i] & (1 << k)) {
                            out << k + pos << " ";
                        }
                    }
                }
                out << endl;
            }
        }
        

        int main() {
            //ofstream out("1.txt");
            int numExperiments = 10;
            readBasis();
            //cout << "Timing Special SERIAL: " << measureFunction(Special_Gauss_SERIAL, numExperiments) << " ms" << endl;
            //reset();
            cout << "Timing Special Parallel: " << measureFunction(Special_Gauss_PARALLEL, numExperiments) << " ms" << endl;
            //cout << "Timing Special Parallel_AVX: " << measureFunction(Special_Gauss_PARALLEL_AVX, numExperiments) << " ms" << endl;
            //writeResult(out);
            return 0;
        }


