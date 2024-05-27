#include<iostream>
#include<omp.h>
#include<windows.h>
#include <fstream>
#include <string>
#include <sstream>
#include <emmintrin.h>
#include <immintrin.h>
#include<map>
#include <unordered_map>

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
    RowFile.close();
    BasisFile.close();
    RowFile.open("D:\\Groebner\\测试样例5 矩阵列数2362，非零消元子1226，被消元行453\\被消元行.txt", ios::in | ios::out);
    BasisFile.open("D:\\Groebner\\测试样例5 矩阵列数2362，非零消元子1226，被消元行453\\消元子.txt", ios::in | ios::out);
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
    RowFile.open("D:\\Groebner\\测试样例5 矩阵列数2362，非零消元子1226，被消元行453\\被消元行.txt", ios::in | ios::out);
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
void Special_Gauss_PARALLEL() {
    int begin = 0;
    int flag;
    bool shouldBreak = false; // 使用标志控制循环退出

    while (true) {
        flag = readRows(begin);
        int num = (flag == -1) ? maxsize : flag;

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num; i++) {
            while (true) {
                bool found = false;
#pragma omp critical(maplookup1)
                {
                    if (firstElementMap.find(i) == firstElementMap.end()) {
                        found = true;
                    }
                }
                if (found) break; // 根据标志位决定是否跳出循环

                int first;
#pragma omp critical(maplookup2)
                {
                    first = firstElementMap.find(i)->second;
                }

#pragma omp critical(update1)
                {
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
#pragma omp critical(update2)
                        {
                            basisMap.insert(std::pair<int, int*>(first, gBasis[first]));
                            resultMap.insert(std::pair<int, int*>(first, gBasis[first]));
                            firstElementMap.erase(i);
                        }
                    }
                }
            }
        }

        if (flag == -1)
            begin += maxsize;
        else
            break;
    }
}
void Special_Gauss_PARALLEL_AVX() {
    int begin = 0;
    int flag;
    bool shouldBreak = false; // 使用标志控制循环退出

    while (true) {
        flag = readRows(begin);
        int num = (flag == -1) ? maxsize : flag;

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num; i++) {
            while (true) {
                bool found = false;
#pragma omp critical(maplookup1)
                {
                    if (firstElementMap.find(i) == firstElementMap.end()) {
                        found = true;
                    }
                }
                if (found) break; // 根据标志位决定是否跳出循环

                int first;
#pragma omp critical(maplookup2)
                {
                    first = firstElementMap.find(i)->second;
                }

#pragma omp critical(update1)
                {
                    if (basisMap.find(first) != basisMap.end()) {
                        int* basis = basisMap.find(first)->second;
                        // 使用 AVX 加速进行计算
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
#pragma omp critical(update2)
                        {
                            basisMap.insert(std::pair<int, int*>(first, gBasis[first]));
                            resultMap.insert(std::pair<int, int*>(first, gBasis[first]));
                            firstElementMap.erase(i);
                        }
                    }
                }
            }
        }

        if (flag == -1)
            begin += maxsize;
        else
            break;
    }
}

void Special_Gauss_PARALLEL_SIMD() {
    int begin = 0;
    int flag;

    while (true) {
        flag = readRows(begin);
        int num = (flag == -1) ? maxsize : flag;

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num; i++) {
            while (true) {
                bool element_found = false;
                int first = -1;

#pragma omp critical
                {
                    if (firstElementMap.find(i) != firstElementMap.end()) {
                        first = firstElementMap.find(i)->second;
                        element_found = true;
                    }
                }

                if (!element_found) break;

                bool basis_found = false;
                int* basis = nullptr;

#pragma omp critical
                {
                    if (basisMap.find(first) != basisMap.end()) {
                        basis = basisMap.find(first)->second;
                        basis_found = true;
                    }
                }

                if (basis_found) {
#pragma omp simd
                    for (int j = 0; j < maxsize; j++) {
                        gRows[i][j] ^= basis[j];
                    }
                    updateFirstElement(i);
                }
                else {
#pragma omp simd
                    for (int j = 0; j < maxsize; j++) {
                        gBasis[first][j] = gRows[i][j];
                    }
#pragma omp critical
                    {
                        basisMap.insert(std::make_pair(first, gBasis[first]));
                        resultMap.insert(std::make_pair(first, gBasis[first]));
                        firstElementMap.erase(i);
                    }
                }
            }
        }

        if (flag == -1)
            begin += maxsize;
        else
            break;
    }
}

void writeResult(ofstream& out) {
    for (auto it = resultMap.rbegin(); it != resultMap.rend(); it++) {
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
int main() {
    //ofstream out("1.txt");
    int numExperiments = 10;
    readBasis();
    cout << "Timing Special SERIAL: " << measureFunction(Special_Gauss_SERIAL, numExperiments) << " ms" << endl;
    cout << "Timing Special Parallel: " << measureFunction(Special_Gauss_PARALLEL, numExperiments) << " ms" << endl;
    cout << "Timing Special Parallel_AVX: " << measureFunction(Special_Gauss_PARALLEL_AVX, numExperiments) << " ms" << endl;
    //cout << "Timing Special Parallel SIMD: " << measureFunction(Special_Gauss_PARALLEL_SIMD, numExperiments) << " ms" << endl;
    //writeResult(out);

    reset();
    return 0;
}