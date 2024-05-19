#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<map>
#include<windows.h>
#include <nmmintrin.h> //SSSE4.2
#include <emmintrin.h>  // SSE2 
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2

using namespace std;

const int maxsize = 3000;//3000*32=96000
const int maxrow =3000;
const int numBasis =90000;

long long head, tail, freq;

map<int, int*> basisMap;
map<int, int> firstElementMap;
map<int, int*> resultMap;

fstream RowFile;//被消元行
fstream BasisFile;//消元子


alignas(16) int gRows[maxrow][maxsize];
alignas(16) int gBasis[numBasis][maxsize];
double measureFunction(void(*func)(), int numRuns) {
	LARGE_INTEGER start, end, freq;
	QueryPerformanceFrequency(&freq);

	double totalExecutionTime = 0.0;

	for (int i = 0; i < numRuns; ++i) {
		QueryPerformanceCounter(&start);

		func();

		QueryPerformanceCounter(&end);
		totalExecutionTime += static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	}

	return totalExecutionTime / numRuns;
}

void reset() {
	memset(gRows, 0, sizeof(gRows));
	memset(gBasis, 0, sizeof(gBasis));
	RowFile.close();
	BasisFile.close();
	RowFile.open("D:\\Groebner\\测试样例4 矩阵列数1011，非零消元子539，被消元行263\\被消元行.txt", ios::in | ios::out);
	BasisFile.open("D:\\Groebner\\测试样例4 矩阵列数1011，非零消元子539，被消元行263\\消元子.txt", ios::in | ios::out);
	basisMap.clear();
	firstElementMap.clear();
	resultMap.clear();
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
			int index = pos / 32;//基地址
			int offset = pos % 32;//偏移量
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
			cout << "End of File!" << endl;
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
				if (gRows[row][i] & (1 << k))
				{
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

void SSE() {
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
					int j = 0;
					for (; j + 4 <= maxsize; j += 4) {
						__m128i vRow = _mm_loadu_si128((__m128i*) & gRows[i][j]);
						__m128i vBasis = _mm_loadu_si128((__m128i*) & basis[j]);
						__m128i vXor = _mm_xor_si128(vRow, vBasis);
						_mm_storeu_si128((__m128i*) & gRows[i][j], vXor);
					}
					// 处理剩余未对齐的部分
					for (; j < maxsize; j++) {
						gRows[i][j] ^= basis[j];
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

int main() {
	//ofstream out("1.txt");
	int numExperiments = 1;
	readBasis();
	cout << "Timing Special Serial: " << measureFunction(SSE, numExperiments) << " ms" << endl;
	//writeResult(out);
	reset();
}

