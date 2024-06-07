#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <windows.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
using namespace std;

#define NUM_THREADS 8

const int maxsize = 3000;
const int maxrow = 40000; //3000*32>90000 ,最多存贮列数90000的被消元行矩阵60000行
const int numBasis = 40000;   //最多存储90000*100000的消元子
int num;

vector<int> tmpAns;

long long head, tail, freq;

map<int, int*> ans; // 答案

int gRows[maxrow][maxsize];   // 被消元行最多60000行，3000列
int gBasis[numBasis][maxsize];  // 消元子最多40000行，3000列
int answers[maxrow][maxsize]; // 存储消元完毕的行
map<int, int> firstToRow; // 记录answers的每行和首项的对应关系

int ifBasis[numBasis] = { 0 };
int ifDone[maxrow] = { 0 };

vector<string> rowFileBuffer;
vector<string> basisFileBuffer;

void reset() {
    memset(gRows, 0, sizeof(gRows));
    memset(gBasis, 0, sizeof(gBasis));
    memset(ifBasis, 0, sizeof(ifBasis));
    rowFileBuffer.clear();
    basisFileBuffer.clear();
    ans.clear();
}

void readFilesToMemory() {
    ifstream rowFile("D:\\Groebner\\测试样例10 矩阵列数43577，非零消元子39477，被消元行54274\\被消元行.txt");
    ifstream basisFile("D:\\Groebner\\测试样例10 矩阵列数43577，非零消元子39477，被消元行54274\\消元子.txt");
    string line;
    while (getline(rowFile, line)) {
        rowFileBuffer.push_back(line);
    }
    while (getline(basisFile, line)) {
        basisFileBuffer.push_back(line);
    }
}

void readBasis() { // 读取消元子
    for (const auto& line : basisFileBuffer) {
        stringstream s(line);
        int pos, row = -1;
        while (s >> pos) {
            if (row == -1) {
                row = pos;
                ifBasis[row] = 1;
            }
            int index = pos / 32;
            int offset = pos % 32;
            gBasis[row][index] |= (1 << offset);
        }
    }
}

int readRowsFrom(int pos) { // 读取被消元行
    memset(gRows, 0, sizeof(gRows)); // 重置为0
    for (int i = pos; i < pos + maxrow && i < rowFileBuffer.size(); i++) {
        stringstream s(rowFileBuffer[i]);
        int tmp;
        while (s >> tmp) {
            int index = tmp / 32;
            int offset = tmp % 32;
            gRows[i - pos][index] |= (1 << offset);
        }
    }
    return (pos + maxrow < rowFileBuffer.size()) ? maxrow : rowFileBuffer.size() - pos;
}

int findfirst(int row) { // 寻找第row行被消元行的首项
    for (int i = maxsize - 1; i >= 0; i--) {
        if (gRows[row][i] != 0) {
            int pos = i * 32;
            for (int k = 31; k >= 0; k--) {
                if (gRows[row][i] & (1 << k)) {
                    return pos + k;
                }
            }
        }
    }
    return -1;
}

int _findfirst(int row) { // 寻找answers的首项
    for (int i = maxsize - 1; i >= 0; i--) {
        if (answers[row][i] != 0) {
            int pos = i * 32;
            for (int k = 31; k >= 0; k--) {
                if (answers[row][i] & (1 << k)) {
                    return pos + k;
                }
            }
        }
    }
    return -1;
}

void writeResult(ofstream& out) {
    for (auto it = ans.rbegin(); it != ans.rend(); it++) {
        int* result = it->second;
        for (int i = maxsize - 1; i >= 0; i--) {
            if (result[i] != 0) {
                int pos = i * 32;
                for (int k = 31; k >= 0; k--) {
                    if (result[i] & (1 << k)) {
                        out << k + pos << " ";
                    }
                }
            }
        }
        out << endl;
    }
}

void Special_MPI(int argc, char* argv[]) {
    int flag;
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int begin = 0, end = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        flag = readRowsFrom(0); // 读取被消元行
        num = (flag == -1) ? maxrow : flag;
        begin = rank * num / total;
        end = (rank == total - 1) ? num : (rank + 1) * (num / total);
        for (int i = 1; i < total; i++) {
            MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD); // 0是被消元行行数
            int b = i * (num / total);
            int e = (i == total - 1) ? num : (i + 1) * (num / total);
            for (int j = b; j < e; j++) {
                MPI_Send(gRows[j], maxsize, MPI_INT, i, 1, MPI_COMM_WORLD); // 1是被消元行数据
            }
        }
    }
    else {
        MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        begin = rank * (num / total);
        end = (rank == total - 1) ? num : (rank + 1) * (num / total);
        for (int i = begin; i < end; i++) {
            MPI_Recv(gRows[i], maxsize, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // 此时每个进程都拿到了数据
    start_time = MPI_Wtime();
    for (int i = begin; i < end; i++) {
        int first = findfirst(i);
        while (first != -1) { // 未消元完毕，存在首项
            if (ifBasis[first] == 1) { // 存在首项为first消元子
                for (int j = 0; j < maxsize; j++) {
                    gRows[i][j] ^= gBasis[first][j]; // 进行异或消元
                }
                first = findfirst(i);
            }
            else { // 升级为消元子
                tmpAns.push_back(first);
                if (rank == 0) {
                    for (int j = 0; j < maxsize; j++) {
                        gBasis[first][j] = gRows[i][j];
                        answers[i][j] = gRows[i][j];
                    }
                    ifBasis[first] = 1; // 仅仅将0号进程消元到底
                }
                break;
            }
        }
        if (first == -1)
            tmpAns.push_back(-1);
    }

    for (int i = 0; i < rank; i++) {
        int b = i * (num / total);
        int e = b + num / total;
        for (int j = b; j < e; j++) {
            MPI_Recv(answers[j], maxsize, MPI_INT, i, 2, MPI_COMM_WORLD, &status); // 接收来自进程i的消元结果，可能作为之后的消元子
            int first = _findfirst(j);
            firstToRow.insert(pair<int, int>(first, j)); // 记录下首项信息
        }
        for (int j = begin; j < end; j++) { // 非0进程要进行二次消元，以此前进程的结果作为消元子
            int first = tmpAns.at(j - begin);
            if (first == -1)
                continue;
            while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) { // 存在可消元项
                if (firstToRow.find(first) != firstToRow.end()) { // 消元结果有消元子
                    int row = firstToRow.find(first)->second;
                    for (int k = 0; k < maxsize; k++) {
                        gRows[j][k] ^= answers[row][k];
                    }
                    first = findfirst(j);
                }
                if (first == -1)
                    break;
                if (ifBasis[first] == 1) {
                    for (int k = 0; k < maxsize; k++) {
                        gRows[j][k] ^= gBasis[first][k]; // 进行异或消元
                    }
                    first = findfirst(j);
                }
            }
        }
    }
    if (rank != 0) {
        for (int i = begin; i < end; i++) {
            int first = findfirst(i);
            if (first == -1)
                continue;
            while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) { // 存在可消元项
                if (firstToRow.find(first) != firstToRow.end()) { // 消元结果有消元子
                    int row = firstToRow.find(first)->second;
                    for (int k = 0; k < maxsize; k++) {
                        gRows[i][k] ^= answers[row][k];
                    }
                    first = findfirst(i);
                }
                if (first == -1)
                    break;
                if (ifBasis[first] == 1) {
                    for (int k = 0; k < maxsize; k++) {
                        gRows[i][k] ^= gBasis[first][k]; // 进行异或消元
                    }
                    first = findfirst(i);
                }
            }
            for (int j = 0; j < maxsize; j++) {
                gBasis[first][j] = gRows[i][j];
                answers[i][j] = gRows[i][j]; // 自身进程的消元结果不会加入firstToRow
            }
            ifBasis[first] = 1;
        }
    }

    for (int i = rank + 1; i < total; i++) {
        for (int j = begin; j < end; j++) {
            MPI_Send(answers[j], maxsize, MPI_INT, i, 2, MPI_COMM_WORLD); // 2是该进程的消元结果，可能作为之后进程的消元子
        }
    }

    if (rank == total - 1) {
        end_time = MPI_Wtime();
        cout << "MPI耗时： " << 1000 * (end_time - start_time) << "ms" << endl;
        ofstream out_mpi("消元结果(MPI).txt");
        writeResult(out_mpi);
        out_mpi.close();
    }
    MPI_Finalize();
}
void Special_MPI_AVX_omp(int argc, char* argv[]) {
	int flag;
	double start_time = 0;
	double end_time = 0;
	MPI_Init(&argc, &argv);
	int total = 0;
	int rank = 0;
	int i = 0;
	int j = 0;
	int begin = 0, end = 0;
	MPI_Status status;
	MPI_Comm_size(MPI_COMM_WORLD, &total);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		flag = readRowsFrom(0);     //读取被消元行
		num = (flag == -1) ? maxrow : flag;
		begin = rank * num / total;
		end = (rank == total - 1) ? num : (rank + 1) * (num / total);
		for (i = 1; i < total; i++) {
			MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);//0是被消元行行数
			int b = i * (num / total);
			int e = (i == total - 1) ? num : (i + 1) * (num / total);
			for (j = b; j < e; j++) {
				MPI_Send(&gRows[j][0], maxsize, MPI_INT, i, 1, MPI_COMM_WORLD);//1时被消元行数据
			}
		}

	}
	else {
		MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		begin = rank * (num / total);
		end = (rank == total - 1) ? num : (rank + 1) * (num / total);
		for (i = begin; i < end; i++) {
			MPI_Recv(&gRows[i][0], maxsize, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);  //此时每个进程都拿到了数据
	start_time = MPI_Wtime();
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j)
#pragma omp for ordered schedule(guided)
	for (i = begin; i < end; i++) {
		int first = findfirst(i);
		while (first != -1) {     //未消元完毕，存在首项
			if (ifBasis[first] == 1) {  //存在首项为first消元子
				for (j = 0; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
				first = findfirst(i);
			}
			else {   //升级为消元子
#pragma omp ordered
				if (rank == 0) {
					while (ifBasis[first] == 1) {
						for (j = 0; j + 8 < maxsize; j += 8) {
							__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
							__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][j]);
							__m256i vx = _mm256_xor_si256(vij, vj);
							_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
						}
						for (; j < maxsize; j++) {
							gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
						}
						first = findfirst(i);
					}
					if (first != -1) {
						for (j = 0; j + 8 < maxsize; j += 8) {
							__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
							_mm256_storeu_si256((__m256i*) & gBasis[first][j], vij);
							_mm256_storeu_si256((__m256i*) & answers[i][j], vij);
						}
						for (; j < maxsize; j++) {
							gBasis[first][j] = gRows[i][j];
							answers[i][j] = gRows[i][j];
						}
						ifBasis[first] = 1;  //仅仅将0号进程消元到底
					}
				}
				first = -1;
			}
		}
	}
	for (i = 0; i < rank; i++) {

		int b = i * (num / total);
		int e = b + num / total;
		for (j = b; j < e; j++) {
			MPI_Recv(&answers[j][0], maxsize, MPI_INT, i, 2, MPI_COMM_WORLD, &status);//接收来自进程i的消元结果，可能作为之后的消元子
			int first = _findfirst(j);
			firstToRow.insert(pair<int, int>(first, j));//记录下首项信息
		}
#pragma omp for schedule(guided)
		for (j = begin; j < end; j++) {  //非0进程要进行二次消元，以此前进程的结果作为消元子
			int first = findfirst(j);
			while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) {  //存在可消元项
				if (firstToRow.find(first) != firstToRow.end()) {  //消元结果有消元子
					int row = firstToRow.find(first)->second;
					int k = 0;
					for (k = 0; k + 8 < maxsize; k += 8) {
						__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[j][k]);
						__m256i vj = _mm256_loadu_si256((__m256i*) & answers[row][k]);
						__m256i vx = _mm256_xor_si256(vij, vj);
						_mm256_storeu_si256((__m256i*) & gRows[j][k], vx);
					}
					for (; k < maxsize; k++) {
						gRows[j][k] = gRows[j][k] ^ answers[row][k];
					}
					first = findfirst(i);
				}
				if (ifBasis[first] == 1) {
					int k = 0;
					for (k = 0; k + 8 < maxsize; k += 8) {
						__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[j][k]);
						__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][k]);
						__m256i vx = _mm256_xor_si256(vij, vj);
						_mm256_storeu_si256((__m256i*) & gRows[j][k], vx);
					}
					for (; k < maxsize; k++) {
						gRows[j][k] = gRows[j][k] ^ gBasis[first][k];
					}
					first = findfirst(i);
				}
			}

		}
	}

	if (rank != 0) {
		for (i = begin; i < end; i++) {
			int first = findfirst(i);
			if (first != -1) {
				while ((firstToRow.find(first) != firstToRow.end() || ifBasis[first] == 1) && first != -1) {  //存在可消元项
					if (firstToRow.find(first) != firstToRow.end()) {  //消元结果有消元子
						int row = firstToRow.find(first)->second;
						int k = 0;
						for (k = 0; k + 8 < maxsize; k += 8) {
							__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][k]);
							__m256i vj = _mm256_loadu_si256((__m256i*) & answers[row][k]);
							__m256i vx = _mm256_xor_si256(vij, vj);
							_mm256_storeu_si256((__m256i*) & gRows[i][k], vx);
						}
						for (; k < maxsize; k++) {
							gRows[i][k] = gRows[i][k] ^ answers[row][k];
						}
						first = findfirst(i);
					}
					if (ifBasis[first] == 1) {
						int k = 0;
						for (k = 0; k + 8 < maxsize; k += 8) {
							__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][k]);
							__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][k]);
							__m256i vx = _mm256_xor_si256(vij, vj);
							_mm256_storeu_si256((__m256i*) & gRows[i][k], vx);
						}
						for (; k < maxsize; k++) {
							gRows[i][k] = gRows[i][k] ^ gBasis[first][k];
						}

						first = findfirst(i);
					}
				}
				if (first == -1) {
					continue;
				}
				for (j = 0; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
					answers[i][j] = gRows[i][j];  //自身进程的消元结果不会加入firstToRow
				}
				ifBasis[first] = 1;
			}

		}

	}
	for (i = rank + 1; i < total; i++) {
		for (j = begin; j < end; j++) {

			MPI_Send(&answers[j][0], maxsize, MPI_INT, i, 2, MPI_COMM_WORLD);//2是该进程的消元结果，可能作为之后进程的消元子
		}
	}

	if (rank == total - 1) {
		end_time = MPI_Wtime();
		cout << "MPI+omp+AVX优化版本耗时： " << 1000 * (end_time - start_time) << "ms" << endl;
	}
	MPI_Finalize();

}
int main(int argc, char* argv[]) {
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    readFilesToMemory();
    readBasis();
    //Special_MPI(argc, argv);
	Special_MPI_AVX_omp(argc, argv);
    cout << "done!" << endl;

    return 0;
}