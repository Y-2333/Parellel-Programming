#define _CRT_SECURE_NO_WARNINGS
#define HAVE_STRUCT_TIMESPEC
#include<pthread.h>
#include <iostream>
#include<semaphore.h>
#include <emmintrin.h>
#include <immintrin.h>
#include<stdlib.h>
#include<windows.h>
using namespace std;
const int N = 1000;
float A[N][N];
float a[N][N];
const int numthreads = 8;

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
        t1 = _mm_set_ps1(A[k][k]); 

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
            m[i][j] = static_cast<float>(rand() % 1000);// Generate a random float between 0 and 1000
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

typedef struct {
    int k;      // 迭代的当前步
    int t_id;   // 线程id
} threadParam_t;

//消去部分
//水平划分线程函数
void* threadFunc_row(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;//消去轮次
    int t_id = p->t_id;//线程编号
    // 线程处理对应的行
    int i = k + t_id +1;  // 获取自己的计算任务
    for (int m = i; m < N;m+=numthreads){  // 确保线程处理的行不同
        for (int j = k + 1; j < N; j++) {
            A[m][j] -= A[m][k] * A[k][j];
        }
        A[m][k] = 0;//下三角设置成0
    }
    pthread_exit(NULL);
    return 0;
}

//动态线程
void dynamic_row() {
    for (int k = 0; k < N; k++) {
        // 主线程做除法操作
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        // 创建工作线程，进行消去操作
        int worker_count = numthreads; // 工作线程数量
        pthread_t* handles = new pthread_t[worker_count]; // 创建对应的线程句柄
        threadParam_t* params = new threadParam_t[worker_count]; // 创建对应的线程数据结构

        // 分配任务
        for (int t_id = 0; t_id < worker_count; t_id++) {
            params[t_id].k = k;
            params[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc_row, (void*)&params[t_id]);
        }

        // 主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }

        delete[] handles;
        delete[] params;
    }
}

//垂直划分线程函数
void* threadFunc_col(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    for (int j = k + 1 + t_id; j < N; j += numthreads) {
        for (int v = k + 1; v < N; v++) {
            A[v][j] = A[v][j] - A[v][k] * A[k][j];
        }
    }
    pthread_exit(NULL);
    return 0;
}

void dynamic_col() {
    for (int k = 0; k < N; k++) {
        // 主线程做除法操作
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        // 创建工作线程，进行消去操作
        int worker_count = numthreads; // 工作线程数量
        pthread_t* handles = new pthread_t[worker_count]; // 创建对应的线程句柄
        threadParam_t* params = new threadParam_t[worker_count]; // 创建对应的线程数据结构

        // 分配任务
        for (int t_id = 0; t_id < worker_count; t_id++) {
            params[t_id].k = k;
            params[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc_col, (void*)&params[t_id]);
        }

        // 主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }
        delete[] handles;
        delete[] params;
    }
    for (int i = 1; i < N; i++) {
        for (int j = 0; j < i; j++) {
            A[i][j] = 0.0;
        }
    }
}

//静态线程+信号同步量版本
typedef struct {
    int t_id; //线程 id
}threadParam_t1;

sem_t sem_main;
sem_t* sem_workerstart;
sem_t* sem_workerend;

void* threadFunc_static_row(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++) {
        sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作
        // 循环划分任务
        for (int i = k + 1 + t_id; i < N; i += numthreads) {
            // 消去
            for (int j = k + 1; j < N; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
        sem_post(&sem_main); // 唤醒主线程
        sem_wait(&sem_workerend[t_id]); // 阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_row() {
    //初始化信号量
    sem_init(&sem_main, 0, 0);
    sem_workerstart = new sem_t[numthreads];
    sem_workerend = new sem_t[numthreads];
    for (int i = 0; i < numthreads; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    //创建线程
    for (int t = 0; t < numthreads; t++) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_static_row, (void*)&params[t]);
    }
    for (int k = 0; k < N; ++k) {
        // 主线程除法操作
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        // 唤醒所有工作线程
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerstart[t]);
        }
        // 等待所有工作线程完成
        for (int t = 0; t < numthreads; t++) {
            sem_wait(&sem_main);
        }
        // 唤醒所有工作线程进入下一轮
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerend[t]);
        }
    }
    // 等待所有线程完成
    for (int t = 0; t < numthreads; t++) {
        pthread_join(threads[t], NULL);
    }
    // 销毁信号量
    sem_destroy(&sem_main);
    for (int i = 0; i < numthreads; i++) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }

}

void* threadFunc_static_col(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++) {
        sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作
        // 循环划分任务
        for (int j = k + t_id+1; j < N; j += numthreads){
            for (int i = k + 1; i < N; i ++) {
            // 消去
                A[i][j] -= A[i][k] * A[k][j];
            }
            //A[j][k] = 0.0;
        }
        sem_post(&sem_main); // 唤醒主线程
        sem_wait(&sem_workerend[t_id]); // 阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_col() {
    //初始化信号量
    sem_init(&sem_main, 0, 0);
    sem_workerstart = new sem_t[numthreads];
    sem_workerend = new sem_t[numthreads];
    for (int i = 0; i < numthreads; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    //创建线程
    for (int t = 0; t < numthreads; t++) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_static_col, (void*)&params[t]);
    }
    for (int k = 0; k < N; ++k) {
        // 主线程除法操作
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        // 唤醒所有工作线程
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerstart[t]);
        }
        // 等待所有工作线程完成
        for (int t = 0; t < numthreads; t++) {
            sem_wait(&sem_main);
        }
        // 唤醒所有工作线程进入下一轮
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerend[t]);
        }
    }
    // 等待所有线程完成
    for (int t = 0; t < numthreads; t++) {
        pthread_join(threads[t], NULL);
    }
    for (int i = 1; i < N; i++) {
        for (int j = 0; j < i; j++) {
            A[i][j] = 0.0;
        }
    }

    // 销毁信号量
    sem_destroy(&sem_main);
    for (int i = 0; i < numthreads; i++) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }

}


//静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数
sem_t sem_leader; // 领导线程信号量
sem_t sem_Division[numthreads - 1]; // 除法操作信号量
sem_t sem_Elimination[numthreads - 1]; // 消去操作信号量


void* threadFunc_sem_tri_row(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; ++k) {
        // t_id 为 0 的线程负责除法操作，其他线程先等待
        if (t_id == 0) {
            for (int j = k + 1; j < N; ++j) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
            for (int i = 0; i < numthreads - 1; ++i) {
                sem_post(&sem_Division[i]);
            }
        }
        else {
            sem_wait(&sem_Division[t_id - 1]); // 阻塞，等待完成除法操作
        }
        // 循环划分任务（采用多种任务划分方式）
        for (int i = k + 1 + t_id; i < N; i += numthreads) {
            for (int j = k + 1; j < N; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
        // t_id 为 0 的线程唤醒其他工作线程，进行消去操作
        if (t_id == 0) {
            for (int i = 0; i < numthreads - 1; i++) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < numthreads - 1; ++i) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]); // 阻塞，等待唤醒进行消去操作
        }
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_tri_row() {
    // 初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < numthreads - 1; ++i) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    // 创建线程
    for (int t = 0; t < numthreads; ++t) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_sem_tri_row, (void*)&params[t]);
    }
    // 等待所有线程完成
    for (int t = 0; t < numthreads; ++t) {
        pthread_join(threads[t], NULL);
    }
    sem_destroy(&sem_leader);
    for (int i = 0; i < numthreads - 1; ++i) {
        sem_destroy(&sem_Division[i]);
        sem_destroy(&sem_Elimination[i]);
    }
}


void* threadFunc_sem_tri_col(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; ++k) {
        // t_id 为 0 的线程负责除法操作，其他线程先等待
        if (t_id == 0) {
            for (int j = k + 1; j < N; ++j) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
            for (int i = 0; i < numthreads - 1; ++i) {
                sem_post(&sem_Division[i]);
            }
        }
        else {
            sem_wait(&sem_Division[t_id - 1]); // 阻塞，等待完成除法操作
        }
        // 循环划分任务（采用多种任务划分方式）
        for (int j = k + 1 + t_id; j < N; j += numthreads) {
            for (int i = k + 1; i< N; ++i) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            //A[i][k] = 0.0;
        }
        // t_id 为 0 的线程唤醒其他工作线程，进行消去操作
        if (t_id == 0) {
            for (int i = 0; i < numthreads - 1; i++) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < numthreads - 1; ++i) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]); // 阻塞，等待唤醒进行消去操作
        }
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_tri_col() {
    // 初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < numthreads - 1; ++i) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    // 创建线程
    for (int t = 0; t < numthreads; ++t) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_sem_tri_col, (void*)&params[t]);
    }
    // 等待所有线程完成
    for (int t = 0; t < numthreads; ++t) {
        pthread_join(threads[t], NULL);
    }
    for (int i = 1; i < N; i++) {
        for (int j = 0; j < i; j++) {
            A[i][j] = 0.0;
        }
    }
    sem_destroy(&sem_leader);
    for (int i = 0; i < numthreads - 1; ++i) {
        sem_destroy(&sem_Division[i]);
        sem_destroy(&sem_Elimination[i]);
    }
}

//静态线程 +barrier 同步

pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

void* threadFunc_barrier_tri_row(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;

    for (int k = 0; k <N; k++) {
        if (t_id == 0) {
            for (int j = k + 1; j <N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }

        // 第一个同步点
        pthread_barrier_wait(&barrier_Division);

        for (int i = k + 1 + t_id; i <N; i += numthreads) {
            for (int j = k + 1; j <N; ++j) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }

        // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }

    pthread_exit(NULL);
    return 0;
}
void static_barrier_tri_row() {
    pthread_barrier_init(&barrier_Division, NULL, numthreads);
    pthread_barrier_init(&barrier_Elimination, NULL, numthreads);
    pthread_t handles[numthreads];
    threadParam_t1 param[numthreads];
    for (int t_id = 0; t_id < numthreads; ++t_id) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_barrier_tri_row, (void*)&param[t_id]);
    }
    for (int t_id = 0; t_id < numthreads; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }
    // 销毁 barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
}

void* threadFunc_barrier_tri_col(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;

    for (int k = 0; k < N; k++) {
        if (t_id == 0) {
            for (int j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
        }

        // 第一个同步点
        pthread_barrier_wait(&barrier_Division);

        for (int j = k + 1 + t_id; j < N; j += numthreads) {
            for (int i = k + 1; i < N; ++i) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            //A[i][k] = 0.0;
        }

        // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }

    pthread_exit(NULL);
    return 0;
}

void static_barrier_tri_col() {
    pthread_barrier_init(&barrier_Division, NULL, numthreads);
    pthread_barrier_init(&barrier_Elimination, NULL, numthreads);
    pthread_t handles[numthreads];
    threadParam_t1 param[numthreads];
    for (int t_id = 0; t_id < numthreads; ++t_id) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_barrier_tri_col, (void*)&param[t_id]);
    }
    for (int t_id = 0; t_id < numthreads; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }
    for (int i = 1; i < N; i++) {
        for (int j = 0; j < i; j++) {
            A[i][j] = 0.0;
        }
    }
    // 销毁 barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
}

//动态线程，SSE
void* threadFunc_row_SSE(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    int v = k + t_id + 1;
    __m128 t1, t2;
    for (int i = v; i < N; i += numthreads) {
        t1 = _mm_set_ps1(A[i][k]);
        int j=k+1;
        for (; j <= N-4; j += 4) {//j + 4 <= N,原来这样就会把多余的不足4的倍数的也算进来
            t2 = _mm_loadu_ps(A[k] + j);
            __m128 t3 = _mm_loadu_ps(A[i] + j);
            t2 = _mm_mul_ps(t2, t1);
            t3 = _mm_sub_ps(t3, t2);
            _mm_storeu_ps(A[i] + j, t3);
        }
        for (; j <N; j++) {
            A[i][j] = A[i][j] - A[i][k] * A[k][j];
        }
        A[i][k] = 0.0;
    }
    pthread_exit(NULL);
    return 0;
}

void dynamic_row_SSE() {
    __m128 t1, t2;
    for (int k = 0; k < N; k++) {
        t1 = _mm_set_ps1(A[k][k]);
        int j=k+1;
        for (; j <=N-4; j += 4) {
            t2 = _mm_loadu_ps(A[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(A[k] + j, t2);
        }
        for (; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1;
        // 创建工作线程，进行消去操作
        int worker_count = numthreads; // 工作线程数量
        pthread_t* handles = new pthread_t[worker_count]; // 创建对应的线程句柄
        threadParam_t* params = new threadParam_t[worker_count]; // 创建对应的线程数据结构

        // 分配任务
        for (int t_id = 0; t_id < worker_count; t_id++) {
            params[t_id].k = k;
            params[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc_row_SSE, (void*)&params[t_id]);
        }

        // 主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }

        delete[] handles;
        delete[] params;
    }
}

//动态线程，AVX
void* threadFunc_row_AVX(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    int v = k + t_id + 1;
    __m256 t1, t2;
    for (int i = v; i < N; i += numthreads) {
        t1 = _mm256_set1_ps(A[i][k]);
        int j=k+1;
        for (int j = k + 1; j <= N - 8; j += 8) {//j + 4 <= N,原来这样就会把多余的不足4的倍数的也算进来
            t2 = _mm256_loadu_ps(A[k] + j);
            __m256 t3 = _mm256_loadu_ps(A[i] + j);
            t2 = _mm256_mul_ps(t2, t1);
            t3 = _mm256_sub_ps(t3, t2);
            _mm256_storeu_ps(A[i] + j, t3);
        }
        for (; j < N; j++) {
            A[i][j] = A[i][j] - A[i][k] * A[k][j];
        }
        A[i][k] = 0.0;
    }
    pthread_exit(NULL);
    return 0;
}

void dynamic_row_AVX() {
    __m256 t1, t2;
    for (int k = 0; k < N; k++) {
        t1 = _mm256_set1_ps(A[k][k]);
        int j=k+1;
        for (int j = k + 1; j <= N - 8; j += 8) {
            t2 = _mm256_loadu_ps(A[k] + j);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(A[k] + j, t2);
        }
        for (; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        // 创建工作线程，进行消去操作
        int worker_count = numthreads; // 工作线程数量
        pthread_t* handles = new pthread_t[worker_count]; // 创建对应的线程句柄
        threadParam_t* params = new threadParam_t[worker_count]; // 创建对应的线程数据结构

        // 分配任务
        for (int t_id = 0; t_id < worker_count; t_id++) {
            params[t_id].k = k;
            params[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc_row_AVX, (void*)&params[t_id]);
        }

        // 主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }

        delete[] handles;
        delete[] params;
    }
}

//静态线程，信号量，SSE
void* threadFunc_static_row_SSE(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++) {
        sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作
        __m128 t1, t2;
        // 循环划分任务
        for (int i = k + 1 + t_id; i < N; i += numthreads) {
            t1 = _mm_set_ps1(A[i][k]);
            // 消去
            int j=k+1;
            for (; j <=N-4; j+=4) {
                t2 = _mm_loadu_ps(A[k] + j);
                __m128 t3 = _mm_loadu_ps(A[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_storeu_ps(A[i] + j, t3);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
        sem_post(&sem_main); // 唤醒主线程
        sem_wait(&sem_workerend[t_id]); // 阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_row_SSE() {
    //初始化信号量
    sem_init(&sem_main, 0, 0);
    sem_workerstart = new sem_t[numthreads];
    sem_workerend = new sem_t[numthreads];
    for (int i = 0; i < numthreads; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    //创建线程
    for (int t = 0; t < numthreads; t++) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_static_row_SSE, (void*)&params[t]);
    }
    for (int k = 0; k < N; ++k) {
        __m128 t1, t2;
        t1 = _mm_set_ps1(A[k][k]);
        int j=k+1;
        // 主线程除法操作
        for (; j <=N-4; j+=4) {
            t2 = _mm_loadu_ps(A[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(A[k] + j, t2);
        }
        for (; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        // 唤醒所有工作线程
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerstart[t]);
        }
        // 等待所有工作线程完成
        for (int t = 0; t < numthreads; t++) {
            sem_wait(&sem_main);
        }
        // 唤醒所有工作线程进入下一轮
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerend[t]);
        }
    }
    // 等待所有线程完成
    for (int t = 0; t < numthreads; t++) {
        pthread_join(threads[t], NULL);
    }
    // 销毁信号量
    sem_destroy(&sem_main);
    for (int i = 0; i < numthreads; i++) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }
}

//静态线程，信号量，AVX
void* threadFunc_static_row_AVX(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++) {
        sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作
        __m256 t1, t2;
        // 循环划分任务
        for (int i = k + 1 + t_id; i < N; i += numthreads) {
            t1 = _mm256_set1_ps(A[i][k]);
            // 消去
            int j=k+1;
            for (int j = k + 1; j <= N - 8; j += 8) {
                t2 = _mm256_loadu_ps(A[k] + j);
                __m256 t3 = _mm256_loadu_ps(A[i] + j);
                t2 = _mm256_mul_ps(t2, t1);
                t3 = _mm256_sub_ps(t3, t2);
                _mm256_storeu_ps(A[i] + j, t3);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
        sem_post(&sem_main); // 唤醒主线程
        sem_wait(&sem_workerend[t_id]); // 阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_row_AVX() {
    //初始化信号量
    sem_init(&sem_main, 0, 0);
    sem_workerstart = new sem_t[numthreads];
    sem_workerend = new sem_t[numthreads];
    for (int i = 0; i < numthreads; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    //创建线程
    for (int t = 0; t < numthreads; t++) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_static_row_AVX, (void*)&params[t]);
    }
    __m256 t1, t2;
    for (int k = 0; k < N; k++) {
        t1 = _mm256_set1_ps(A[k][k]);
        int j=k+1;
        for (int j = k + 1; j <= N - 8; j += 8) {
            t2 = _mm256_loadu_ps(A[k] + j);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(A[k] + j, t2);
        }
        for (; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        // 唤醒所有工作线程
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerstart[t]);
        }
        // 等待所有工作线程完成
        for (int t = 0; t < numthreads; t++) {
            sem_wait(&sem_main);
        }
        // 唤醒所有工作线程进入下一轮
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerend[t]);
        }
    }
    // 等待所有线程完成
    for (int t = 0; t < numthreads; t++) {
        pthread_join(threads[t], NULL);
    }
    // 销毁信号量
    sem_destroy(&sem_main);
    for (int i = 0; i < numthreads; i++) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }
}

//静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数,SSE
void* threadFunc_sem_tri_row_SSE(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    __m128 t1, t2;
    for (int k = 0; k < N; ++k) {
        // t_id 为 0 的线程负责除法操作，其他线程先等待
        if (t_id == 0) {
            t1 = _mm_set_ps1(A[k][k]);
            int j=k+1;
            for (; j <=N-4; j+=4) {
                t2 = _mm_loadu_ps(A[k] + j);
                t2 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(A[k] + j, t2);
            }
            for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
            for (int i = 0; i < numthreads - 1; ++i) {
                sem_post(&sem_Division[i]);
            }
        }
        else {
            sem_wait(&sem_Division[t_id - 1]); // 阻塞，等待完成除法操作
        }
        // 循环划分任务（采用多种任务划分方式）
        for (int i = k + 1 + t_id; i < N; i += numthreads) {
            t1 = _mm_set_ps1(A[i][k]);
            int j=k+1;
            for (; j <=N-4; j+=4) {
                t2 = _mm_loadu_ps(A[k] + j);
                __m128 t3 = _mm_loadu_ps(A[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_storeu_ps(A[i] + j, t3);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
        // t_id 为 0 的线程唤醒其他工作线程，进行消去操作
        if (t_id == 0) {
            for (int i = 0; i < numthreads - 1; i++) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < numthreads - 1; ++i) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]); // 阻塞，等待唤醒进行消去操作
        }
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_tri_row_SSE() {
    // 初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < numthreads - 1; ++i) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    // 创建线程
    for (int t = 0; t < numthreads; ++t) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_sem_tri_row_SSE, (void*)&params[t]);
    }
    // 等待所有线程完成
    for (int t = 0; t < numthreads; ++t) {
        pthread_join(threads[t], NULL);
    }
    sem_destroy(&sem_leader);
    for (int i = 0; i < numthreads - 1; ++i) {
        sem_destroy(&sem_Division[i]);
        sem_destroy(&sem_Elimination[i]);
    }
}

void* threadFunc_sem_tri_row_AVX(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    __m256 t1, t2;
    for (int k = 0; k < N; ++k) {
        // t_id 为 0 的线程负责除法操作，其他线程先等待
        if (t_id == 0) {
            t1 = _mm256_set1_ps(A[k][k]);
            int j=k+1;
            for (int j = k + 1; j <= N - 8; j += 8) {
                t2 = _mm256_loadu_ps(A[k] + j);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(A[k] + j, t2);
            }
            for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
            for (int i = 0; i < numthreads - 1; ++i) {
                sem_post(&sem_Division[i]);
            }
        }
        else {
            sem_wait(&sem_Division[t_id - 1]); // 阻塞，等待完成除法操作
        }
        // 循环划分任务（采用多种任务划分方式）
        for (int i = k + 1 + t_id; i < N; i += numthreads) {
            t1 = _mm256_set1_ps(A[i][k]);
            int j=k+1;
            for (int j = k + 1; j <= N - 8; j += 8) {
                t2 = _mm256_loadu_ps(A[k] + j);
                __m256 t3 = _mm256_loadu_ps(A[i] + j);
                t2 = _mm256_mul_ps(t2, t1);
                t3 = _mm256_sub_ps(t3, t2);
                _mm256_storeu_ps(A[i] + j, t3);
            }
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
        // t_id 为 0 的线程唤醒其他工作线程，进行消去操作
        if (t_id == 0) {
            for (int i = 0; i < numthreads - 1; i++) {
                sem_wait(&sem_leader);
            }
            for (int i = 0; i < numthreads - 1; ++i) {
                sem_post(&sem_Elimination[i]);
            }
        }
        else {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]); // 阻塞，等待唤醒进行消去操作
        }
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_tri_row_AVX() {
    // 初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < numthreads - 1; ++i) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    // 创建线程
    for (int t = 0; t < numthreads; ++t) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_sem_tri_row_AVX, (void*)&params[t]);
    }
    // 等待所有线程完成
    for (int t = 0; t < numthreads; ++t) {
        pthread_join(threads[t], NULL);
    }
    sem_destroy(&sem_leader);
    for (int i = 0; i < numthreads - 1; ++i) {
        sem_destroy(&sem_Division[i]);
        sem_destroy(&sem_Elimination[i]);
    }
}

void* threadFunc_barrier_tri_row_SSE(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    __m128 t1, t2;
    for (int k = 0; k < N; k++) {
        if (t_id == 0) {
            t1 = _mm_set_ps1(A[k][k]);
            int j=0;
            for (int j = k + 1; j < N; j += 4) {
                t2 = _mm_loadu_ps(A[k] + j);
                t2 = _mm_div_ps(t2, t1);
                _mm_storeu_ps(A[k] + j, t2);
            }
            /*for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }*/
            A[k][k] = 1.0;
        }

        // 第一个同步点
        pthread_barrier_wait(&barrier_Division);

        for (int i = k + 1 + t_id; i < N; i += numthreads) {
            t1 = _mm_set_ps1(A[i][k]);
            int j=0;
            for (int j = k + 1; j < N; j += 4) {
                t2 = _mm_loadu_ps(A[k] + j);
                __m128 t3 = _mm_loadu_ps(A[i] + j);
                t2 = _mm_mul_ps(t2, t1);
                t3 = _mm_sub_ps(t3, t2);
                _mm_storeu_ps(A[i] + j, t3);
            }
            A[i][k] = 0.0;
        }

        // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }

    pthread_exit(NULL);
    return 0;
}

void static_barrier_tri_row_SSE() {
    pthread_barrier_init(&barrier_Division, NULL, numthreads);
    pthread_barrier_init(&barrier_Elimination, NULL, numthreads);
    pthread_t handles[numthreads];
    threadParam_t1 param[numthreads];
    for (int t_id = 0; t_id < numthreads; ++t_id) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_barrier_tri_row_SSE, (void*)&param[t_id]);
    }
    for (int t_id = 0; t_id < numthreads; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }
    // 销毁 barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
}


void* threadFunc_barrier_tri_row_AVX(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    __m256 t1, t2;
    for (int k = 0; k < N; k++) {
        if (t_id == 0) {
            t1 = _mm256_set1_ps(A[k][k]);
            int j = k + 1;
            for (; j <= N - 8; j += 8) {
                t2 = _mm256_loadu_ps(A[k] + j);
                t2 = _mm256_div_ps(t2, t1);
                _mm256_storeu_ps(A[k] + j, t2);
            }
            // 处理剩余的元素
            for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }

        // 第一个同步点
        pthread_barrier_wait(&barrier_Division);

        for (int i = k + 1 + t_id; i < N; i += numthreads) {
            t1 = _mm256_set1_ps(A[i][k]);
            int j = k + 1;
            for (; j <= N - 8; j += 8) {
                t2 = _mm256_loadu_ps(A[k] + j);
                __m256 t3 = _mm256_loadu_ps(A[i] + j);
                t2 = _mm256_mul_ps(t2, t1);
                t3 = _mm256_sub_ps(t3, t2);
                _mm256_storeu_ps(A[i] + j, t3);
            }
            // 处理剩余的元素
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }

        // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }

    pthread_exit(NULL);
    return NULL;
}
void static_barrier_tri_row_AVX() {
    pthread_barrier_init(&barrier_Division, NULL, numthreads);
    pthread_barrier_init(&barrier_Elimination, NULL, numthreads);
    pthread_t handles[numthreads];
    threadParam_t1 param[numthreads];
    for (int t_id = 0; t_id < numthreads; ++t_id) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc_barrier_tri_row_AVX, (void*)&param[t_id]);
    }
    for (int t_id = 0; t_id < numthreads; ++t_id) {
        pthread_join(handles[t_id], NULL);
    }
    // 销毁 barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
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

int main() {
    m_reset(A);
    backup(A, a);
    int numRuns = 20;
    cout << "serial:" << measureFunction(serial, numRuns) << " ms" << endl;
    cout << "serial_SSE:" << measureFunction(serial_SSE, numRuns) << " ms" << endl;
    //cout << "dynamic_row:" << measureFunction(dynamic_row, numRuns) << " ms" << endl;
    //cout << "dynamic_col:"<< measureFunction(dynamic_col, numRuns) << " ms" << endl;
    //cout << "static_sem_row:" << measureFunction(static_sem_row, numRuns) << " ms" << endl;
    //cout << "static_sem_col:" << measureFunction(static_sem_col, numRuns) << " ms" << endl;
    //cout << "static_sem_tri_row:" << measureFunction(static_sem_tri_row, numRuns) << " ms" << endl;
    //cout << "static_sem_tri_col:" << measureFunction(static_sem_tri_col, numRuns) << " ms" << endl;
    cout << "static_barrier_tri_row:" << measureFunction(static_barrier_tri_row, numRuns) << " ms" << endl;
    //ut << "static_barrier_tri_col:" << measureFunction(static_barrier_tri_col, numRuns) << " ms" << endl;
    //cout << "dynamic_row_SSE:" << measureFunction(dynamic_row_SSE, numRuns) << " ms" << endl;
    //cout << "dynamic_row_AVX:" << measureFunction(dynamic_row_AVX, numRuns) << " ms" << endl;
    //cout << "static_sem_row_SSE:" << measureFunction(static_sem_row_SSE, numRuns) << " ms" << endl;
    //cout << "static_sem_row_AVX:" << measureFunction(static_sem_row_AVX, numRuns) << " ms" << endl;
    //cout << "static_sem_tri_row_SSE:" << measureFunction(static_sem_tri_row_SSE, numRuns) << " ms" << endl;
    //cout << "static_sem_tri_row_AVX:" << measureFunction(static_sem_tri_row_AVX, numRuns) << " ms" << endl;
    //cout << "static_barrier_tri_row_SSE:" << measureFunction(static_barrier_tri_row_SSE, numRuns) << " ms" << endl;
    //cout << "static_barrier_tri_row_AVX:" << measureFunction(static_barrier_tri_row_AVX, numRuns) << " ms" << endl;
}