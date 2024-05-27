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
    for (int k = 0; k < N; k++) {  // ����ÿһ�У���Ϊ��׼��
        for (int i = k + 1; i < N; i++) {  // ������׼��֮�µ�ÿһ��
            for (int j = 0; j < N; j++) {  // ����ÿһ��
                m[i][j] += m[k][j];  // ����׼�е�Ԫ��ֵ�ۼӵ���ǰ�еĶ�ӦԪ����
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
    int k;      // �����ĵ�ǰ��
    int t_id;   // �߳�id
} threadParam_t;

//��ȥ����
//ˮƽ�����̺߳���
void* threadFunc_row(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;//��ȥ�ִ�
    int t_id = p->t_id;//�̱߳��
    // �̴߳����Ӧ����
    int i = k + t_id +1;  // ��ȡ�Լ��ļ�������
    for (int m = i; m < N;m+=numthreads){  // ȷ���̴߳�����в�ͬ
        for (int j = k + 1; j < N; j++) {
            A[m][j] -= A[m][k] * A[k][j];
        }
        A[m][k] = 0;//���������ó�0
    }
    pthread_exit(NULL);
    return 0;
}

//��̬�߳�
void dynamic_row() {
    for (int k = 0; k < N; k++) {
        // ���߳�����������
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        // ���������̣߳�������ȥ����
        int worker_count = numthreads; // �����߳�����
        pthread_t* handles = new pthread_t[worker_count]; // ������Ӧ���߳̾��
        threadParam_t* params = new threadParam_t[worker_count]; // ������Ӧ���߳����ݽṹ

        // ��������
        for (int t_id = 0; t_id < worker_count; t_id++) {
            params[t_id].k = k;
            params[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc_row, (void*)&params[t_id]);
        }

        // ���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }

        delete[] handles;
        delete[] params;
    }
}

//��ֱ�����̺߳���
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
        // ���߳�����������
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;

        // ���������̣߳�������ȥ����
        int worker_count = numthreads; // �����߳�����
        pthread_t* handles = new pthread_t[worker_count]; // ������Ӧ���߳̾��
        threadParam_t* params = new threadParam_t[worker_count]; // ������Ӧ���߳����ݽṹ

        // ��������
        for (int t_id = 0; t_id < worker_count; t_id++) {
            params[t_id].k = k;
            params[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc_col, (void*)&params[t_id]);
        }

        // ���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
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

//��̬�߳�+�ź�ͬ�����汾
typedef struct {
    int t_id; //�߳� id
}threadParam_t1;

sem_t sem_main;
sem_t* sem_workerstart;
sem_t* sem_workerend;

void* threadFunc_static_row(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++) {
        sem_wait(&sem_workerstart[t_id]); // �������ȴ�������ɳ�������
        // ѭ����������
        for (int i = k + 1 + t_id; i < N; i += numthreads) {
            // ��ȥ
            for (int j = k + 1; j < N; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
        sem_post(&sem_main); // �������߳�
        sem_wait(&sem_workerend[t_id]); // �������ȴ����̻߳��ѽ�����һ��
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_row() {
    //��ʼ���ź���
    sem_init(&sem_main, 0, 0);
    sem_workerstart = new sem_t[numthreads];
    sem_workerend = new sem_t[numthreads];
    for (int i = 0; i < numthreads; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    //�����߳�
    for (int t = 0; t < numthreads; t++) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_static_row, (void*)&params[t]);
    }
    for (int k = 0; k < N; ++k) {
        // ���̳߳�������
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        // �������й����߳�
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerstart[t]);
        }
        // �ȴ����й����߳����
        for (int t = 0; t < numthreads; t++) {
            sem_wait(&sem_main);
        }
        // �������й����߳̽�����һ��
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerend[t]);
        }
    }
    // �ȴ������߳����
    for (int t = 0; t < numthreads; t++) {
        pthread_join(threads[t], NULL);
    }
    // �����ź���
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
        sem_wait(&sem_workerstart[t_id]); // �������ȴ�������ɳ�������
        // ѭ����������
        for (int j = k + t_id+1; j < N; j += numthreads){
            for (int i = k + 1; i < N; i ++) {
            // ��ȥ
                A[i][j] -= A[i][k] * A[k][j];
            }
            //A[j][k] = 0.0;
        }
        sem_post(&sem_main); // �������߳�
        sem_wait(&sem_workerend[t_id]); // �������ȴ����̻߳��ѽ�����һ��
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_col() {
    //��ʼ���ź���
    sem_init(&sem_main, 0, 0);
    sem_workerstart = new sem_t[numthreads];
    sem_workerend = new sem_t[numthreads];
    for (int i = 0; i < numthreads; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    //�����߳�
    for (int t = 0; t < numthreads; t++) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_static_col, (void*)&params[t]);
    }
    for (int k = 0; k < N; ++k) {
        // ���̳߳�������
        for (int j = k + 1; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        // �������й����߳�
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerstart[t]);
        }
        // �ȴ����й����߳����
        for (int t = 0; t < numthreads; t++) {
            sem_wait(&sem_main);
        }
        // �������й����߳̽�����һ��
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerend[t]);
        }
    }
    // �ȴ������߳����
    for (int t = 0; t < numthreads; t++) {
        pthread_join(threads[t], NULL);
    }
    for (int i = 1; i < N; i++) {
        for (int j = 0; j < i; j++) {
            A[i][j] = 0.0;
        }
    }

    // �����ź���
    sem_destroy(&sem_main);
    for (int i = 0; i < numthreads; i++) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }

}


//��̬�߳� + �ź���ͬ���汾 + ����ѭ��ȫ�������̺߳���
sem_t sem_leader; // �쵼�߳��ź���
sem_t sem_Division[numthreads - 1]; // ���������ź���
sem_t sem_Elimination[numthreads - 1]; // ��ȥ�����ź���


void* threadFunc_sem_tri_row(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; ++k) {
        // t_id Ϊ 0 ���̸߳�����������������߳��ȵȴ�
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
            sem_wait(&sem_Division[t_id - 1]); // �������ȴ���ɳ�������
        }
        // ѭ���������񣨲��ö������񻮷ַ�ʽ��
        for (int i = k + 1 + t_id; i < N; i += numthreads) {
            for (int j = k + 1; j < N; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }
        // t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
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
            sem_wait(&sem_Elimination[t_id - 1]); // �������ȴ����ѽ�����ȥ����
        }
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_tri_row() {
    // ��ʼ���ź���
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < numthreads - 1; ++i) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    // �����߳�
    for (int t = 0; t < numthreads; ++t) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_sem_tri_row, (void*)&params[t]);
    }
    // �ȴ������߳����
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
        // t_id Ϊ 0 ���̸߳�����������������߳��ȵȴ�
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
            sem_wait(&sem_Division[t_id - 1]); // �������ȴ���ɳ�������
        }
        // ѭ���������񣨲��ö������񻮷ַ�ʽ��
        for (int j = k + 1 + t_id; j < N; j += numthreads) {
            for (int i = k + 1; i< N; ++i) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            //A[i][k] = 0.0;
        }
        // t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
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
            sem_wait(&sem_Elimination[t_id - 1]); // �������ȴ����ѽ�����ȥ����
        }
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_tri_col() {
    // ��ʼ���ź���
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < numthreads - 1; ++i) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    // �����߳�
    for (int t = 0; t < numthreads; ++t) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_sem_tri_col, (void*)&params[t]);
    }
    // �ȴ������߳����
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

//��̬�߳� +barrier ͬ��

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

        // ��һ��ͬ����
        pthread_barrier_wait(&barrier_Division);

        for (int i = k + 1 + t_id; i <N; i += numthreads) {
            for (int j = k + 1; j <N; ++j) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }

        // �ڶ���ͬ����
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
    // ���� barrier
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

        // ��һ��ͬ����
        pthread_barrier_wait(&barrier_Division);

        for (int j = k + 1 + t_id; j < N; j += numthreads) {
            for (int i = k + 1; i < N; ++i) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            //A[i][k] = 0.0;
        }

        // �ڶ���ͬ����
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
    // ���� barrier
    pthread_barrier_destroy(&barrier_Division);
    pthread_barrier_destroy(&barrier_Elimination);
}

//��̬�̣߳�SSE
void* threadFunc_row_SSE(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    int v = k + t_id + 1;
    __m128 t1, t2;
    for (int i = v; i < N; i += numthreads) {
        t1 = _mm_set_ps1(A[i][k]);
        int j=k+1;
        for (; j <= N-4; j += 4) {//j + 4 <= N,ԭ�������ͻ�Ѷ���Ĳ���4�ı�����Ҳ�����
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
        // ���������̣߳�������ȥ����
        int worker_count = numthreads; // �����߳�����
        pthread_t* handles = new pthread_t[worker_count]; // ������Ӧ���߳̾��
        threadParam_t* params = new threadParam_t[worker_count]; // ������Ӧ���߳����ݽṹ

        // ��������
        for (int t_id = 0; t_id < worker_count; t_id++) {
            params[t_id].k = k;
            params[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc_row_SSE, (void*)&params[t_id]);
        }

        // ���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }

        delete[] handles;
        delete[] params;
    }
}

//��̬�̣߳�AVX
void* threadFunc_row_AVX(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int k = p->k;
    int t_id = p->t_id;
    int v = k + t_id + 1;
    __m256 t1, t2;
    for (int i = v; i < N; i += numthreads) {
        t1 = _mm256_set1_ps(A[i][k]);
        int j=k+1;
        for (int j = k + 1; j <= N - 8; j += 8) {//j + 4 <= N,ԭ�������ͻ�Ѷ���Ĳ���4�ı�����Ҳ�����
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
        // ���������̣߳�������ȥ����
        int worker_count = numthreads; // �����߳�����
        pthread_t* handles = new pthread_t[worker_count]; // ������Ӧ���߳̾��
        threadParam_t* params = new threadParam_t[worker_count]; // ������Ӧ���߳����ݽṹ

        // ��������
        for (int t_id = 0; t_id < worker_count; t_id++) {
            params[t_id].k = k;
            params[t_id].t_id = t_id;
            pthread_create(&handles[t_id], NULL, threadFunc_row_AVX, (void*)&params[t_id]);
        }

        // ���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
        for (int t_id = 0; t_id < worker_count; t_id++) {
            pthread_join(handles[t_id], NULL);
        }

        delete[] handles;
        delete[] params;
    }
}

//��̬�̣߳��ź�����SSE
void* threadFunc_static_row_SSE(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++) {
        sem_wait(&sem_workerstart[t_id]); // �������ȴ�������ɳ�������
        __m128 t1, t2;
        // ѭ����������
        for (int i = k + 1 + t_id; i < N; i += numthreads) {
            t1 = _mm_set_ps1(A[i][k]);
            // ��ȥ
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
        sem_post(&sem_main); // �������߳�
        sem_wait(&sem_workerend[t_id]); // �������ȴ����̻߳��ѽ�����һ��
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_row_SSE() {
    //��ʼ���ź���
    sem_init(&sem_main, 0, 0);
    sem_workerstart = new sem_t[numthreads];
    sem_workerend = new sem_t[numthreads];
    for (int i = 0; i < numthreads; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    //�����߳�
    for (int t = 0; t < numthreads; t++) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_static_row_SSE, (void*)&params[t]);
    }
    for (int k = 0; k < N; ++k) {
        __m128 t1, t2;
        t1 = _mm_set_ps1(A[k][k]);
        int j=k+1;
        // ���̳߳�������
        for (; j <=N-4; j+=4) {
            t2 = _mm_loadu_ps(A[k] + j);
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(A[k] + j, t2);
        }
        for (; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0;
        // �������й����߳�
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerstart[t]);
        }
        // �ȴ����й����߳����
        for (int t = 0; t < numthreads; t++) {
            sem_wait(&sem_main);
        }
        // �������й����߳̽�����һ��
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerend[t]);
        }
    }
    // �ȴ������߳����
    for (int t = 0; t < numthreads; t++) {
        pthread_join(threads[t], NULL);
    }
    // �����ź���
    sem_destroy(&sem_main);
    for (int i = 0; i < numthreads; i++) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }
}

//��̬�̣߳��ź�����AVX
void* threadFunc_static_row_AVX(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++) {
        sem_wait(&sem_workerstart[t_id]); // �������ȴ�������ɳ�������
        __m256 t1, t2;
        // ѭ����������
        for (int i = k + 1 + t_id; i < N; i += numthreads) {
            t1 = _mm256_set1_ps(A[i][k]);
            // ��ȥ
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
        sem_post(&sem_main); // �������߳�
        sem_wait(&sem_workerend[t_id]); // �������ȴ����̻߳��ѽ�����һ��
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_row_AVX() {
    //��ʼ���ź���
    sem_init(&sem_main, 0, 0);
    sem_workerstart = new sem_t[numthreads];
    sem_workerend = new sem_t[numthreads];
    for (int i = 0; i < numthreads; i++) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    //�����߳�
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

        // �������й����߳�
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerstart[t]);
        }
        // �ȴ����й����߳����
        for (int t = 0; t < numthreads; t++) {
            sem_wait(&sem_main);
        }
        // �������й����߳̽�����һ��
        for (int t = 0; t < numthreads; t++) {
            sem_post(&sem_workerend[t]);
        }
    }
    // �ȴ������߳����
    for (int t = 0; t < numthreads; t++) {
        pthread_join(threads[t], NULL);
    }
    // �����ź���
    sem_destroy(&sem_main);
    for (int i = 0; i < numthreads; i++) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
    }
}

//��̬�߳� + �ź���ͬ���汾 + ����ѭ��ȫ�������̺߳���,SSE
void* threadFunc_sem_tri_row_SSE(void* param) {
    threadParam_t1* p = (threadParam_t1*)param;
    int t_id = p->t_id;
    __m128 t1, t2;
    for (int k = 0; k < N; ++k) {
        // t_id Ϊ 0 ���̸߳�����������������߳��ȵȴ�
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
            sem_wait(&sem_Division[t_id - 1]); // �������ȴ���ɳ�������
        }
        // ѭ���������񣨲��ö������񻮷ַ�ʽ��
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
        // t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
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
            sem_wait(&sem_Elimination[t_id - 1]); // �������ȴ����ѽ�����ȥ����
        }
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_tri_row_SSE() {
    // ��ʼ���ź���
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < numthreads - 1; ++i) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    // �����߳�
    for (int t = 0; t < numthreads; ++t) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_sem_tri_row_SSE, (void*)&params[t]);
    }
    // �ȴ������߳����
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
        // t_id Ϊ 0 ���̸߳�����������������߳��ȵȴ�
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
            sem_wait(&sem_Division[t_id - 1]); // �������ȴ���ɳ�������
        }
        // ѭ���������񣨲��ö������񻮷ַ�ʽ��
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
        // t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
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
            sem_wait(&sem_Elimination[t_id - 1]); // �������ȴ����ѽ�����ȥ����
        }
    }
    pthread_exit(NULL);
    return 0;
}

void static_sem_tri_row_AVX() {
    // ��ʼ���ź���
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < numthreads - 1; ++i) {
        sem_init(&sem_Division[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }
    pthread_t threads[numthreads];
    threadParam_t1 params[numthreads];
    // �����߳�
    for (int t = 0; t < numthreads; ++t) {
        params[t].t_id = t;
        pthread_create(&threads[t], NULL, threadFunc_sem_tri_row_AVX, (void*)&params[t]);
    }
    // �ȴ������߳����
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

        // ��һ��ͬ����
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

        // �ڶ���ͬ����
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
    // ���� barrier
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
            // ����ʣ���Ԫ��
            for (; j < N; j++) {
                A[k][j] /= A[k][k];
            }
            A[k][k] = 1.0;
        }

        // ��һ��ͬ����
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
            // ����ʣ���Ԫ��
            for (; j < N; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][k] = 0.0;
        }

        // �ڶ���ͬ����
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
    // ���� barrier
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