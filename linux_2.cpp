#include <iostream>
#include <sys/time.h>
#include<unistd.h>
using namespace std;
// 链式求和算法
void chain(int* b, int k) {
    int sum = 0;
    for (int i = 0; i < k; i++) {
        sum += b[i];
    }
}
// 循环展开求和算法
void loopUnrolling(int* b, int k) {
    int sum1 = 0;
    int sum2 = 0;
    for (int i = 0; i < k; i += 2) {
        sum1 += b[i];
        sum2 += b[i + 1];
    }
    int sum = sum1 + sum2;
}
// 递归求和算法
void recursion(int* a, int n) {
    if (n <= 1) return;
    for (int i = 0; i < n / 2; i++) {
        a[i] += a[n - i - 1];
    }
    recursion(a, n / 2);
}

// 二重循环求和算法
void nestedLoop(int* b, int k) {
    for (int m = k; m > 1; m /= 2)
        for (int i = 0; i < m / 2; i++) {
            b[i] = b[i * 2] + b[i * 2 + 1];
        }
}

int main() {
    for (int n = 1; n <= 28; n++) {
        int k = 1 << n;
        int* b = new int[k];
        for (int i = 0; i < k; i++) {
            b[i] = 1; // Initialize b[i] as 1
        }

        struct timeval start, end;
        gettimeofday(&start, NULL);

        double total_time = 0.0;
        int counter = 0;
        while (total_time < 5.0) {
            chain(b, k);
            //loopUnrolling(b, k);
            //recursion(b, k);
            //nestedLoop(b, k);
            gettimeofday(&end, NULL);
            total_time = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
            counter++;
        }

        cout << "n=" << n << " " << counter << " average time: " << total_time / counter << "ms" << endl;
        delete[] b;
    }
    return 0;
}
