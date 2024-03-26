#include <iostream>
#include <sys/time.h>
#include <unistd.h>

using namespace std;

const int n = 10000;

int main() {
    double** a = new double* [n];
    for (int i = 0; i < n; ++i)
        a[i] = new double[n];
    double* b = new double[n];
    double* sum = new double[n];

    for (int i = 0; i < n; i++) {
        b[i] = 1.0;
        for (int j = 0; j < n; j++) {
            a[i][j] = i + j;
        }
    }

    int step = 10;
    for (int k = 10; k <= n; k += step) {
        for (int i = 0; i < k; i++) {
            sum[i] = 0.0;
        }

        struct timeval tstart, tend;
        double timeUsed;
        gettimeofday(&tstart, NULL);
        double total_time = 0.0;
        int counter = 0;

        while (total_time < 5) { 
            // ordinary
            for (int i = 0; i < k; i++)
                for (int j = 0; j < k; j++)
                    sum[i] += a[j][i] * b[j];
            // optimized
            for (int j = 0; j < k; j++)
                for (int i = 0; i < k; i++)
                    sum[i] += a[j][i] * b[j];

            gettimeofday(&tend, NULL);
            timeUsed = 1000000 * (tend.tv_sec - tstart.tv_sec) + tend.tv_usec - tstart.tv_usec;
            total_time += timeUsed / 1000.0;
            counter++;
        }

        cout << "n=" << k << " " << counter << " time:" << total_time / counter << "ms" << endl;

        if (k >= 100) step = 100;
        if (k >= 1000) step = 1000;
    }

    for (int i = 0; i < n; ++i)
        delete[] a[i];
    delete[] a;
    delete[] b;
    delete[] sum;

    return 0;
}


