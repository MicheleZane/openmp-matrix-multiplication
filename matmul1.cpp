#include <iostream>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <string>
#include <algorithm>
#include <omp.h>

using namespace std;

void fillMatrix(vector<vector<float>>& matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = float(rand() % 10);
        }
    }
}

void MultiplyMatricesSequential(const vector<vector<float>>& A, 
                                 const vector<vector<float>>& B, 
                                 vector<vector<float>>& C, 
                                 int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < size; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void MultiplyMatricesParallel(const vector<vector<float>>& A, 
                               const vector<vector<float>>& B, 
                               vector<vector<float>>& C, 
                               int size, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < size; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void MultiplyMatricesParallelSchedule(const vector<vector<float>>& A,
                                       const vector<vector<float>>& B,
                                       vector<vector<float>>& C,
                                       int size, int num_threads, 
                                       const string& schedule_type) {
    omp_set_num_threads(num_threads);
    if (schedule_type == "static") {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                C[i][j] = 0;
                for (int k = 0; k < size; ++k)
                    C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    else if (schedule_type == "dynamic") {
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                C[i][j] = 0;
                for (int k = 0; k < size; ++k)
                    C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    else if (schedule_type == "guided") {
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                C[i][j] = 0;
                for (int k = 0; k < size; ++k)
                    C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void MultiplyMatricesParallelTiled(const vector<vector<float>>& A,
                                    const vector<vector<float>>& B,
                                    vector<vector<float>>& C,
                                    int size, int num_threads, int block_size) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            C[i][j] = 0.0f;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < size; ii += block_size) {
        for (int jj = 0; jj < size; jj += block_size) {
            for (int kk = 0; kk < size; kk += block_size) {
                for (int i = ii; i < min(ii + block_size, size); ++i) {
                    for (int j = jj; j < min(jj + block_size, size); ++j) {
                        float sum = C[i][j];
                        for (int k = kk; k < min(kk + block_size, size); ++k)
                            sum += A[i][k] * B[k][j];
                        C[i][j] = sum;
                    }
                }
            }
        }
    }
}


void MultiplyMatricesParallelTiledOptimized(const vector<vector<float>>& A,
                                              const vector<vector<float>>& B,
                                              vector<vector<float>>& C,
                                              int size, int num_threads, int block_size) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            C[i][j] = 0.0f;

    #pragma omp parallel for schedule(dynamic)
    for (int ii = 0; ii < size; ii += block_size) {
        int i_max = min(ii + block_size, size);
        for (int jj = 0; jj < size; jj += block_size) {
            int j_max = min(jj + block_size, size);
            for (int kk = 0; kk < size; kk += block_size) {
                int k_max = min(kk + block_size, size);
                for (int i = ii; i < i_max; ++i) {
                    for (int k = kk; k < k_max; ++k) {
                        float a_ik = A[i][k];         
                        for (int j = jj; j < j_max; ++j)
                            C[i][j] += a_ik * B[k][j]; 
                    }
                }
            }
        }
    }
}

bool verifyResults(const vector<vector<float>>& C_seq, 
                   const vector<vector<float>>& C_par, 
                   int size) {
    float epsilon = 1e-3;
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            if (fabs(C_seq[i][j] - C_par[i][j]) > epsilon) {
                cout << "Mismatch at [" << i << "][" << j << "]: " 
                     << C_seq[i][j] << " vs " << C_par[i][j] << endl;
                return false;
            }
    return true;
}

int main(int argc, char** argv) {
    int size        = 512;
    int num_threads = 8;
    int block_size  = 64;

    if (argc >= 2) size        = atoi(argv[1]);
    if (argc >= 3) num_threads = atoi(argv[2]);

    string mode = "normal";
    if (argc >= 4) mode = argv[3];
    if (argc >= 5 && (mode == "tiled" || mode == "tiled_opt"))
        block_size = atoi(argv[4]);

    srand(42);

    if (mode == "seq") {
        int nruns = (argc >= 5) ? atoi(argv[4]) : 3;
        vector<vector<float>> A(size, vector<float>(size));
        vector<vector<float>> B(size, vector<float>(size));
        vector<vector<float>> C(size, vector<float>(size, 0));
        fillMatrix(A, size);
        fillMatrix(B, size);
        MultiplyMatricesSequential(A, B, C, size); // warm-up
        vector<double> times;
        for (int r = 0; r < nruns; r++) {
            auto t0 = chrono::high_resolution_clock::now();
            MultiplyMatricesSequential(A, B, C, size);
            times.push_back(chrono::duration<double>(
                chrono::high_resolution_clock::now() - t0).count());
        }
        sort(times.begin(), times.end());
        printf("%.6f\n", times[nruns / 2]);
        return 0;
    }

    if (mode == "par_only") {
        double seq_time = (argc >= 5) ? atof(argv[4]) : 1.0;
        int    nruns    = (argc >= 6) ? atoi(argv[5]) : 10;
        vector<vector<float>> A(size, vector<float>(size));
        vector<vector<float>> B(size, vector<float>(size));
        vector<vector<float>> C(size, vector<float>(size, 0));
        vector<vector<float>> C_seq(size, vector<float>(size, 0));
        fillMatrix(A, size);
        fillMatrix(B, size);
        MultiplyMatricesSequential(A, B, C_seq, size);
        MultiplyMatricesParallel(A, B, C, size, num_threads); // warm-up
        bool correct = verifyResults(C_seq, C, size);
        vector<double> times;
        for (int r = 0; r < nruns; r++) {
            auto t0 = chrono::high_resolution_clock::now();
            MultiplyMatricesParallel(A, B, C, size, num_threads);
            times.push_back(chrono::duration<double>(
                chrono::high_resolution_clock::now() - t0).count());
        }
        sort(times.begin(), times.end());
        double par_time   = times[nruns / 2];
        double mean = 0; for (auto t : times) mean += t; mean /= nruns;
        double var  = 0; for (auto t : times) var += (t-mean)*(t-mean); var /= nruns;
        double stddev     = sqrt(var);
        double speedup    = seq_time / par_time;
        double efficiency = speedup / num_threads;
        printf("%d,%d,%.6f,%.6f,%.4f,%.4f,%s,%.6f\n",
               size, num_threads, seq_time, par_time, speedup, efficiency,
               correct ? "OK" : "FAIL", stddev);
        return 0;
    }

    if (mode == "static_only" || mode == "dynamic_only" || mode == "guided_only") {
        double seq_time = (argc >= 5) ? atof(argv[4]) : 1.0;
        int    nruns    = (argc >= 6) ? atoi(argv[5]) : 10;
        string sched = mode.substr(0, mode.find("_only"));
        vector<vector<float>> A(size, vector<float>(size));
        vector<vector<float>> B(size, vector<float>(size));
        vector<vector<float>> C(size, vector<float>(size, 0));
        vector<vector<float>> C_seq(size, vector<float>(size, 0));
        fillMatrix(A, size);
        fillMatrix(B, size);
        MultiplyMatricesSequential(A, B, C_seq, size);
        MultiplyMatricesParallelSchedule(A, B, C, size, num_threads, sched);
        bool correct = verifyResults(C_seq, C, size);
        vector<double> times;
        for (int r = 0; r < nruns; r++) {
            auto t0 = chrono::high_resolution_clock::now();
            MultiplyMatricesParallelSchedule(A, B, C, size, num_threads, sched);
            times.push_back(chrono::duration<double>(
                chrono::high_resolution_clock::now() - t0).count());
        }
        sort(times.begin(), times.end());
        double par_time   = times[nruns / 2];
        double mean = 0; for (auto t : times) mean += t; mean /= nruns;
        double var  = 0; for (auto t : times) var += (t-mean)*(t-mean); var /= nruns;
        double stddev     = sqrt(var);
        double speedup    = seq_time / par_time;
        double efficiency = speedup / num_threads;
        printf("%s,%d,%d,%.6f,%.4f,%.4f,%s,%.6f\n",
               sched.c_str(), size, num_threads, par_time, speedup, efficiency,
               correct ? "OK" : "FAIL", stddev);
        return 0;
    }

    if (mode == "tiled_only" || mode == "tiled_opt_only") {
        int    block    = (argc >= 5) ? atoi(argv[4]) : 64;
        double seq_time = (argc >= 6) ? atof(argv[5]) : 1.0;
        int    nruns    = (argc >= 7) ? atoi(argv[6]) : 10;
        vector<vector<float>> A(size, vector<float>(size));
        vector<vector<float>> B(size, vector<float>(size));
        vector<vector<float>> C(size, vector<float>(size, 0));
        vector<vector<float>> C_seq(size, vector<float>(size, 0));
        fillMatrix(A, size);
        fillMatrix(B, size);
        MultiplyMatricesSequential(A, B, C_seq, size);

        if (mode == "tiled_only")
            MultiplyMatricesParallelTiled(A, B, C, size, num_threads, block);
        else
            MultiplyMatricesParallelTiledOptimized(A, B, C, size, num_threads, block);
        bool correct = verifyResults(C_seq, C, size);
        vector<double> times;
        for (int r = 0; r < nruns; r++) {
            auto t0 = chrono::high_resolution_clock::now();
            if (mode == "tiled_only")
                MultiplyMatricesParallelTiled(A, B, C, size, num_threads, block);
            else
                MultiplyMatricesParallelTiledOptimized(A, B, C, size, num_threads, block);
            times.push_back(chrono::duration<double>(
                chrono::high_resolution_clock::now() - t0).count());
        }
        sort(times.begin(), times.end());
        double par_time   = times[nruns / 2];
        double mean = 0; for (auto t : times) mean += t; mean /= nruns;
        double var  = 0; for (auto t : times) var += (t-mean)*(t-mean); var /= nruns;
        double stddev     = sqrt(var);
        double speedup    = seq_time / par_time;
        double efficiency = speedup / num_threads;
        string label = (mode == "tiled_only") ? "tiled" : "tiled_opt";
        printf("%s,%d,%d,%d,%.6f,%.4f,%.4f,%s,%.6f\n",
               label.c_str(), block, size, num_threads, par_time, speedup, efficiency,
               correct ? "OK" : "FAIL", stddev);
        return 0;
    }

    vector<vector<float>> A(size, vector<float>(size));
    vector<vector<float>> B(size, vector<float>(size));
    vector<vector<float>> C_seq(size, vector<float>(size));
    vector<vector<float>> C_par(size, vector<float>(size));

    fillMatrix(A, size);
    fillMatrix(B, size);

    auto start_seq = chrono::high_resolution_clock::now();
    MultiplyMatricesSequential(A, B, C_seq, size);
    auto end_seq   = chrono::high_resolution_clock::now();
    double seq_time = chrono::duration<double>(end_seq - start_seq).count();

    double par_time = 0.0;
    bool   correct  = true;

    if (mode == "static" || mode == "dynamic" || mode == "guided") {
        auto start_par = chrono::high_resolution_clock::now();
        MultiplyMatricesParallelSchedule(A, B, C_par, size, num_threads, mode);
        auto end_par = chrono::high_resolution_clock::now();
        par_time = chrono::duration<double>(end_par - start_par).count();
        if (size <= 256) correct = verifyResults(C_seq, C_par, size);
    }
    else if (mode == "tiled") {
        auto start_par = chrono::high_resolution_clock::now();
        MultiplyMatricesParallelTiled(A, B, C_par, size, num_threads, block_size);
        auto end_par = chrono::high_resolution_clock::now();
        par_time = chrono::duration<double>(end_par - start_par).count();
        if (size <= 256) correct = verifyResults(C_seq, C_par, size);
    }
    else if (mode == "tiled_opt") {
        auto start_par = chrono::high_resolution_clock::now();
        MultiplyMatricesParallelTiledOptimized(A, B, C_par, size, num_threads, block_size);
        auto end_par = chrono::high_resolution_clock::now();
        par_time = chrono::duration<double>(end_par - start_par).count();
        if (size <= 256) correct = verifyResults(C_seq, C_par, size);
    }
    else {
        auto start_par = chrono::high_resolution_clock::now();
        MultiplyMatricesParallel(A, B, C_par, size, num_threads);
        auto end_par = chrono::high_resolution_clock::now();
        par_time = chrono::duration<double>(end_par - start_par).count();
        if (size <= 256) correct = verifyResults(C_seq, C_par, size);
    }

    double speedup    = seq_time / par_time;
    double efficiency = speedup / num_threads;

    if (mode == "quick") {
        cout << size << "\t" << num_threads << "\t" 
             << seq_time << "\t" << par_time << "\t" 
             << speedup << "\t" << efficiency << endl;
    }
    else if (mode == "verify") {
        cout << "Verification: " << (correct ? "PASSED" : "FAILED") << endl;
        if (correct)
            cout << size << "," << num_threads << "," << seq_time << "," 
                 << par_time << "," << speedup << "," << efficiency << endl;
    }
    else if (mode == "bench") {
        cout << size << "," << num_threads << "," << seq_time << "," 
             << par_time << "," << speedup << "," << efficiency << endl;
    }
    else if (mode == "static" || mode == "dynamic" || mode == "guided") {
        cout << mode << "," << size << "," << num_threads << "," 
             << par_time << "," << speedup << "," << efficiency << "," 
             << (correct ? "OK" : "FAIL") << endl;
    }
    else if (mode == "tiled" || mode == "tiled_opt") {
        cout << mode << "," << block_size << "," << size << "," << num_threads << "," 
             << par_time << "," << speedup << "," << efficiency << "," 
             << (correct ? "OK" : "FAIL") << endl;
    }
    else {
        cout << "Matrix size: "       << size << "x" << size << endl;
        cout << "Number of threads: " << num_threads << endl;
        cout << "Mode: "              << mode << endl;
        if (mode == "tiled" || mode == "tiled_opt")
            cout << "Block size: "    << block_size << endl;
        cout << "Sequential time: "   << seq_time << " seconds" << endl;
        cout << "Parallel time: "     << par_time << " seconds" << endl;
        cout << "Speedup: "           << speedup << endl;
        cout << "Efficiency: "        << efficiency << endl;
        if (!correct && size <= 256)
            cout << "WARNING: Results do not match!" << endl;
    }

    return 0;
}