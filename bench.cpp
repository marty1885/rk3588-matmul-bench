#include <rknn_matmul_api.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <span>
#include <cstring>
#include <fstream>

#include <trantor/net/EventLoopThreadPool.h>

template <typename T, typename U>
void fill_random(std::span<T> data, U min, U max)
{
    using Dist = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;
    std::random_device rd;
    std::mt19937 gen(rd());
    Dist dis(min, max);
    for (auto &x : data)
    {
        x = dis(gen);
    }
}

struct RKNNMatMul
{
    RKNNMatMul(int m, int k, int n, rknn_matmul_type type, bool ac_native, bool b_native)
        : m(m), k(k), n(n), type(type)
    {
        memset(&ctx, 0, sizeof(info));
        info.M = m;
        info.K = k;
        info.N = n;
        info.type = type;
        info.B_layout = b_native;
        info.AC_layout = ac_native;

        memset(&attr, 0, sizeof(attr));
        int ret = rknn_matmul_create(&ctx, &info, &attr);
        if (ret != 0)
        {
            std::cerr << "rknn_matmul_create failed: " << ret << std::endl;
            return;
        }

        void *a = nullptr, *b = nullptr;
        if(type == RKNN_INT8_MM_INT8_TO_INT32)
        {
            a = malloc(m * k);
            b = malloc(k * n);

            fill_random(std::span<int8_t>((int8_t*)a, m * k), -128, 127);
            fill_random(std::span<int8_t>((int8_t*)b, k * n), -128, 127);
        }
        else if(type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32)
        {
            a = malloc(m * k * 2);
            b = malloc(k * n * 2);

            fill_random(std::span<uint16_t>((uint16_t*)a, m * k), -1.0, 1.0);
            fill_random(std::span<uint16_t>((uint16_t*)b, k * n), -1.0, 1.0);
        }
        else if(type == RKNN_INT4_MM_INT4_TO_INT16)
        {
            a = malloc(m * k / 2);
            b = malloc(k * n / 2);

            fill_random(std::span<int8_t>((int8_t*)a, m * k / 2), -8, 7);
            fill_random(std::span<int8_t>((int8_t*)b, k * n / 2), -8, 7);
        }
        else
        {
            std::cerr << "Unsupported type: " << type << std::endl;
            return;
        }

        A = rknn_create_mem(ctx, attr.A.size);
        if(A == nullptr)
        {
            std::cerr << "rknn_create_mem failed" << std::endl;
            free(a);
            free(b);
            return;
        }

        B = rknn_create_mem(ctx, attr.B.size);
        if(B == nullptr)
        {
            std::cerr << "rknn_create_mem failed" << std::endl;
            free(a);
            free(b);
            return;
        }
        C = rknn_create_mem(ctx, attr.C.size);
        if(C == nullptr)
        {
            std::cerr << "rknn_create_mem failed" << std::endl;
            free(a);
            free(b);
            return;
        }

        memcpy(A->virt_addr, a, A->size);
        memcpy(B->virt_addr, b, B->size);

        free(a);
        free(b);

        ret = rknn_matmul_set_io_mem(ctx, A, &attr.A);
        if(ret != 0)
        {
            std::cerr << "rknn_matmul_set_io_mem failed: " << ret << std::endl;
            return;
        }
        ret = rknn_matmul_set_io_mem(ctx, B, &attr.B);
        if(ret != 0)
        {
            std::cerr << "rknn_matmul_set_io_mem failed: " << ret << std::endl;
            return;
        }
        ret = rknn_matmul_set_io_mem(ctx, C, &attr.C);
        if(ret != 0)
        {
            std::cerr << "rknn_matmul_set_io_mem failed: " << ret << std::endl;
            return;
        }
    }

    void run()
    {
        int ret = rknn_matmul_run(ctx);
        if(ret != 0)
        {
            std::cerr << "rknn_matmul_compute failed: " << ret << std::endl;
            return;
        }
    }

    ~RKNNMatMul()
    {
        rknn_destroy_mem(ctx, A);
        rknn_destroy_mem(ctx, B);
        rknn_destroy_mem(ctx, C);
        rknn_matmul_destroy(ctx);
    }

    int m, k, n;
    rknn_matmul_type type;
    rknn_matmul_ctx ctx;
    rknn_matmul_info info;
    rknn_matmul_io_attr attr;
    rknn_tensor_mem *A = nullptr, *B = nullptr, *C = nullptr;
};

int main()
{
    constexpr size_t num_threads = 3;
    static_assert(num_threads <= 3); // Only 3 NPU cores on RK3588
    trantor::EventLoopThreadPool pool(num_threads);
    pool.start();

    size_t run_count = 30;
    std::vector<int> m = {1, 2, 4, 8, 16, 32, 64, 128};
    std::vector<int> k = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    std::vector<int> n = {64, 128, 256, 512, 1024, 2048, 4096, 8192};

    std::ofstream file("result.csv");
    if(!file.good())
    {
        std::cerr << "Failed to open result.csv" << std::endl;
        return 1;
    }

    std::ofstream initfile("init.csv");
    if(!initfile.good())
    {
        std::cerr << "Failed to open init.csv" << std::endl;
        return 1;
    }
    file << "count,m,k,n,type,ac_native,b_native,time_ns,gops,threads\n";
    initfile << "m,k,n,type,ac_native,b_native,time_ns,threads\n";
    for(auto m_ : m)
    {
        for(auto k_ : k)
        {
            for(auto n_ : n)
            {
                for(int type = 0; type < 3; type++)
                {
                    for(int ac_native = 0; ac_native < 2; ac_native++)
                    {
                        for(int b_native = 0; b_native < 2; b_native++)
                        {
                            rknn_matmul_type t;
                            std::string type_str;
                            if(type == 0)
                            {
                                t = RKNN_INT8_MM_INT8_TO_INT32;
                                type_str = "RKNN_INT8_MM_INT8_TO_INT32";
                            }
                            else if(type == 1)
                            {
                                t = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
                                type_str = "RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32";
                            }
                            else
                            {
                                t = RKNN_INT4_MM_INT4_TO_INT16;
                                type_str = "RKNN_INT4_MM_INT4_TO_INT16";
                            }

                            std::vector<std::shared_ptr<RKNNMatMul>> matmuls;
                            std::vector<std::promise<void>> promises;
                            promises.resize(num_threads);
                            matmuls.resize(num_threads);

                            auto start = std::chrono::high_resolution_clock::now();
                            for(size_t i = 0; i < num_threads; i++)
                            {
                                pool.getLoop(i)->queueInLoop([&, i]() {
                                    matmuls[i] = std::make_shared<RKNNMatMul>(m_, k_, n_, t, ac_native, b_native);
                                    promises[i].set_value();
                                });
                            }
                            for(auto &p : promises)
                            {
                                p.get_future().get();
                            }
                            auto end = std::chrono::high_resolution_clock::now();
                            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                            initfile << m_ << "," << k_ << "," << n_ << "," << type_str << "," << ac_native << "," << b_native << "," << duration.count() << ", " << num_threads << "\n";
                            std::cout << "INIT m: " << m_ << ", k: " << k_ << ", n: " << n_ << ", type: " << type_str << ", ac_native: " << ac_native << ", b_native: " << b_native << ", init time: " << duration.count() << "ns " << num_threads << " threads\n";

                            for(size_t i = 0; i < run_count; i++)
                            {
                                std::vector<std::promise<void>> promises(num_threads);
                                auto start = std::chrono::high_resolution_clock::now();
                                for(size_t j = 0; j < num_threads; j++)
                                {
                                    auto &prom = promises[j];
                                    pool.getLoop(j)->runInLoop([&, i, j]() mutable {
                                        matmuls[j]->run();
                                        prom.set_value();
                                    });
                                }
                                for(auto &p : promises)
                                {
                                    p.get_future().get();
                                }
                                auto end = std::chrono::high_resolution_clock::now();
                                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                                auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                                auto gops = (uint64_t)m_ * n_ * (2 * k_ - 1) / 1000UL / ((double)duration_us.count());
                                file << i << "," << m_ << "," << k_ << "," << n_ << "," << type_str << "," << ac_native << "," << b_native << "," << duration.count() << "," << gops << "," << num_threads << "\n";
                                std::cout << "m: " << m_ << ", k: " << k_ << ", n: " << n_ << ", type: " << type_str << ", ac_native: " << ac_native << ", b_native: " << b_native << ", time: " << duration.count() << "ns, " << gops*num_threads << "GOPS, threads: " << num_threads << "\n";
                            }
                        }
                    }
                }
            }
        }
    }

    file.close();
    initfile.close();
}
