# RK3588 matmul benchmark

Quick and dirty benchmarking tool to measure the performance of RK3588 NPU. 

## How to use

```c++
> c++ bench.cpp -o bench -lrknnrt -O3 -std=c++20
> tasket -c 4-7 ./bench
```
And 2 files should appear.
* `result.csv` - the actual performance of the matrix multiplcation itself
* `init.csv` - How long does it take to setup the matrix multiplcation
