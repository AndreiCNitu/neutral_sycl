#ifndef __SHAREDHDR
#define __SHAREDHDR

#pragma once

#include "profiler.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <CL/sycl.hpp>

#define ARCH_PARAMS "arch.params"
#define ENABLE_VISIT_DUMPS 1 // Enables visit dumps
#define VEC_ALIGN 256 // The vector alignment to be used by memory allocators
#define TAG_VISIT0 1000
#define TAG_VISIT1 1001
#define MAX_STR_LEN 1024
#define MAX_KEYS 10
#define GB ((1024.0) * (1024.0) * (1024.0))

// Helper macros
#define strmatch(a, b) (strcmp((a), (b)) == 0)

#ifndef __cplusplus
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#define samesign(a, b) ((a * b) > 0.0)
#define absmin(a, b) ((fabs(a) < fabs(b)) ? (a) : (b))
#define minmod(a, b) (samesign((a), (b)) ? (absmin((a), (b))) : (0.0))
#define within_tolerance(a, b, eps)                                            \
  (!isnan(a) && !isnan(b) &&                                                   \
   fabs(a - b) <= ((fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * eps))
#define kronecker_delta(a, b) (((a) == (b)) ? 1 : 0)
#define triangle(a) ((a) * ((a) + 1) / 2)
#define dswap(a, b)                                                            \
  {                                                                            \
    double t = a;                                                              \
    a = b;                                                                     \
    b = t;                                                                     \
  }

#define TERMINATE(...)                                                         \
  fprintf(stderr, __VA_ARGS__);                                                \
  fprintf(stderr, " %s:%d\n", __FILE__, __LINE__);                             \
  exit(EXIT_FAILURE);

enum { RECV = 0, SEND = 1 }; // Whether data is sent to/received from device

// Global profile hooks
extern struct Profile compute_profile;
extern struct Profile comms_profile;

#ifdef __cplusplus
extern "C" {
#endif

// Allocation and deallocation routines (these need templating away)
size_t allocate_data(cl::sycl::queue queue, cl::sycl::buffer<double, 1>** buf, size_t len);
size_t allocate_data_w_host(cl::sycl::queue queue, cl::sycl::buffer<double, 1>** buf, double* h_buf, size_t len);
size_t allocate_float_data(cl::sycl::queue queue, cl::sycl::buffer<float, 1>** buf, size_t len);
size_t allocate_int_data(cl::sycl::queue queue, cl::sycl::buffer<int, 1>** buf, size_t len);
size_t allocate_int_data_w_host(cl::sycl::queue queue, cl::sycl::buffer<int, 1>** buf, int* h_buf, size_t len);
size_t allocate_uint64_data(cl::sycl::queue queue, cl::sycl::buffer<uint64_t, 1>** buf, const size_t len);

void allocate_host_data(double** buf, size_t len);
void allocate_host_float_data(float* buf, size_t len);
void allocate_host_int_data(int** buf, size_t len);
void allocate_host_uint64_data(uint64_t* buf, size_t len);

// Write out data for visualisation in visit
void write_to_visit(const int nx, const int ny, const int x_off,
                    const int y_off, const double* data, const char* name,
                    const int step, const double time);
void write_to_visit_3d(const int nx, const int ny, const int nz,
                       const int x_off, const int y_off, const int z_off,
                       const double* data, const char* name, const int step,
                       const double time);

// Collects all of the mesh data from the fleet of ranks and then writes to
// visit
void write_all_ranks_to_visit(const int global_nx, const int global_ny,
                              const int local_nx, const int local_ny,
                              const int pad, const int x_off, const int y_off,
                              const int rank, const int nranks, int* neighbours,
                              double* local_arr, const char* name, const int tt,
                              const double elapsed_sim_time);

#ifdef __cplusplus
}
#endif

#endif
