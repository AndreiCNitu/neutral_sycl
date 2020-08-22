#include "mesh.h"
#include "params.h"
#include "shared.h"
#include "shared_data.h"
#include "shared.h"
#include <math.h>
#include <stdlib.h>

class allocate_data_kernel;
class allocate_data_w_host_kernel;
class allocate_float_data_kernel;
class allocate_int_data_kernel;
class allocate_int_data_w_host_kernel;
class allocate_uint64_data;
class edgedx_init_kernel;
class celldx_init_kernel;
class edgedy_init_kernel;
class celldy_init_kernel;
class set_problem_2d_kernel;

// Allocates a double precision array
size_t allocate_data(cl::sycl::queue queue, cl::sycl::buffer<double, 1>** buf, size_t len) {
    if(len == 0) {
        return 0;
    }

    *buf = new cl::sycl::buffer<double, 1>(cl::sycl::range<1>(len));

    queue.submit([&] (cl::sycl::handler& cgh) {
      auto buf_acc = (*buf)->get_access<cl::sycl::access::mode::discard_write>(cgh);

      cgh.parallel_for<class allocate_data_kernel>(cl::sycl::range<1>(len), [=](cl::sycl::id<1> idx) {
        buf_acc[idx] = 0.0f;
      });
    });

    return sizeof(double) * len;
}

size_t allocate_data_w_host(cl::sycl::queue queue, cl::sycl::buffer<double, 1>** buf, double* h_buf, size_t len) {
    if(len == 0) {
        return 0;
    }

    *buf = new cl::sycl::buffer<double, 1>(h_buf, cl::sycl::range<1>(len));

    queue.submit([&] (cl::sycl::handler& cgh) {
      auto buf_acc = (*buf)->get_access<cl::sycl::access::mode::discard_write>(cgh);

      cgh.parallel_for<class allocate_data_w_host_kernel>(cl::sycl::range<1>(len), [=](cl::sycl::id<1> idx) {
        buf_acc[idx] = 0.0f;
      });
    });

    return sizeof(double) * len;
}

size_t allocate_float_data(cl::sycl::queue queue, cl::sycl::buffer<float, 1>** buf, size_t len) {
    if(len == 0) {
        return 0;
    }

    *buf = new cl::sycl::buffer<float, 1>(cl::sycl::range<1>(len));

    queue.submit([&] (cl::sycl::handler& cgh) {
      auto buf_acc = (*buf)->get_access<cl::sycl::access::mode::discard_write>(cgh);

      cgh.parallel_for<class allocate_float_data_kernel>(cl::sycl::range<1>(len), [=](cl::sycl::id<1> idx) {
        buf_acc[idx] = 0.0f;
      });
    });

    return sizeof(float) * len;
}

size_t allocate_int_data(cl::sycl::queue queue, cl::sycl::buffer<int, 1>** buf, size_t len) {
    if(len == 0) {
        return 0;
    }

    *buf = new cl::sycl::buffer<int, 1>(cl::sycl::range<1>(len));

    queue.submit([&] (cl::sycl::handler& cgh) {
      auto buf_acc = (*buf)->get_access<cl::sycl::access::mode::discard_write>(cgh);

      cgh.parallel_for<class allocate_int_data_kernel>(cl::sycl::range<1>(len), [=](cl::sycl::id<1> idx) {
        buf_acc[idx] = 0.0f;
      });
    });

    return sizeof(int) * len;
}

size_t allocate_int_data_w_host(cl::sycl::queue queue, cl::sycl::buffer<int, 1>** buf, int* h_buf, size_t len) {
    if(len == 0) {
        return 0;
    }

    *buf = new cl::sycl::buffer<int, 1>(h_buf, cl::sycl::range<1>(len));

    queue.submit([&] (cl::sycl::handler& cgh) {
      auto buf_acc = (*buf)->get_access<cl::sycl::access::mode::discard_write>(cgh);

      cgh.parallel_for<class allocate_int_data_w_host_kernel>(cl::sycl::range<1>(len), [=](cl::sycl::id<1> idx) {
        buf_acc[idx] = 0.0f;
      });
    });

    return sizeof(int) * len;
}

size_t allocate_uint64_data(cl::sycl::queue queue, cl::sycl::buffer<uint64_t, 1>** buf, size_t len) {
    if(len == 0) {
        return 0;
    }

    *buf = new cl::sycl::buffer<uint64_t, 1>(cl::sycl::range<1>(len));

    queue.submit([&] (cl::sycl::handler& cgh) {
      auto buf_acc = (*buf)->get_access<cl::sycl::access::mode::discard_write>(cgh);

      cgh.parallel_for<class allocate_uint64_data>(cl::sycl::range<1>(len), [=](cl::sycl::id<1> idx) {
        buf_acc[idx] = 0.0f;
      });
    });

    return sizeof(uint64_t) * len;
}

// Deallocate a [..] array
//

// Allocates some double precision data
void allocate_host_data(double** buf, const size_t len) {
    if(len == 0) {
        return;
    }

    *buf = (double*) malloc(sizeof(double) * len);

    for (int i = 0; i < len; ++i) {
        (*buf)[i] = 1.0; // TODO Why 1, not 0 ?
    }
}

// Allocates some single precision data
void allocate_host_float_data(float* buf, const size_t len) {
    if(len == 0) {
        return;
    }

    buf = (float*) malloc(sizeof(float) * len);

    for (int i = 0; i < len; ++i) {
        buf[i] = 0.0f;
    }
}

void allocate_host_int_data(int** buf, const size_t len) {
    if(len == 0) {
        return;
    }

    *buf = (int*) malloc(sizeof(int) * len);

    for (int i = 0; i < len; ++i) {
        (*buf)[i] = 0;
    }
}

void allocate_host_uint64_t_data(uint64_t* buf, const size_t len) {
    if(len == 0) {
        return;
    }

    buf = (uint64_t*) malloc(sizeof(uint64_t) * len);

    for (int i = 0; i < len; ++i) {
        buf[i] = 0;
    }
}

// Deallocates a data array
//

// Initialises mesh data in device specific manner
void mesh_data_init_2d(cl::sycl::queue queue,
    const int local_nx, const int local_ny,
    const int global_nx, const int global_ny, const int pad,
    const int x_off, const int y_off, const double width,
    const double height, cl::sycl::buffer<double, 1>* edgex, cl::sycl::buffer<double, 1>* edgey,
    cl::sycl::buffer<double, 1>* edgedx, cl::sycl::buffer<double, 1>* edgedy,
    cl::sycl::buffer<double, 1>* celldx, cl::sycl::buffer<double, 1>* celldy) {

    cl::sycl::buffer<double, 1> edgedx_ = *edgedx;
    cl::sycl::buffer<double, 1> edgex_ = *edgex;
    queue.submit([&] (cl::sycl::handler& cgh) {
      auto edgedx_acc = edgedx_.get_access<cl::sycl::access::mode::read_write>(cgh);
      auto edgex_acc = edgex_.get_access<cl::sycl::access::mode::discard_write>(cgh);

      // Simple uniform rectilinear initialisation
      cgh.parallel_for<class edgedx_init_kernel>(cl::sycl::range<1>(local_nx+1), [=](cl::sycl::id<1> idx) {
        edgedx_acc[idx] = width / (global_nx);

        // Note: correcting for padding
        edgex_acc[idx] = edgedx_acc[idx] * (x_off + idx[0] - pad);
      });
    });

    queue.submit([&] (cl::sycl::handler& cgh) {
      auto celldx_acc = celldx->get_access<cl::sycl::access::mode::discard_write>(cgh);

      cgh.parallel_for<class celldx_init_kernel>(cl::sycl::range<1>(local_nx), [=](cl::sycl::id<1> idx) {
        celldx_acc[idx] = width / (global_nx);
      });
    });

    queue.submit([&] (cl::sycl::handler& cgh) {
      auto edgedy_acc = edgedy->get_access<cl::sycl::access::mode::read_write>(cgh);
      auto edgey_acc = edgey->get_access<cl::sycl::access::mode::discard_write>(cgh);

      // Simple uniform rectilinear initialisation
      cgh.parallel_for<class edgedy_init_kernel>(cl::sycl::range<1>(local_ny+1), [=](cl::sycl::id<1> idx) {
        edgedy_acc[idx] = height / (global_ny);

        // Note: correcting for padding
        edgey_acc[idx] = edgedy_acc[idx] * (y_off + idx[0] - pad);
      });
    });

    queue.submit([&] (cl::sycl::handler& cgh) {
      auto celldy_acc = celldy->get_access<cl::sycl::access::mode::discard_write>(cgh);

      cgh.parallel_for<class celldy_init_kernel>(cl::sycl::range<1>(local_ny), [=](cl::sycl::id<1> idx) {
        celldy_acc[idx] = height / (global_ny);
      });
    });
    // Kokkos::fence();
}

// TODO remove, not needed in SYCL
// void copy_buffer_SEND(const size_t len, Kokkos::View<double*>::HostMirror* src, Kokkos::View<double*>* dst) {
//     deep_copy(*dst, *src);
// }
//
// void copy_float_buffer_SEND(const size_t len, Kokkos::View<float*>::HostMirror* src, Kokkos::View<float*>* dst) {
//     deep_copy(*dst, *src);
// }
//
// void copy_int_buffer_SEND(const size_t len, Kokkos::View<int*>::HostMirror* src, Kokkos::View<int*>* dst) {
//     deep_copy(*dst, *src);
// }
//
// void copy_buffer_RECEIVE(const size_t len, Kokkos::View<double*>* src, Kokkos::View<double*>::HostMirror* dst) {
//     deep_copy(*dst, *src);
// }

void move_host_buffer_to_device(cl::sycl::queue queue,
                                const size_t len,
                                double* src,
                                cl::sycl::buffer<double, 1>* dst) {
  allocate_data_w_host(queue, &dst, src, len);
}

// Initialise state data in device specific manner
void set_problem_2d(cl::sycl::queue queue,
                    const int local_nx, const int local_ny, const int pad,
                    const double mesh_width, const double mesh_height,
                    cl::sycl::buffer<double, 1>* edgex,
                    cl::sycl::buffer<double, 1>* edgey,
                    const int ndims,
                    const char* problem_def_filename,
                    cl::sycl::buffer<double, 1>* density,
                    cl::sycl::buffer<double, 1>* energy,
                    cl::sycl::buffer<double, 1>* temperature) {

    int* h_keys;
    cl::sycl::buffer<int, 1>* d_keys;
    allocate_host_int_data(&h_keys, MAX_KEYS);

    double* h_values;
    cl::sycl::buffer<double, 1>* d_values;
    allocate_host_data(&h_values, MAX_KEYS);

    int nentries = 0;
    while (1) {
        char specifier[MAX_STR_LEN];
        char keys[MAX_STR_LEN * MAX_KEYS];
        sprintf(specifier, "problem_%d", nentries++);

        int nkeys = 0;
    if (!get_key_value_parameter(specifier, problem_def_filename, keys,
          h_values, &nkeys)) {
      break;
    }

    // The last four keys are the bound specification
    double xpos = h_values[nkeys - 4] * mesh_width;
    double ypos = h_values[nkeys - 3] * mesh_height;
    double width = h_values[nkeys - 2] * mesh_width;
    double height = h_values[nkeys - 1] * mesh_height;

    for (int kk = 0; kk < nkeys - (2 * ndims); ++kk) {
      const char* key = &keys[kk * MAX_STR_LEN];
      if (strmatch(key, "density")) {
        h_keys[kk] = DENSITY_KEY;
      } else if (strmatch(key, "energy")) {
        h_keys[kk] = ENERGY_KEY;
      } else if (strmatch(key, "temperature")) {
        h_keys[kk] = TEMPERATURE_KEY;
      } else {
        TERMINATE("Found unrecognised key in %s : %s.\n", problem_def_filename,
            key);
      }
    }

    // copy_int_buffer_SEND(MAX_KEYS, &h_keys, &d_keys);
    // copy_buffer_SEND(MAX_KEYS, &h_values, &d_values);

    allocate_int_data_w_host(queue, &d_keys, h_keys, MAX_KEYS);
    allocate_data_w_host(queue, &d_values, h_values, MAX_KEYS);

    auto d_keys_acc = d_keys->get_access<cl::sycl::access::mode::write>();
    auto d_values_acc = d_values->get_access<cl::sycl::access::mode::write>();
    for (int kk = 0; kk < MAX_KEYS; ++kk) {
      d_keys_acc[kk] = h_keys[kk];
      d_values_acc[kk] = h_values[kk];
    }

    queue.submit([&] (cl::sycl::handler& cgh) {
      auto edgex_acc = edgex->get_access<cl::sycl::access::mode::read>(cgh);
      auto edgey_acc = edgey->get_access<cl::sycl::access::mode::read>(cgh);
      auto d_keys_acc = d_keys->get_access<cl::sycl::access::mode::read>(cgh);
      auto d_values_acc = d_values->get_access<cl::sycl::access::mode::read>(cgh);
      auto density_acc = density->get_access<cl::sycl::access::mode::write>(cgh);
      auto energy_acc = energy->get_access<cl::sycl::access::mode::write>(cgh);
      auto temperature_acc = temperature->get_access<cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class set_problem_2d_kernel>(cl::sycl::range<1>(local_nx*local_ny), [=](cl::sycl::id<1> idx) {
        const int ii = idx[0] / local_nx;
        const int jj = idx[0] % local_nx;
        double global_xpos = edgex_acc[jj];
        double global_ypos = edgey_acc[ii];

        // Check we are in bounds of the problem entry
        if (global_xpos >= xpos &&
            global_ypos >= ypos &&
            global_xpos < xpos + width &&
            global_ypos < ypos + height) {

            // The upper bound excludes the bounding box for the entry
            for (int nn = 0; nn < nkeys - (2 * ndims); ++nn) {
                const int key = d_keys_acc[nn];
                if (key == DENSITY_KEY) {
                    density_acc[idx] = d_values_acc[nn];
                } else if (key == ENERGY_KEY) {
                    energy_acc[idx] = d_values_acc[nn];
                } else if (key == TEMPERATURE_KEY) {
                    temperature_acc[idx] = d_values_acc[nn];
                }
            }
        }
      });
    });
  }
}
