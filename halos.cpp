#include "comms.h"
#include "mesh.h"
#include "shared.h"

class reflect_north_kernel;
class reflect_south_kernel;
class reflect_east_kernel;
class reflect_west_kernel;


// Enforce reflective boundary conditions on the problem state
void handle_boundary_2d(cl::sycl::queue queue,
                        const int nx, const int ny,
                        Mesh* mesh, cl::sycl::buffer<double, 1>* arr,
                        const int invert, const int pack) {
  START_PROFILING(&comms_profile);

  const int pad = mesh->pad;
  int* neighbours = mesh->neighbours;

  // Perform the boundary reflections, potentially with the data updated from
  // neighbours
  double x_inversion_coeff = (invert == INVERT_X) ? -1.0 : 1.0;
  double y_inversion_coeff = (invert == INVERT_Y) ? -1.0 : 1.0;

   // Reflect at the north
  if (neighbours[NORTH] == EDGE) {
    for (int dd = 0; dd < pad; ++dd) {
      queue.submit([&] (cl::sycl::handler& cgh) {
        auto arr_acc = arr->get_access<cl::sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class reflect_north_kernel>(cl::sycl::range<1>(nx-2*pad), [=](cl::sycl::id<1> idx) {
          arr_acc[(ny - pad + dd) * nx + idx[0] + pad] =
              y_inversion_coeff * arr_acc[(ny - 1 - pad - dd) * nx + idx[0] + pad];
        });
      });
    }
  }
  // reflect at the south
  if (neighbours[SOUTH] == EDGE) {
    for (int dd = 0; dd < pad; ++dd) {

      queue.submit([&] (cl::sycl::handler& cgh) {
        auto arr_acc = arr->get_access<cl::sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class reflect_south_kernel>(cl::sycl::range<1>(nx-2*pad), [=](cl::sycl::id<1> idx) {
          arr_acc[(pad - 1 - dd) * nx + idx[0] + pad] =
              y_inversion_coeff * arr_acc[(pad + dd) * nx + idx[0] + pad];
        });
      });
    }
  }
  // reflect at the east
  if (neighbours[EAST] == EDGE) {
    queue.submit([&] (cl::sycl::handler& cgh) {
      auto arr_acc = arr->get_access<cl::sycl::access::mode::read_write>(cgh);

      cgh.parallel_for<class reflect_east_kernel>(cl::sycl::range<1>(ny-2*pad), [=](cl::sycl::id<1> idx) {
        for (int dd = 0; dd < pad; ++dd) {
          arr_acc[(idx[0] + pad) * nx + (nx - pad + dd)] =
              x_inversion_coeff * arr_acc[(idx[0] + pad) * nx + (nx - 1 - pad - dd)];
        }
      });
    });
  }
  // reflect at the west
  if (neighbours[WEST] == EDGE) {
    queue.submit([&] (cl::sycl::handler& cgh) {
      auto arr_acc = arr->get_access<cl::sycl::access::mode::read_write>(cgh);

      cgh.parallel_for<class reflect_west_kernel>(cl::sycl::range<1>(ny-2*pad), [=](cl::sycl::id<1> idx) {
        for (int dd = 0; dd < pad; ++dd) {
          arr_acc[(idx[0] + pad) * nx + (pad - 1 - dd)] =
              x_inversion_coeff * arr_acc[(idx[0] + pad) * nx + (pad + dd)];
        }
      });
    });
  }
  STOP_PROFILING(&comms_profile, __func__);
}
