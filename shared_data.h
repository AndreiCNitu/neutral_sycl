#ifndef __SHAREDDATAHDR
#define __SHAREDDATAHDR

#include "mesh.h"

#ifdef __cplusplus
extern "C" {
#endif

// TODO: MAKE IT SO THAT shared_data IS LOCAL TO THE APPLICATIONS???

// Contains all of the shared_data information for the solver
typedef struct {
  // Shared shared_data (share data)
  cl::sycl::buffer<double, 1>* density; // Density
  cl::sycl::buffer<double, 1>* energy;  // Energy

  // Paired shared_data (share capacity)
  cl::sycl::buffer<double, 1>* Ap;          // HOT: Coefficient matrix A, by conjugate vector p
  cl::sycl::buffer<double, 1>* density_old; // FLOW: Density at beginning of timestep

  cl::sycl::buffer<double, 1>* s_x; // HOT: Coefficients in temperature direction
  cl::sycl::buffer<double, 1>* Qxx; // FLOW: Artificial viscous term in temperature direction

  cl::sycl::buffer<double, 1>* s_y; // HOT: Coefficients in y direction
  cl::sycl::buffer<double, 1>* Qyy; // FLOW: Artificial viscous term in y direction

  cl::sycl::buffer<double, 1>* s_z; // HOT: Coefficients in z direction
  cl::sycl::buffer<double, 1>* Qzz; // FLOW: Artificial viscous term in z direction

  cl::sycl::buffer<double, 1>* r;        // HOT: The residual vector
  cl::sycl::buffer<double, 1>* pressure; // FLOW: The pressure

  cl::sycl::buffer<double, 1>* temperature; // HOT: The solution vector (new energy)
  cl::sycl::buffer<double, 1>* u;           // FLOW: The velocity in the temperature direction

  cl::sycl::buffer<double, 1>* p; // HOT: The conjugate vector
  cl::sycl::buffer<double, 1>* v; // FLOW: The velocity in the y direction

  cl::sycl::buffer<double, 1>* reduce_array0;
  cl::sycl::buffer<double, 1>* reduce_array1;

} SharedData;

// Initialises the shared_data variables
void initialise_shared_data_2d(cl::sycl::queue queue,
                               const int local_nx, const int local_ny,
                               const int pad, const double mesh_width,
                               const double mesh_height,
                               const char* problem_def_filename,
                               cl::sycl::buffer<double, 1>* edgex,
                               cl::sycl::buffer<double, 1>* edgey,
                               SharedData* shared_data);

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
                    cl::sycl::buffer<double, 1>* temperature);

// // Initialises the shared_data variables
// void initialise_shared_data_3d(
//     const int local_nx, const int local_ny, const int local_nz, const int pad,
//     const double mesh_width, const double mesh_height, const double mesh_depth,
//     const char* problem_def_filename, const double* edgex, const double* edgey,
//     const double* edgez, SharedData* shared_data);

// void set_problem_3d(const int local_nx, const int local_ny, const int local_nz,
//                     const int pad, const double mesh_width,
//                     const double mesh_height, const double mesh_depth,
//                     const double* edgex, const double* edgey,
//                     const double* edgez, const int ndims,
//                     const char* problem_def_filename, double* density,
//                     double* energy, double* temperature);

// // Deallocate all of the shared_data memory
// void finalise_shared_data(SharedData* shared_data);

#ifdef __cplusplus
}
#endif

#endif
