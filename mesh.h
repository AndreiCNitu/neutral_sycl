#ifndef __MESHHDR
#define __MESHHDR

#include <CL/sycl.hpp>

/* Problem-Independent Constants */
#define LOAD_BALANCE 0 // Whether decomposition should attempt to load balance
#define NNEIGHBOURS 6  // This is max size required - for 3d

#ifdef __cplusplus
extern "C" {
#endif

/*
 * STRUCTURED MESHES
 */

// Contains all of the data regarding a particular mesh
typedef struct {
  int local_nx;  // Number of cells in rank in x direction (inc. halo)
  int local_ny;  // Number of cells in rank in y direction (inc. halo)
  int local_nz;  // Number of cells in rank in z direction (inc. halo)
  int global_nx; // Number of cells globally in x direction (exc. halo)
  int global_ny; // Number of cells globally in x direction (exc. halo)
  int global_nz; // Number of cells globally in x direction (exc. halo)
  int niters;    // Number of timestep iterations
  double width;  // Width of the problem domain
  double height; // Height of the problem domain
  double depth;  // Depth of the problem domain

  int pad;

  // Mesh differentials
  cl::sycl::buffer<double, 1>* edgex;
  cl::sycl::buffer<double, 1>* edgey;
  cl::sycl::buffer<double, 1>* edgez;

  cl::sycl::buffer<double, 1>* edgedx;
  cl::sycl::buffer<double, 1>* edgedy;
  cl::sycl::buffer<double, 1>* edgedz;

  cl::sycl::buffer<double, 1>* celldx;
  cl::sycl::buffer<double, 1>* celldy;
  cl::sycl::buffer<double, 1>* celldz;

  // Offset in global mesh of rank owning this local mesh
  int x_off;
  int y_off;
  int z_off;

  // Timesteps
  double dt;
  double dt_h;
  double max_dt;
  double sim_end;

  int rank;                    // Rank that owns this mesh object
  int ranks_x;                 // The number of ranks in the x dimension
  int ranks_y;                 // The number of ranks in the y dimension
  int ranks_z;                 // The number of ranks in the z dimension
  int nranks;                  // Total number of ranks that exist
  int neighbours[NNEIGHBOURS]; // List of neighbours
  int ndims;                   // The number of dimensions
} Mesh;

// Initialises the mesh
void initialise_mesh_2d(cl::sycl::queue queue, Mesh* mesh);
void mesh_data_init_2d(cl::sycl::queue queue,
                       const int local_nx, const int local_ny,
                       const int global_nx, const int global_ny, const int pad,
                       const int x_off, const int y_off, const double width,
                       const double height, cl::sycl::buffer<double, 1>* edgex, cl::sycl::buffer<double, 1>* edgey,
                       cl::sycl::buffer<double, 1>* edgedx, cl::sycl::buffer<double, 1>* edgedy,
                       cl::sycl::buffer<double, 1>* celldx, cl::sycl::buffer<double, 1>* celldy);

#ifdef __cplusplus
}
#endif

#endif
