#include "mesh.h"
#include "params.h"
#include "shared.h"
#include <assert.h>
#include <stdlib.h>

// Initialise the mesh describing variables
void initialise_mesh_2d(cl::sycl::queue queue, Mesh* mesh) {
  allocate_data(queue, &(mesh->edgex), (mesh->local_nx + 1));
  allocate_data(queue, &(mesh->edgey), (mesh->local_ny + 1));
  allocate_data(queue, &(mesh->edgedx), (mesh->local_nx + 1));
  allocate_data(queue, &(mesh->edgedy), (mesh->local_ny + 1));
  allocate_data(queue, &(mesh->celldx), (mesh->local_nx + 1));
  allocate_data(queue, &(mesh->celldy), (mesh->local_ny + 1));

  mesh_data_init_2d(queue,
                    mesh->local_nx, mesh->local_ny, mesh->global_nx,
                    mesh->global_ny, mesh->pad, mesh->x_off, mesh->y_off,
                    mesh->width, mesh->height, mesh->edgex, mesh->edgey,
                    mesh->edgedx, mesh->edgedy, mesh->celldx, mesh->celldy);
}

// TODO
// Deallocate all of the mesh memory
// void finalise_mesh(Mesh* mesh) {
//   deallocate_data(mesh->edgedy);
//   deallocate_data(mesh->celldy);
//   deallocate_data(mesh->edgedx);
//   deallocate_data(mesh->celldx);
//   deallocate_data(mesh->north_buffer_out);
//   deallocate_data(mesh->east_buffer_out);
//   deallocate_data(mesh->south_buffer_out);
//   deallocate_data(mesh->west_buffer_out);
//   deallocate_data(mesh->north_buffer_in);
//   deallocate_data(mesh->east_buffer_in);
//   deallocate_data(mesh->south_buffer_in);
//   deallocate_data(mesh->west_buffer_in);
// }
