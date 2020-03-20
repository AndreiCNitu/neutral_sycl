#include "comms.h"
#include "mesh.h"
#include "shared.h"

// Enforce reflective boundary conditions on the problem state
void handle_boundary_2d(const int nx, const int ny, Mesh* mesh, Kokkos::View<double *> arr,
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
      Kokkos::parallel_for(Kokkos::RangePolicy< >( pad, nx-pad ), KOKKOS_LAMBDA (int jj) {
        arr[(ny - pad + dd) * nx + jj] =
            y_inversion_coeff * arr[(ny - 1 - pad - dd) * nx + jj];
      });
    }
  }
  // reflect at the south
  if (neighbours[SOUTH] == EDGE) {
    for (int dd = 0; dd < pad; ++dd) {
      Kokkos::parallel_for(Kokkos::RangePolicy< >( pad, nx-pad ), KOKKOS_LAMBDA (int jj) {
        arr[(pad - 1 - dd) * nx + jj] =
            y_inversion_coeff * arr[(pad + dd) * nx + jj];
      });
    }
  }
  // reflect at the east
  if (neighbours[EAST] == EDGE) {
      Kokkos::parallel_for(Kokkos::RangePolicy< >( pad, ny-pad ), KOKKOS_LAMBDA (int ii) {
      for (int dd = 0; dd < pad; ++dd) {
        arr[ii * nx + (nx - pad + dd)] =
            x_inversion_coeff * arr[ii * nx + (nx - 1 - pad - dd)];
      }
    });
  }
  if (neighbours[WEST] == EDGE) {
// reflect at the west
      Kokkos::parallel_for(Kokkos::RangePolicy< >( pad, ny-pad ), KOKKOS_LAMBDA (int ii) {
      for (int dd = 0; dd < pad; ++dd) {
        arr[ii * nx + (pad - 1 - dd)] =
            x_inversion_coeff * arr[ii * nx + (pad + dd)];
      }
    });
  }
  STOP_PROFILING(&comms_profile, __func__);
}
