#include "mesh.h"
#include "params.h"
#include "profiler.h"
#include "neutral_data.h"
#include "neutral_interface.h"
#include "shared.h"
#include "shared_data.h"
#include "comms.h"
#include <sys/time.h>
#include <sys/resource.h>

#ifdef MPI
#include "mpi.h"
#endif
#define MASTER 0

int main(int argc, char *argv[])
{
  if (argc != 2) {
    TERMINATE("usage: ./neutral.exe <param_file>\n");
  }

  int rank = MASTER;
  int nranks = 1;
  initialise_mpi(argc, argv, &rank, &nranks);

  cl::sycl::default_selector device_selector;

  auto exception_handler = [] (cl::sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch(cl::sycl::exception const& e) {
        std::cout << "Caught asynchronous SYCL exception:"
                  << std::endl << e.what() << std::endl;
      }
    }
  };

  cl::sycl::queue queue(device_selector, exception_handler, {});
  std::cout << "Running on "
            << queue.get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;

  // Store the dimensions of the mesh
  Mesh mesh;
  NeutralData neutral_data;
  neutral_data.neutral_params_filename = argv[1];
  mesh.global_nx =
      get_int_parameter("nx", neutral_data.neutral_params_filename);
  mesh.global_ny =
      get_int_parameter("ny", neutral_data.neutral_params_filename);
  mesh.pad = 0;
  mesh.local_nx = mesh.global_nx + 2 * mesh.pad;
  mesh.local_ny = mesh.global_ny + 2 * mesh.pad;
  mesh.width = get_double_parameter("width", ARCH_ROOT_PARAMS);
  mesh.height = get_double_parameter("height", ARCH_ROOT_PARAMS);
  mesh.dt = get_double_parameter("dt", neutral_data.neutral_params_filename);
  mesh.sim_end = get_double_parameter("sim_end", ARCH_ROOT_PARAMS);
  mesh.niters =
      get_int_parameter("iterations", neutral_data.neutral_params_filename);
  mesh.rank = rank;
  mesh.nranks = nranks;
  mesh.ndims = 2;

  SharedData shared_data;

auto num_groups = queue.get_device().get_info<cl::sycl::info::device::max_compute_units>();
auto work_group_size = queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
auto total_threads = num_groups * work_group_size;
neutral_data.nthreads = total_threads;
  {

  printf("Starting up with %d threads.\n", neutral_data.nthreads);
  printf("Loading problem from %s.\n", neutral_data.neutral_params_filename);
#ifdef ENABLE_PROFILING
  // The timing code has to be called so many times that the API calls
  // actually begin to influence the performance dramatically.
  fprintf(stderr,
          "Warning. Profiling is enabled and will increase the runtime.\n\n");
#endif

  // Perform the general initialisation steps for the mesh etc
  initialise_comms(&mesh);
  initialise_mesh_2d(queue, &mesh);
  initialise_shared_data_2d(queue, mesh.local_nx, mesh.local_ny, mesh.pad, mesh.width,
      mesh.height, neutral_data.neutral_params_filename, mesh.edgex, mesh.edgey, &shared_data);

  handle_boundary_2d(queue, mesh.local_nx, mesh.local_ny, &mesh, shared_data.density,
                     NO_INVERT, PACK);

  initialise_neutral_data(queue, &neutral_data, &mesh);

 // Main timestep loop where we will track each particle through time
  int tt;
  double wallclock = 0.0;
  double elapsed_sim_time = 0.0;
  for (tt = 1; tt <= mesh.niters; ++tt) {

    if (mesh.rank == MASTER) {
      printf("\nIteration  %d\n", tt);
    }

    uint64_t facet_events = 0;
    uint64_t collision_events = 0;

    struct  timeval timstr; /* structure to hold elapsed time */
    struct  rusage ru;      /* structure to hold CPU time--system and user */
    double  tic, toc;       /* floating point numbers to calculate elapsed wallclock time */
    double  usrtim;         /* floating point number to record elapsed user CPU time */
    double  systim;         /* floating point number to record elapsed system CPU time */

    gettimeofday(&timstr, NULL);
    double w0 = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

    // Begin the main solve step
    solve_transport_2d(
        mesh.local_nx - 2 * mesh.pad, mesh.local_ny - 2 * mesh.pad,
        mesh.global_nx, mesh.global_ny, tt, mesh.pad, mesh.x_off, mesh.y_off,
        mesh.dt, neutral_data.nparticles, &neutral_data.nlocal_particles,
        neutral_data.local_particles,
        shared_data.density, mesh.edgex, mesh.edgey,
        &(neutral_data.cs_scatter_table), &(neutral_data.cs_absorb_table),
        neutral_data.energy_deposition_tally,
        &facet_events, &collision_events, queue);

    gettimeofday(&timstr, NULL);
    double step_time = timstr.tv_sec + (timstr.tv_usec / 1000000.0) - w0;
    wallclock += step_time;
    printf("Step time  %.4fs\n", step_time);
    printf("Wallclock  %.4fs\n", wallclock);
    printf("Facets     %lu\n", facet_events);
    printf("Collisions %lu\n", collision_events);

    elapsed_sim_time += mesh.dt;

    // Leave the simulation if we have reached the simulation end time
    if (elapsed_sim_time >= mesh.sim_end) {
      if (mesh.rank == MASTER)
        printf("Reached end of simulation time\n");
      break;
    }
  }

  if (mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&p);

    printf("Final Wallclock %.9fs\n", wallclock);
  }
    validate(mesh.local_nx - 2 * mesh.pad, mesh.local_ny - 2 * mesh.pad,
           neutral_data.neutral_params_filename, mesh.rank,
           neutral_data.energy_deposition_tally);
}


  return 0;
}
