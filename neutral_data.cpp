#include "neutral_data.h"
#include "params.h"
#include "profiler.h"
#include "shared.h"
#include "neutral_interface.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

class init_ranks_kernel;

#define max(a, b) (((a) > (b)) ? (a) : (b))

// Initialises the set of cross sections
void initialise_cross_sections(cl::sycl::queue queue, NeutralData* neutral_data, Mesh* mesh);

// Initialises all of the neutral-specific data structures.
void initialise_neutral_data(cl::sycl::queue queue, NeutralData* neutral_data, Mesh* mesh) {
  const int pad = mesh->pad;
  const int local_nx = mesh->local_nx - 2 * pad;
  const int local_ny = mesh->local_ny - 2 * pad;

  neutral_data->nparticles =
      get_int_parameter("nparticles", neutral_data->neutral_params_filename);
  neutral_data->initial_energy = get_double_parameter(
      "initial_energy", neutral_data->neutral_params_filename);

  int nkeys = 0;
  char* keys = (char*)malloc(sizeof(char) * MAX_KEYS * MAX_STR_LEN);
  double* values = (double*)malloc(sizeof(double) * MAX_KEYS);

  if (!get_key_value_parameter_double("source", neutral_data->neutral_params_filename,
                               keys, values, &nkeys)) {
    TERMINATE("Parameter file %s did not contain a source entry.\n",
              neutral_data->neutral_params_filename);
  }

  // The last four keys are the bound specification
  const double source_xpos = values[nkeys - 4] * mesh->width;
  const double source_ypos = values[nkeys - 3] * mesh->height;
  const double source_width = values[nkeys - 2] * mesh->width;
  const double source_height = values[nkeys - 1] * mesh->height;

  double rank_xpos_0 =10;
  double rank_ypos_0 =11;
  double rank_xpos_1 =60;
  double rank_ypos_1 =75;
  auto edgex_acc = mesh->edgex->get_access<cl::sycl::access::mode::read>();
  auto edgey_acc = mesh->edgey->get_access<cl::sycl::access::mode::read>();
  rank_xpos_0 = edgex_acc[mesh->x_off + pad];
  rank_ypos_0 = edgey_acc[mesh->y_off + pad];
  rank_xpos_1 = edgex_acc[local_nx + mesh->x_off + pad];
  rank_ypos_1 = edgey_acc[local_ny + mesh->y_off + pad];

  // Calculate the shaded bounds
  const double local_particle_left_off = max(0.0, source_xpos - rank_xpos_0);
  const double local_particle_bottom_off = max(0.0, source_ypos - rank_ypos_0);
  const double local_particle_right_off =
      max(0.0, rank_xpos_1 - (source_xpos + source_width));
  const double local_particle_top_off =
      max(0.0, rank_ypos_1 - (source_ypos + source_height));

  const double local_particle_width =
      max(0.0, (rank_xpos_1 - rank_xpos_0) -
                   (local_particle_right_off + local_particle_left_off));
  const double local_particle_height =
      max(0.0, (rank_ypos_1 - rank_ypos_0) -
                   (local_particle_top_off + local_particle_bottom_off));

  // Calculate the number of particles we need based on the shaded area that
  // is covered by our source
  const double nlocal_particles_real =
      neutral_data->nparticles *
      (local_particle_width * local_particle_height) /
      (source_width * source_height);

  // Rounding hack to make sure correct number of particles is selected
  neutral_data->nlocal_particles = nlocal_particles_real + 0.5;

  allocate_host_data(&(neutral_data->h_energy_deposition_tally), local_nx * local_ny);

  size_t allocation = allocate_data_w_host(queue, &(neutral_data->energy_deposition_tally),
                                    neutral_data->h_energy_deposition_tally, local_nx * local_ny);

  allocation += allocate_uint64_data(queue, &(neutral_data->nfacets_reduce_array),
                                     neutral_data->nparticles);
  allocation += allocate_uint64_data(queue, &(neutral_data->ncollisions_reduce_array),
                                     neutral_data->nparticles);
  allocation += allocate_uint64_data(queue, &(neutral_data->nprocessed_reduce_array),
                                     neutral_data->nparticles);

  // Inject some particles into the mesh if we need to
  if (neutral_data->nlocal_particles) {
    printf("Allocated %.4fGB of data (injecting more).\n", allocation / GB);
    allocation += inject_particles( queue,
        neutral_data->nparticles, mesh->global_nx, mesh->local_nx,
        mesh->local_ny, pad, local_particle_left_off, local_particle_bottom_off,
        local_particle_width, local_particle_height, mesh->x_off, mesh->y_off,
        mesh->dt, mesh->edgex, mesh->edgey, neutral_data->initial_energy,
        &(neutral_data->local_particles));
  }

  printf("Allocated %.4fGB of data.\n", allocation / GB);

  initialise_cross_sections(queue, neutral_data, mesh);
}

// Reads in a cross-sectional data file
void read_cs_file(cl::sycl::queue queue, const char* filename, CrossSection* cs, Mesh* mesh) {
  FILE* fp = fopen(filename, "r");
  if (!fp) {
    TERMINATE("Could not open the cross section file: %s\n", filename);
  }

  // Count the number of entries in the file
  int ch;
  cs->nentries = 0;
  while ((ch = fgetc(fp)) != EOF) {
    if (ch == '\n') {
      cs->nentries++;
    }
  }

  if (mesh->rank == MASTER) {
    printf("File %s contains %d entries\n", filename, cs->nentries);
  }

  rewind(fp);

  double* h_keys;
  double* h_values;
  allocate_host_data(&h_keys, cs->nentries);
  allocate_host_data(&h_values, cs->nentries);

  for (int ii = 0; ii < cs->nentries; ++ii) {
    // Skip whitespace tokens
    while ((ch = fgetc(fp)) == ' ' || ch == '\n' || ch == '\r') {
    };

    // Jump out if we reach the end of the file early
    if (ch == EOF) {
      cs->nentries = ii;
      break;
    }

    ungetc(ch, fp);
    fscanf(fp, "%lf", &h_keys[ii]);
    while ((ch = fgetc(fp)) == ' ') {
    };
    ungetc(ch, fp);
    fscanf(fp, "%lf", &h_values[ii]);
  }

  allocate_data(queue, &(cs->keys), cs->nentries);
  allocate_data(queue, &(cs->values), cs->nentries);

  auto cs_keys_acc = cs->keys->get_access<cl::sycl::access::mode::write>();
  auto cs_values_acc = cs->values->get_access<cl::sycl::access::mode::write>();
  for (int kk = 0; kk < cs->nentries; ++kk) {
    cs_keys_acc[kk] = h_keys[kk];
    cs_values_acc[kk] = h_values[kk];
  }
}

// Initialises the state
void initialise_cross_sections(cl::sycl::queue queue, NeutralData* neutral_data, Mesh* mesh) {
  neutral_data->cs_scatter_table = (CrossSection*)malloc(sizeof(CrossSection));
  neutral_data->cs_absorb_table = (CrossSection*)malloc(sizeof(CrossSection));

  read_cs_file(queue, CS_SCATTER_FILENAME, neutral_data->cs_scatter_table, mesh);
  read_cs_file(queue, CS_CAPTURE_FILENAME, neutral_data->cs_absorb_table, mesh);
}
