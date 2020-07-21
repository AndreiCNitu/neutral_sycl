#include "shared.h"
#include "neutral_interface.h"

using accessor_t =
  cl::sycl::accessor<double, 1,
                 cl::sycl::access::mode::read_write,
                 cl::sycl::access::target::global_buffer>;
using read_accessor_t =
  cl::sycl::accessor<double, 1,
                 cl::sycl::access::mode::read,
                 cl::sycl::access::target::global_buffer>;
using local_accessor_t =
  cl::sycl::accessor<uint64_t, 1,
                 cl::sycl::access::mode::read_write,
                 cl::sycl::access::target::local>;
using particle_accessor_t =
  cl::sycl::accessor<Particle, 1,
                 cl::sycl::access::mode::read_write,
                 cl::sycl::access::target::global_buffer>;

// Handle facet event
inline int facet_event(const int global_nx, const int global_ny, const int nx,
                const int ny, const int x_off, const int y_off,
                const double inv_ntotal_particles, const double distance_to_facet,
                const double speed, const double cell_mfp, const int x_facet,
                read_accessor_t density_acc,
                const int* neighbours,
                particle_accessor_t particles_acc,
                double* energy_deposition,
                double* number_density,
                double* microscopic_cs_scatter,
                double* microscopic_cs_absorb,
                double* macroscopic_cs_scatter,
                double* macroscopic_cs_absorb,
                accessor_t energy_deposition_tally,
                int* cellx,
                int* celly,
                double* local_density,
                cl::sycl::id<1> idx);

// Handles a collision event
inline int collision_event(
    const int global_nx, const int nx, const int x_off, const int y_off,
    const uint64_t pkey, const uint64_t master_key,
    const double inv_ntotal_particles, const double distance_to_collision,
    const double local_density,
    read_accessor_t cs_scatter_keys_acc,
    read_accessor_t cs_scatter_values_acc,
    const int cs_scatter_nentries,
    read_accessor_t cs_absorb_keys_acc,
    read_accessor_t cs_absorb_values_acc,
    const int cs_absorb_nentries,
    particle_accessor_t particles_acc,
    uint64_t* counter,
    double* energy_deposition,
    double* number_density,
    double* microscopic_cs_scatter,
    double* microscopic_cs_absorb,
    double* macroscopic_cs_scatter,
    double* macroscopic_cs_absorb,
    accessor_t energy_deposition_tally_acc,
    int* scatter_cs_index,
    int* absorb_cs_index,
    double rn[NRANDOM_NUMBERS],
    double* speed,
    cl::sycl::id<1> idx);


inline void census_event(const int global_nx, const int nx, const int x_off,
                  const int y_off, const double inv_ntotal_particles,
                  const double distance_to_census, const double cell_mfp,
                  particle_accessor_t particles_acc,
                  double* energy_deposition,
                  double* number_density,
                  double* microscopic_cs_scatter,
                  double* microscopic_cs_absorb,
                  accessor_t energy_deposition_tally_acc,
                  cl::sycl::id<1> idx);

// Tallies the energy deposition in the cell
inline void update_tallies(const int nx, const int x_off, const int y_off,
                                Particle particle,
                                const double inv_ntotal_particles,
                                const double energy_deposition,
                                accessor_t energy_deposition_tally_acc);

// // Calculate the distance to the next facet
inline void calc_distance_to_facet(const int global_nx, const double x, const double y,
                            const int pad, const int x_off, const int y_off,
                            const double omega_x, const double omega_y,
                            const double speed, const int particle_cellx,
                            const int particle_celly, double* distance_to_facet,
                            int* x_facet,
                            read_accessor_t edgex_acc,
                            read_accessor_t edgey_acc);

// Calculate the energy deposition in the cell
inline double calculate_energy_deposition(
    const int global_nx, const int nx, const int x_off, const int y_off,
    Particle particle, const double inv_ntotal_particles,
    const double path_length, const double number_density,
    const double microscopic_cs_absorb, const double microscopic_cs_total);

// Fetch the cross section for a particular energy value
inline double microscopic_cs_for_energy(read_accessor_t keys,
                                        read_accessor_t values,
                                        const int nentries,
                                        const double energy,
                                        int* cs_index);

inline void generate_random_numbers(const uint64_t pkey, const uint64_t master_key,
                             const uint64_t counter, double* rn0, double* rn1);
