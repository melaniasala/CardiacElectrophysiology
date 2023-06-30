#include "CardiacElectrophysiology.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int N      = 19;
  const unsigned int degree = 1;

  const double T      = 0.02085;
  const double deltat = 0.0005;
  const std::string tissue_type = "epicardium"; // da terminale (?)

  BuenoOrovioModel problem(N, degree, T, deltat, tissue_type);

  problem.setup();
  problem.solve();

  return 0;
}