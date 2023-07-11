#include "CardiacElectrophysiology.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);// MPI_Init(&argc, &argv);
  
  // default values
  //const unsigned int N      = 19;
  const unsigned int degree = 1;
  double T      = 1.5;
  double deltat = 5e-5; // 0.05 ms
  short int tissue_type = 0; // epicardium
  short int mesh_type = 0; // slab small


  // Process command line arguments
  for (int i = 1; i < argc; ++i)
  {
      if (std::string(argv[i]) == "-tt" && i + 1 < argc) // tissue type
      {
          tissue_type = std::stoi(argv[i + 1]);
          ++i;
      }
      else if (std::string(argv[i]) == "-mt" && i + 1 < argc) // mesh type
      {
          mesh_type = std::stoi(argv[i + 1]);
          ++i;
      }
      else if (std::string(argv[i]) == "-T" && i + 1 < argc){ // Total time
          T = std::stod(argv[i + 1]);
          ++i;
      }
      else if (std::string(argv[i]) == "-dT" && i + 1 < argc){
          deltat = std::stod(argv[i + 1]);
          ++i;
      }
  }

  
  BuenoOrovioModel problem(degree, T, deltat, tissue_type, mesh_type);  
    
  problem.setup();
  problem.solve();

  // MPI_Finalize();
  return 0;
}
