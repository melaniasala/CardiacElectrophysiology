# Parallel Finite Element Simulation of Cardiac Electrophysiology of the Heart

## Project Description
This project, conducted at Politecnico di Milano during the academic year 2022/23 under the supervision of Prof. L. Dede', focuses on solving the cardiac electrophysiology problem using numerical methods, specifically the Finite Element method. The aim is to solve the monodomain equation coupled with the Bueno-Orovio ionic model for the evolution of certain variables within a 3D slab domain and an idealized prolated ellipsoid.

The problem was solved in a parallel setting utilizing the HPC infrastructure provided by MOX Mathematics Department, Politecnico di Milano, to enhance computational efficiency.

## Objectives
- Utilize the Finite Element method to solve the monodomain equation coupled with the Bueno-Orovio ionic model.
- Select an appropriate method for coupling the monodomain and ionic models based on computational costs, solver stability, and accuracy.
- Compare results obtained for different mesh and time step sizes.
- Discuss the methods used, their accuracy, and computational aspects.

## Components
- **Software Libraries**:
  - **gmsh**: Utilized for mesh generation.
  - **deal.II**: Used for Finite Element computations.
  - **ParaView**: Employed for visualization of results.
- **HPC Infrastructure**: Utilized the MOX Mathematics Department's HPC infrastructure for parallel computation.

## References
- Bueno-Orovio A, Cherry EM, Fenton FH. Minimal model for human ventricular action potentials in tissue. J Theor Biol. 2008 Aug 7;253(3):544-60. doi: 10.1016/j.jtbi.2008.03.029. Epub 2008 Apr 8. PMID: 18495166.
- Salvador, M., Dede’, L. & Quarteroni, A. An intergrid transfer operator using radial basis functions with application to cardiac electromechanics. Comput Mech 66, 491–511 (2020). https://doi.org/10.1007/s00466-020-01861-x
- Niederer SA, Kerfoot E, Benson AP, et al. Verification of cardiac tissue electrophysiology simulators using an N-version benchmark. Philos Trans A Math Phys Eng Sci. 2011;369(1954):4331-4351. doi:10.1098/rsta.2011.0139
