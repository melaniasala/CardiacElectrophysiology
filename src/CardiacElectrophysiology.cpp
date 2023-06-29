#include "CardiacElectrophysiology.hpp"

void
BuenoOrovioModel::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    // TODO: genetate mesh
    const std::string mesh_file_name = "./mesh/rectangular_slab.msh"; // "./mesh/rectangular_slab_big.msh"
      //"../mesh/mesh-cube-" + std::to_string(N + 1) + ".msh";

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    // FE_SimplexP<dim> fe_scalar(r);
    // fe_ionic = std::make_unique<FESystem<dim>>(fe_scalar, dim_ionic);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler for the potential problem." << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // // Initialize the DoF handler.
  // {
  //   pcout << "Initializing the DoF handler for the ionic variables problem." << std::endl;

  //   dof_handler_ionic.reinit(mesh);
  //   dof_handler_ionic.distribute_dofs(*fe_ionic);

  //   // locally_owned_dofs_ionic = dof_handler_ionic.locally_owned_dofs();
  //   // DoFTools::extract_locally_relevant_dofs(dof_handler_ionic, locally_relevant_dofs_ionic);

  //   pcout << "  Number of DoFs = " << dof_handler_ionic.n_dofs() << std::endl;
  // }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;
    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

    // pcout << "  Initializing the ionic variables vector" << std::endl;
    // TODO initialize, with the correct dofs from the correct fespace
    // const unsigned int n_q           = quadrature->size();
    // std::vector<double> z_init(dim_ionic);
    // func
    
    // ionicvars_old_loc.assign(n_q, z_init);  // z_n
    // ionicvars_loc(n_q, std::vector<double>(dim_ionic));      // z_n+1
    // ionicvars_owned.reinit(locally_owned_dofs_ionic, MPI_COMM_WORLD);
    // ionicvars.reinit(locally_owned_dofs_ionic, locally_relevant_dofs_ionic, MPI_COMM_WORLD);
  }
}

void
BuenoOrovioModel::assemble_matrices()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix      = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_mass_matrix      = 0.0;
      cell_stiffness_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Evaluate coefficients on this quadrature node.
          const double mu_loc = mu.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                            fe_values.shape_value(j, q) /
                                            deltat * fe_values.JxW(q);    // mass_matrix/deltat!

                  cell_stiffness_matrix(i, j) +=
                    mu_loc * fe_values.shape_grad(i, q) *
                    fe_values.shape_grad(j, q) * fe_values.JxW(q);
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      mass_matrix.add(dof_indices, cell_mass_matrix);
      stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
    }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // We build the matrix on the left-hand side of the algebraic problem: later 
  // (in assemble_lhs) we will add also the J_ion_lin (derivative of Jion that
  // linearly depends on u) contribution.
  // LHS:
  //      (mass_matrix/tau + stifness_matrix) * u_n+1
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix.add(1, stiffness_matrix); // constant 1 needed

  // We build the matrix on the right-hand side (the one that multiplies the old
  // solution u_n).
  // RHS:
  //      mass_matrix/tau * u_n - J_ion(u_n, z_n+1) + J_app(t_n+1)
  rhs_matrix.copy_from(mass_matrix);

}

// This method compute the values of the ionic variables for the current quadrature node, taking as
// input the reference to the gating variables' vector and updates z_new with the computed values.
void 
BuenoOrovioModel::solve_ionic_system(const std::vector<double> &z_old, const double &u, std::vector<double> &z_new) {
   
        double tau_v_min = (1 - H(u, tissue_parameters.theta_v_min))* tissue_parameters.tau_v1_min + H(u, tissue_parameters.theta_v_min)* tissue_parameters.tau_v2_min;
        double tau_w_min =  tissue_parameters.tau_w1_min + (tissue_parameters.tau_w2_min - tissue_parameters.tau_w1_min)*(1 + std::tanh(tissue_parameters.k_w_min *(u - tissue_parameters.u_w_min)))/ 2.;
        double tau_s = (1 - H(u, tissue_parameters.theta_w)) * tissue_parameters.tau_s1 + H(u, tissue_parameters.theta_w)* tissue_parameters.tau_s2;
        double v_inf = 1 - H(u, tissue_parameters.theta_v_min);
        double w_inf = (1 - H(u, tissue_parameters.theta_0))*(1 - (u / tissue_parameters.tau_w_inf))* tissue_parameters.w_inf_star;

        z_new[0] = (z_old[0] * tau_v_min + (1 - H(u, tissue_parameters.theta_v)) * deltat * v_inf) * tissue_parameters.tau_v_plus / (tau_v_min * tissue_parameters.tau_v_plus + deltat * tissue_parameters.tau_v_plus + H(u, tissue_parameters.theta_v) * deltat * (tau_v_min - tissue_parameters.tau_v_plus));
        z_new[1] = (z_old[1] * tau_w_min + (1 - H(u, tissue_parameters.theta_w)) * deltat * w_inf) * tissue_parameters.tau_w_plus / (tau_w_min * tissue_parameters.tau_w_plus + deltat * tissue_parameters.tau_w_plus + H(u, tissue_parameters.theta_w) * deltat * (tau_w_min - tissue_parameters.tau_w_plus));
        z_new[2] = (z_old[2] * tau_s + 1 + std::tanh(tissue_parameters.k_s * (u - tissue_parameters.u_s))) / (2 * (tau_s + 1));
}

void
BuenoOrovioModel::assemble_rhs(const double &time)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  // Since the ionic problem is vectorial and not scalar as the one 
  // for the potential we need different fe_values  
  // FEValues<dim> fe_values_ionic(*fe_ionic,
  //                         *quadrature,
  //                         update_values | update_quadrature_points |
  //                           update_JxW_values);

  // FEEvaluation<dim> fe_evaluation_ionic(*fe_ionic,
  //                         *quadrature,
  //                         update_values | update_quadrature_points |
  //                           update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs = 0.0;

  // Value of the solution (u_n) and of the ionicvars (z_n, z_n+1) on current cell.
  std::vector<double>                      solution_old_loc(n_q);                                   // u_n
  // std::vector<std::vector<double>>         ionicvars_old_loc(n_q, std::vector<double>(dim_ionic));  // z_n
  // std::vector<std::vector<double>>         ionicvars_loc(n_q, std::vector<double>(dim_ionic));      // z_n+1

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);
      // fe_values_ionic.reinit(cell);
      // fe_evaluation_ionic.reinit(cell);

      cell_rhs = 0.0;

      // Get the value of the solution and the ionic variables at the previous timestep,
      // in each quadrature node, for the current cell.
      fe_values.get_function_values(solution, solution_old_loc);
      // fe_values_ionic.get_function_values(ionicvars_old, ionicvars_old_loc); 
      // terrei z e z_old separate, magari aggiornando z sulla cella corrente modifico valori che potrebbero servire a un'altra cella
      // sarà necessario implementare un aggiornamento z_old = z
      
      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Here we compute the non-linear contribution of  
          // of J_ion(u_n, z_n+1) as well as the contribution
          // of the forcing term (applied current).

          // Compute ionic variables (depends on solution_old_loc[q] ->u_n and on ionic_old_loc[q] -> z_n at prev step)
          // The method value takes as input ionicvars_old_loc[q] (z_n) and solution_old_loc[q] (u_n), computing z_n+1
          // and updating ionicvars_loc[q].
          solve_ionic_system(ionicvars_old_loc[q], solution_old_loc[q], ionicvars_loc[q]);
          
          // Compute Jion(u_n) (depends on ionic variables, and on solution_loc[q] -> u_n)
           const double j_ion_loc = j_ion.value(ionicvars_loc[q] /* z_n+1 */, solution_old_loc[q] /* u_n */); 

          // Compute Japp(u_n+1)
            const double j_app_loc = j_app.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_rhs(i) += (-j_ion_loc + j_app_loc) * fe_values.shape_value(i, q) * fe_values.JxW(q); // check!
            }
        }

      cell->get_dof_indices(dof_indices);
      system_rhs.add(dof_indices, cell_rhs);
    }

  system_rhs.compress(VectorOperation::add);

  // Add the term that comes from the old solution.
  rhs_matrix.vmult_add(system_rhs, solution_owned);
}

void
BuenoOrovioModel::solve_time_step()
{
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
    lhs_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(lhs_matrix, solution_owned, system_rhs, preconditioner);
  pcout << "   " << solver_control.last_step() << " GMRES iterations"
        << std::endl;

  solution = solution_owned;
}

void
BuenoOrovioModel::output(const unsigned int &time_step, const double &time) const
{
  // ToDo: output ionic?
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  std::string output_file_name = std::to_string(time_step);

  // Pad with zeros.
  output_file_name = "output-" + std::string(4 - output_file_name.size(), '0') +
                     output_file_name;

  DataOutBase::DataOutFilter data_filter(
    DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices = */ false,
                                    /*xdmf_hdf5_output = */ true));
  data_out.write_filtered_data(data_filter);
  data_out.write_hdf5_parallel(data_filter,
                               output_file_name + ".h5",
                               MPI_COMM_WORLD);

  std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(
    data_filter, output_file_name + ".h5", time, MPI_COMM_WORLD)});
  data_out.write_xdmf_file(xdmf_entries,
                           output_file_name + ".xdmf",
                           MPI_COMM_WORLD);
}

void
BuenoOrovioModel::solve()
{
  assemble_matrices();

  pcout << "===============================================" << std::endl;

  time = 0.0;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition for diffusion problem" << std::endl;

    VectorTools::interpolate(dof_handler, u0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0, 0.0);
    pcout << "-----------------------------------------------" << std::endl;

    pcout << "Applying the initial condition for ionic problem" << std::endl;

    // VectorTools::interpolate(dof_handler_ionic, z0, ionicvars_owned);
    // ionicvars = ionicvars_owned;

    const unsigned int n_q = quadrature->size();
    std::vector<double> z0(dim_ionic);
    functionz0.vector_value(z0);
    
    ionicvars_old_loc.assign(n_q, z0);    // z_n
    ionicvars_loc.assign(n_q,z0);           // z_n+1

    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat)
    {
      time += deltat;
      ++time_step;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      assemble_rhs(time);
      solve_time_step();
      output(time_step, time);

      // Store the value of the ionic variables, it will be needed at the next timestep.
      ionicvars_old_loc.swap(ionicvars_loc);
    }
}