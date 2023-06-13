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
    const std::string mesh_file_name =
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
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

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

    pcout << "  Initializing the gating variables vector" << std::endl;
    gating_vector = Vector<double> (dim_gating); // do we need this? init the vector with the correct dimensions....
    // Or setting it equal to initial conditions (see solve method) is enough?
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

void
BuenoOrovioModel::assemble_lhs(const double &time) // NOT NEEDED
{
    const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_lhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Here we compute the contribution of the derivative of J_ion 
          // (the one that linearly depends on u) at the previous timestep t_n.

          // Compute J(u_n)
        //   forcing_term.set_time(time - deltat);
        //   const double f_old_loc =
        //     forcing_term.value(fe_values.quadrature_point(q));

        //   for (unsigned int i = 0; i < dofs_per_cell; ++i)
        //     {
        //       cell_rhs(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) *
        //                      fe_values.shape_value(i, q) * fe_values.JxW(q);
        //     }
        }

      cell->get_dof_indices(dof_indices);
      system_lhs.add(dof_indices, cell_rhs);

    }

  system_lhs.compress(VectorOperation::add);

  // Add the term that comes from the old solution.
  // rhs_matrix.vmult_add(system_rhs, solution_owned);
  system_lhs.add(1, rhs_matrix);
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

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs = 0.0;

  // From lab06, HeatNonLinear
  // Value of the solution (u_n) and of the ionics (z_n, z_n+1) on current cell.
  std::vector<double>         solution_old_loc(n_q);  // u_n
  std::vector<double>         ionic_old_loc(n_q);     // z_n
  std::vector<double>         ionic_loc(n_q);         // z_n+1

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_rhs = 0.0;

      // From lab06, HeatNonLinear
      fe_values.get_function_values(solution, solution_old_loc);
      // same for z?

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Here we compute the non-linear contribution of  
          // of J_ion(u_n, z_n+1) as well as the contribution
          // of the forcing term (applied current).

          // see LAB06 for solution_loc (HeatNonLinear)
          // Compute ionic variables (depends on solution_old_loc[q] ->u_n and on ionic_old_loc[q] -> z_n at prev step)

          // Compute Jion(u_n) (depends on ionic variables, and on solution_loc[q] -> u_n)

          // Compute Japp(u_n+1)
        //   forcing_term.set_time(time);
        //   const double f_loc =
        //     forcing_term.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
            //   cell_rhs(i) += (-J_ion + J_app) * fe_values.shape_value(i, q) * fe_values.JxW(q); (???)
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

  solver.solve(system_lhs /* forse qui ci va matrix_lhs*/, solution_owned, system_rhs, preconditioner);
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

void BuenoOrovioModel::solve_gating_system() // NOT NEEDED
{
  // We create a custom vectorial function in the hpp file (FunctionGatingSys -> gating_sys)
  // and we only call its method solve.
  // In this case we can avoid creating the dedicated method solve_gating_system()...
  gating_system.solve(gating_vector)

  // If we want a (simpler(?)) alternative:
  // Compute v.
  gating_vector[0] = /* equation for v, depends on the old value gating_vector[0] */;

  // Compute w.
  gating_vector[1] = /* equation for w, depends on the old value gating_vector[1] */;

  // Compute s.
  gating_vector[2] = /* equation for s, depends on the old value gating_vector[2] */;

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

    pcout << "Applying the initial condition for gating problem" << std::endl;
    gating_vector = z0;

    // Output the initial solution.
    output(0, 0.0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat)
    {
      time += deltat;
      ++time_step;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      // solve_gating_system();

      assemble_rhs(time);
      // assemble_lhs(time);
      solve_time_step();
      output(time_step, time);
    }
}