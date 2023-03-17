#ifndef CARDIAC_ELECTROPHYSIOLOGY_HPP
#define CARDIAC_ELECTROPHYSIOLOGY_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Ma fare delle classi friends con i metodi relativi ai due problemi??

// Class representing the cardiac electrophysiology problem.
class CardiacElectrophysiology
{
public:
  // Physical dimension of the diffusion problem (PDE)(3D).
  static constexpr unsigned int dim = 3;

  // Physical dimension of the gating problem (ODE)(3D). (v, w, s)
  static constexpr unsigned int dim_gating = 4;

  // Function for the D coefficient (diffusion coefficient).
  class DiffusionCoefficient : public Function<dim>
  {
  public:
    // define value method
  };

  // Function for the ionic current.
  class IonicCurrent : public Function<dim>
  {
  public:
    // define value method
  };

  // Function for the forcing term (applied current).
  class AppliedCurrent : public Function<dim>
  {
  public:
    // define value method
  };

  // Neumann boundary conditions.
  class FunctionH : public Function<dim>
  {
  public:
    // Constructor.
    FunctionH()
    {}

    // Evaluation:
    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const
    {
      return 0.;
    }
  };

  // Function dv/dt.
  class DerivativeV : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      double heaviside = ((p[0] - theta) > 0);
      return (1 - heaviside) * (v_inf - p[1]) / tau_minus -
             heaviside * p[1] / tau_plus;
    }

  protected:
    const double theta     = 0.;
    const double v_inf     = 0.;
    const double tau_minus = 0.;
    const double tau_plus  = 0.;
  };

  // Function dw/dt.
  class DerivativeW : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      double heaviside = ((p[0] - theta) > 0);
      return (1 - heaviside) * (w_inf - p[2]) / tau_minus -
             heaviside * p[2] / tau_plus;
    }

  protected:
    const double theta     = 0.;
    const double w_inf     = 0.;
    const double tau_minus = 0.;
    const double tau_plus  = 0.;
  };

  // Function ds/dt.
  class DerivativeS : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return ((1 + std::tanh(k_s * (p[0] - u_s))) / 2 - p[3]) / tau_s;
    }

  protected:
    const double k_s   = 0.;
    const double u_s   = 0.;
    const double tau_s = 0.;
  };

  // Constructor.
  CardiacElectrophysiology(const unsigned int &N_, const unsigned int &r_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , N(N_)
    , r(r_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Assemble the diffusion and the gating problems.
  void
  assemble_systems();

  // Solve the tangent problem.
  void
  solve_diffusion_system();

  // Solve the gating problem.
  void
  solve_gating_system();

  // Output.
  void
  output() const;

protected:
  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Diffusion problem definition. ///////////////////////////////////////////////////////

  // Diffusion coefficient.
  DiffusionCoefficient D;

  // Ionic current.
  IonicCurrent I_ion;

  // Applied current.
  AppliedCurrent I_app;

  // Neumann boundary condition.
  FunctionH function_h;

  // Gating problem definition. /////////////////////////////////////////////////
  
  // Derivative of v wrt time.
  DerivativeV derivative_v;

  // Derivative of w wrt time.
  DerivativeW derivative_w;

  // Derivative of s wrt time.
  DerivativeS derivative_s;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh refinement.
  const unsigned int N;

  // Polynomial degree.
  const unsigned int r;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Jacobian matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // Residual vector.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;
};