#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <string>
#include <cmath>

using namespace dealii;

#define H(x, c) (x >= c)

// Class representing the Bueno-Orovio model of the heart
class BuenoOrovioModel
{
public:
    // Physical dimension of diffusion problem (3D)
    static constexpr unsigned int dim = 3;

    // Physical dimension of the ionic problem (ODE)(3D). (v, w, s)
    static constexpr unsigned int dim_ionic = 3;

    // Diffusion coefficient.
    static constexpr double D = 1.171; //+-0.221

    // Functions. ///////////////////////////////////////////////////////////////

    // Function for the mu coefficient.
    class FunctionMu : public Function<dim>
    {
    public:
        virtual double
        value(const Point<dim> & /*p*/,
              const unsigned int /*component*/ = 0) const override
        {
            return BuenoOrovioModel::D;
        }
    };

    // class Heaviside : public Function<dim>
    // {
    // public:
    //     virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    //     {
    //         u(p);
    //         return std::sin(5 * M_PI * get_time()) * std::sin(2 * M_PI * p[0]) *
    //                std::sin(3 * M_PI * p[0]) * std::sin(4 * M_PI * p[2]);
    //     }
    // };

    // Function for the forcing term (applied current).
    class ForcingTerm : public Function<dim>
    {
    public:
        virtual double
        value(const Point<dim> &p,
              const unsigned int /*component*/ = 0) const override
        {
            return (p==source && get_time()<threshold)*val;
        }
        
    private:
        // TODO tune values here
        double threshold = 1e-4; 
        double val = 1e-2;
        Point<dim> source{0,0,0};
    };

    // Function for the initial condition of u.
    class FunctionU0 : public Function<dim>
    {
    public:
        virtual double
        value(const Point<dim> &/*p*/,
              const unsigned int /*component*/ = 0) const override
        {
            return 0;
        }
    };

    // Vector valued function for initial condition of ionic variables.
    class FunctionZ0
    {
    public:
        void
        vector_value(std::vector<double> &values)
        {
        values[0] = 1.0; // v
        values[1] = 1.0; // w
        values[2] = 0.0; // s
        }
    };

    // // Vector valued function corresponding to the gating variable system.
    // class IonicSystem
    // {
    // public:
    //     IonicSystem(double u_) {
    //         this.u = u_;
    //     }

    //     // This method compute the values of the ionic variables for the current quadrature node, taking as
    //     // input the reference to the gating variables' vector and updates z_new with the computed values.
    //     void
    //     value(const std::vector<double> &z_old, std::vector<double> &z_new) const
    //     {
    //         z_new[0] = (z_old[0] * tau_v_min + (1 - H(u, tissue_parameters.theta_v)) * deltat * v_inf) 
    //                     * tissue_parameters.tau_v_plus 
    //                     / (tau_v_min * tissue_parameters.tau_v_plus + deltat * tissue_parameters.tau_v_plus + H(u, tissue_parameters.theta_v) * deltat * (tau_v_min - tissue_parameters.tau_v_plus));
    //         z_new[1] = (z_old[1] * tau_w_min + (1 - H(u, tissue_parameters.theta_w)) * deltat * w_inf) 
    //                     * tau_w_plus 
    //                     / (tau_w_min * tissue_parameters.tau_w_plus + deltat * tissue_parameters.tau_w_plus + H(u, tissue_parameters.theta_w) * deltat * (tau_w_min - tissue_parameters.tau_w_plus));
    //         z_new[2] = (z_old[2] * tau_s + 1 + std::tanh(tissue_parameters.k_s * (u - tissue_parameters.u_s))) 
    //                     / (2 * (tau_s + 1))
    //     }

    // protected:
    //     // Here some constant for computing the system.
    //     // maybe the H??

    //     constexpr double tau_v_min = (1 - H(u, tissue_parameters.theta_v_min))* tissue_parameters.tau_v1_min + H(u, tissue_parameters.theta_v_min)* tissue_parameters.tau_v2_min;
    //     constexpr double tau_w_min =  tissue_parameters.tau_w1_min + (tissue_parameters.tau_w2_min - tissue_parameters.tau_w1_min)*(1 + std::tanh(tissue_parameters.k_w_min *(u - tissue_parameters.u_w_min)))/ 2.;
    //     constexpr double tau_s = (1 - H(u, tissue_parameters.theta_w)) * tissue_parameters.tau_s1 + H(u, tissue_parameters.theta_w)* tissue_parameters.tau_s2;
    //     constexpr double v_inf = 1 - H(u, tissue_parameters.theta_v_min);
    //     constexpr double w_inf = (1 - H(u, tissue_parameters.theta_o))*(1 - (u / tissue_parameters.tau_w_inf))* tissue_parameters.w_inf_star;

    // private:
    //     double u;
    // };

    class FunctionJion
    {
    public:
        FunctionJion(BuenoOrovioModel &x):m(x){}
        double value(const std::vector<double> &z, const double &u_old) const
        {

            double tau_so =  m.tissue_parameters.tau_so1 + (m.tissue_parameters.tau_so2 - m.tissue_parameters.tau_so1)*(1 + std::tanh(m.tissue_parameters.k_so *(u_old - m.tissue_parameters.u_so)))/ 2.;
            double tau_o = (1 - H(u_old, m.tissue_parameters.theta_0)) * m.tissue_parameters.tau_o1 + H(u_old, m.tissue_parameters.theta_0)* m.tissue_parameters.tau_o2;

            // expression of J_ion
            double J_fi = -z[0]*H(u_old, m.tissue_parameters.theta_v)*(u_old - m.tissue_parameters.theta_v)*(m.tissue_parameters.u_u - u_old)/m.tissue_parameters.tau_fi ;
            double J_so = ((u_old - m.tissue_parameters.u_0)*(1 - H(u_old, m.tissue_parameters.theta_w)/tau_o))+((H(u_old,m.tissue_parameters.theta_w)/tau_so)); // manca
            double J_si = (H(u_old, m.tissue_parameters.theta_w)*z[1]*z[2])/m.tissue_parameters.tau_si;

            return J_fi + J_so + J_si;
        }
    private:
        BuenoOrovioModel &m;
    };

    // Constructor. We provide the final time, time step Delta t and theta method
    // parameter as constructor arguments.
    BuenoOrovioModel(const unsigned int &N_,
                const unsigned int &r_,
                const double &T_,
                const double &deltat_,
                const std::string &tissue_type_)
        : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank == 0)
        , T(T_)
        , N(N_)
        , r(r_)
        , deltat(deltat_)
        , tissue_parameters(tissue_type_)
        , mesh(MPI_COMM_WORLD)
        , j_ion(*this)
    {}

    // Initialization.
    void setup();

    // Solve the problem.
    void solve();

protected:
    // Model's parameters: //////////////
    class TissueParameters
    {
    public:
        TissueParameters(const std::string &tissue_type_){
            if (!tissue_type_.compare("epicardium"))
            {
                u_0 = 0.0;
                u_u = 1.55;
                theta_v = 0.3;
                theta_w = 0.13;
                theta_v_min = 0.006;
                theta_0 = 0.006;
                tau_v1_min = 60e-3;
                tau_v2_min = 1150e-3;
                tau_v_plus = 1.4506e-3;
                tau_w1_min = 60e-3;
                tau_w2_min = 15e-3;
                k_w_min = 65;
                u_w_min = 0.03;
                tau_w_plus = 200e-3;
                tau_fi = 0.11e-3;
                tau_o1 = 400e-3;
                tau_o2 = 6e-3;
                tau_so1 = 30.0181e-3;
                tau_so2 = 0.9957e-3;
                k_so = 2.0458;
                u_so = 0.65;
                tau_s1 = 2.7342e-3;
                tau_s2 = 16e-3;
                k_s = 2.0994;
                u_s = 0.9087;
                tau_si = 1.8875e-3;
                tau_w_inf = 0.07;
                w_inf_star = 0.9;
            }
            else if(!tissue_type_.compare("endocardium"))
            {
                u_0 = 0.0;
                u_u = 1.56;
                theta_v = 0.3;
                theta_w = 0.13;
                theta_v_min = 0.2;
                theta_0 = 0.006;
                tau_v1_min = 75e-3;
                tau_v2_min = 10e-3;
                tau_v_plus = 1.4506e-3;
                tau_w1_min = 6e-3;
                tau_w2_min = 140e-3;
                k_w_min = 200;
                u_w_min = 0.0016;
                tau_w_plus = 280e-3;
                tau_fi = 0.1e-3;
                tau_o1 = 470e-3;
                tau_o2 = 6e-3;
                tau_so1 = 40e-3;
                tau_so2 = 1.2e-3;
                k_so = 2;
                u_so = 0.65;
                tau_s1 = 2.7342e-3;
                tau_s2 = 2e-3;
                k_s = 2.0994;
                u_s = 0.9087;
                tau_si = 2.9013e-3;
                tau_w_inf = 0.0273;
                w_inf_star = 0.7;
            }
            else if(!tissue_type_.compare("myocardium")){
                u_0 = 0.0;
                u_u = 1.61;
                theta_v = 0.3;
                theta_w = 0.13;
                theta_v_min = 0.1;
                theta_0 = 0.005;
                tau_v1_min = 80e-3;
                tau_v2_min = 1.4506e-3;
                tau_v_plus = 1.4506e-3;
                tau_w1_min = 70e-3;
                tau_w2_min = 8e-3;
                k_w_min = 200;
                u_w_min = 0.0016;
                tau_w_plus = 280e-3;
                tau_fi = 0.078e-3;
                tau_o1 = 410e-3;
                tau_o2 = 7e-3;
                tau_so1 = 91e-3;
                tau_so2 = 0.8e-3;
                k_so = 2.1;
                u_so = 0.6;
                tau_s1 = 2.7342e-3;
                tau_s2 = 4e-3;
                k_s = 2.0994;
                u_s = 0.9087;
                tau_si = 3.3849e-3;
                tau_w_inf = 0.01;
                w_inf_star = 0.;
            }
            else {
                throw std::runtime_error("Unknown tissue type: " + tissue_type_);
            }
        }
        double u_0;
        double u_u;
        double theta_v;
        double theta_w;
        double theta_v_min;
        double theta_0;
        double tau_v1_min;
        double tau_v2_min;
        double tau_v_plus;
        double tau_w1_min;
        double tau_w2_min;
        double k_w_min;
        double u_w_min;
        double tau_w_plus;
        double tau_fi;
        double tau_o1;
        double tau_o2;
        double tau_so1;
        double tau_so2;
        double k_so;
        double u_so;
        double tau_s1;
        double tau_s2;
        double k_s;
        double u_s;
        double tau_si;
        double tau_w_inf;
        double w_inf_star;
    };

    // FE methods: //////////////////////

    // Assemble the mass and stiffness matrices.
    void assemble_matrices();

    void solve_ionic_system(const std::vector<double> &z_old, const double &u, std::vector<double> &z_new);

    // Assemble the right-hand side of the problem.
    void assemble_rhs(const double &time);

    // Solve the problem for one time step.
    void solve_time_step();

    // Output.
    void output(const unsigned int &time_step, const double &time) const;

    // MPI parallel. /////////////////////////////////////////////////////////////

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Parallel output stream.
    ConditionalOStream pcout;

    // Problem definition. ///////////////////////////////////////////////////////

    // mu coefficient.
    FunctionMu mu;

    // Forcing term.
    ForcingTerm j_app;

    // Initial conditions.
    FunctionU0 u0;
    FunctionZ0 functionz0;

    // Current time.
    double time;

    // Final time.
    const double T;

    // Discretization. ///////////////////////////////////////////////////////////

    // Mesh refinement.
    const unsigned int N;

    // Polynomial degree.
    const unsigned int r;

    // Time step.
    const double deltat;
    
    // Tissue parameters.
    const TissueParameters tissue_parameters;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite element space for potential problem.
    std::unique_ptr<FiniteElement<dim>> fe;

    // Finite element space for ionic variables problem.
    std::unique_ptr<FiniteElement<dim>> fe_ionic;

    // Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // DoF handler for ionic variables.
    DoFHandler<dim> dof_handler_ionic;

    // DoFs owned by current process.
    IndexSet locally_owned_dofs;

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs;

    // Mass matrix M / deltat.
    TrilinosWrappers::SparseMatrix mass_matrix;

    // Stiffness matrix A.
    TrilinosWrappers::SparseMatrix stiffness_matrix;

    // Matrix on the left-hand side (M / deltat + theta A).
    TrilinosWrappers::SparseMatrix lhs_matrix;

    // Matrix on the right-hand side (M / deltat - (1 - theta) A).
    TrilinosWrappers::SparseMatrix rhs_matrix;

    // Right-hand side vector in the linear system.
    TrilinosWrappers::MPI::Vector system_rhs;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::Vector solution_owned;

    // System solution (including ghost elements).
    TrilinosWrappers::MPI::Vector solution;

    // Ionic variables vectors at the previous timestep and at the current one.
    std::vector<std::vector<double>>         ionicvars_old_loc;  // z_n
    std::vector<std::vector<double>>         ionicvars_loc;      // z_n+1

    // Ionic current.
    FunctionJion j_ion;
};