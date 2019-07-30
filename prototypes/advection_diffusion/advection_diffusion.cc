/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */


// @sect3{Include files}

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/function.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <fstream>
#include <iostream>

#include <deal.II/grid/manifold_lib.h>

#include <deal.II/base/logstream.h>


// The final step, as in previous programs, is to import all the deal.II class
// and function names into the global namespace:
using namespace dealii;
enum simCase {original, TaylorCouette, MMS, PECLET};

template <int dim>
class advectionDiffusionSolver
{
public:
  advectionDiffusionSolver ();
  void run ();
  double getL2Error(){return L2Error_;}
  void setRefinementLevel(unsigned int refinementLevel);

private:
  void make_grid ();
  void make_cube_grid (unsigned int level);
  void make_ring_grid();
  void refine_mesh_uniform();
  void setup_dofs();
  void initialize_system();
  void assemble_system ();
  void solve ();
  void output_results (std::string simulation_case, unsigned int refinement_level) const;
  void calculateL2Error();
  void readParameters();

  Triangulation<dim>   triangulation;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;
  Point<dim>           center;

  simCase              simulationCase_;
  unsigned int         refinementLevel_=1;
  double               L2Error_;

  Function<dim>        *exact_solution_;
  Function<dim>        *source_term_;
  Function<dim>        *velocity_;

};





template <int dim>
class SourceZero : public Function<dim>
{
public:
  SourceZero () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
double SourceZero<dim>::value (const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
  return 0.;
}

template <int dim>
class SourceMMS : public Function<dim>
{
public:
  SourceMMS () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
double SourceMMS<dim>::value (const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
  double return_value = 0.0;
  double x = p(0);
  double y = p(1);

  return_value = -2.*M_PI*M_PI * std::sin(M_PI*x)*std::sin(M_PI*y);

  return return_value;
}

template <int dim>
class CircleBoundaryValues : public Function<dim>
{
public:
  CircleBoundaryValues () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};




template <int dim>
double CircleBoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int /*component*/) const
{
  return p.square();
}


template <int dim>
class VelocityFieldNone : public Function<dim>
{
public:
  VelocityFieldNone () : Function<dim>() {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const;
};




template<int dim>
void VelocityFieldNone<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> &values) const
{

    values(0) = 0.;
    values(1) = 0.;


}



template <int dim>
advectionDiffusionSolver<dim>::advectionDiffusionSolver ()
  :
  fe (1),
  dof_handler (triangulation)
{
}

template <int dim>
void advectionDiffusionSolver<dim>::make_grid ()
{
 if (simulationCase_==TaylorCouette) make_ring_grid();
}

template <int dim>
void advectionDiffusionSolver<dim>::make_cube_grid (unsigned int mesh_level)
{
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (mesh_level);

  std::cout << "   Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: "
            << triangulation.n_cells()
            << std::endl;
}

// Ring mesh that is globally refined
template <int dim>
void advectionDiffusionSolver<dim>::make_ring_grid ()
{
  const double inner_radius = 0.25,
               outer_radius = 1.0;
  if (dim==2) center = Point<dim>(0,0);
  GridGenerator::hyper_shell (triangulation,
                              center, inner_radius, outer_radius,
                              4, true);

  static const SphericalManifold<dim> manifold_description(center);
  triangulation.set_manifold (0, manifold_description);
  triangulation.set_all_manifold_ids(0);
  triangulation.refine_global(refinementLevel_);

  std::cout << "Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl;

  std::cout << "Number of total cells: "
            << triangulation.n_cells()
            << std::endl;
}


template <int dim>
void advectionDiffusionSolver<dim>::setup_dofs ()
{
  dof_handler.distribute_dofs (fe);

  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;


}


template <int dim>
void advectionDiffusionSolver<dim>::initialize_system()
{
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}


template <int dim>
void advectionDiffusionSolver<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(5);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);
  Vector<double>       velocity_vec(dim);
  Tensor<1,dim>        velocity;

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      cell_matrix = 0;
      cell_rhs = 0;

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
      {
        velocity_->vector_value(fe_values.quadrature_point(q_index),velocity_vec);
        // Establish the force vector
        for( int i=0; i<dim; ++i )
        {
            const unsigned int component_i = fe.system_to_component_index(i).first;
            velocity[i] = velocity_vec(component_i);
        }
        for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
          {
            //Stiffness Matrix
            cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                                 fe_values.shape_grad (j, q_index) *
                                 fe_values.JxW (q_index));
            //Advection Matrix
            cell_matrix(i,j) += (fe_values.shape_grad(j,q_index) * velocity * fe_values.shape_value(i,q_index)*
                                 fe_values.JxW (q_index));
          }

          if (source_term_)
          {
            // Right Hand Side
            cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                            source_term_->value (fe_values.quadrature_point (q_index)) *
                            fe_values.JxW (q_index));
          }

        }

      // Assemble global matrix
      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
      }
    }


  std::map<types::global_dof_index,double> boundary_values;
  if (simulationCase_==original)
  {
      VectorTools::interpolate_boundary_values (dof_handler,
                                                0,
                                                CircleBoundaryValues<dim>(),
                                                boundary_values);
  }
  if (simulationCase_==TaylorCouette)
  {

      VectorTools::interpolate_boundary_values (dof_handler,
                                                0,
                                                Functions::ZeroFunction<dim>(),
                                                boundary_values);
      VectorTools::interpolate_boundary_values (dof_handler,
                                                1,
                                                 Functions::ConstantFunction<dim>(1.),
                                                boundary_values);
  }

  if (simulationCase_==MMS)
  {
      VectorTools::interpolate_boundary_values (dof_handler,
                                                0,
                                                Functions::ZeroFunction<dim>(),
                                                boundary_values);
  }

  if (simulationCase_== PECLET)
  {
      VectorTools::interpolate_boundary_values (dof_handler,
                                                1,
                                                Functions::ZeroFunction<dim>(),
                                                boundary_values);
      VectorTools::interpolate_boundary_values (dof_handler,
                                                0,
                                                Functions::ConstantFunction<dim>(1.),
                                                boundary_values);
  }

  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}


template <int dim>
void advectionDiffusionSolver<dim>::solve ()
{
  SolverControl           solver_control (100000, 1e-10);
  SolverGMRES<>           solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());

  std::cout << "   " << solver_control.last_step()
            << " Iterations needed to obtain convergence."
            << std::endl;
}


template <int dim>
void advectionDiffusionSolver<dim>::refine_mesh_uniform ()
{
    SolutionTransfer<dim, Vector<double> > solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(solution);
    triangulation.refine_global(1);
    setup_dofs();
    Vector<double> tmp(dof_handler.n_dofs());
    solution_transfer.interpolate(solution, tmp);
    initialize_system();
    solution = tmp;
}



template <int dim>
void advectionDiffusionSolver<dim>::output_results (std::string simulation_case, unsigned int refinement_level) const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");

  data_out.build_patches ();

  std::string fname= simulation_case.c_str()+std::string(".")+Utilities::int_to_string(refinement_level)+".vtk";

  std::ofstream output (fname.c_str());
  data_out.write_vtk (output);
}

//Find the l2 norm of the error between the finite element sol'n and the exact sol'n
template <int dim>
void advectionDiffusionSolver<dim>::calculateL2Error()
{

  QGauss<dim>  quadrature_formula(5);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);

  const unsigned int   			dofs_per_cell = fe.dofs_per_cell;         // This gives you dofs per cell
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell); //  Local connectivity

  const unsigned int   n_q_points    = quadrature_formula.size();

  double l2error=0.;

  //loop over elements
  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);

    //Retrieve the effective "connectivity matrix" for this element
    cell->get_dof_indices (local_dof_indices);


    for(unsigned int q=0; q<n_q_points; q++)
    {
      const double x = fe_values.quadrature_point(q)[0];
      const double y = fe_values.quadrature_point(q)[1];
//      if (dim>2) const double z= fe_values.quadrature_point(q)[2];

      double u_exact=0.;
      if (simulationCase_==MMS)
         u_exact = -sin(M_PI * x) * std::sin(M_PI*y);

      double u_sim=0;

      //Find the values of x and u_h (the finite element solution) at the quadrature points
      for(unsigned int i=0; i<dofs_per_cell; i++)
      {
        u_sim += fe_values.shape_value(i,q) * solution[local_dof_indices[i]];
      }
      l2error += (u_sim-u_exact)*(u_sim-u_exact) * fe_values.JxW(q);
    }
  }
  std::cout << "L2Error is : " << std::sqrt(l2error) << std::endl;
  L2Error_=std::sqrt(l2error);

}



template <int dim>
void advectionDiffusionSolver<dim>::run ()
{
  simulationCase_=MMS;
  source_term_ = new  SourceMMS<2>();
  velocity_ = new VelocityFieldNone<2>();
  make_cube_grid(3);
  setup_dofs();
  initialize_system();
  for (unsigned int i = 0 ; i < 3 ; ++i)
  {
    if (i!=0 )refine_mesh_uniform();
    assemble_system ();
    solve ();
    output_results ("MMS",0);
    calculateL2Error();
  }
}

int main ()
{
  deallog.depth_console (0);
  advectionDiffusionSolver<2> problem;
  problem.run();

  return 0;
}
