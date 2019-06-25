#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/conditional_ostream.h>



#ifndef LETHE_INITIALCONDITIONS_H
#define LETHE_INITIALCONDITIONS_H

using namespace dealii;


namespace  Parameters
{

// Type of initial conditions
enum InitialConditionType {none, L2projection, viscous, nodal};

template <int dim>
class InitialConditions
{
public:
  InitialConditions():uvwp(dim+1){}

  InitialConditionType type;

  // Velocity components
  Functions::ParsedFunction<dim>  uvwp;

  // Artificial viscosity
  double viscosity;

  void declare_parameters (ParameterHandler &prm);
  void parse_parameters (ParameterHandler &prm);
};




template <int dim>
void InitialConditions<dim>::declare_parameters (ParameterHandler &prm)
{
  prm.enter_subsection("initial conditions");
  {
    prm.declare_entry("type", "none",
                      Patterns::Selection("none|L2projection|viscous|nodal"),
                      "Type of initial condition"
                      "Choices are <none|L2projection|viscous|nodal>.");
    prm.enter_subsection("uvwp");
    uvwp.declare_parameters(prm,dim);
    if (dim==2) prm.set("Function expression","0; 0; 0");
    if (dim==3) prm.set("Function expression","0; 0; 0; 0");
    prm.leave_subsection();

    prm.declare_entry("viscosity", "1",Patterns::Double(),"viscosity for viscous initial conditions");
  }
  prm.leave_subsection();
}

template <int dim>
void InitialConditions<dim>::parse_parameters (ParameterHandler &prm)
{
  prm.enter_subsection("initial conditions");
  {
    const std::string op = prm.get("type");
    if (op == "none")
      type = none;
    if (op == "L2projection")
      type = L2projection;
    if (op == "viscous")
      type = viscous;
    if (op== "nodal")
      type = nodal;

    viscosity = prm.get_double("viscosity");
    prm.enter_subsection("uvwp");
    uvwp.parse_parameters(prm);
    prm.leave_subsection();
  }
  prm.leave_subsection();
}
}

#endif