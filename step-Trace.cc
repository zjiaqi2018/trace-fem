/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 - 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */

// @sect3{Include files}

// The first include files have all been treated in previous examples.
#define USE_TRILINOS

#include <deal.II/base/function.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/base/timer.h>


#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_ilu.h>

#ifdef USE_TRILINOS
#  include <deal.II/lac/block_sparsity_pattern.h>
#  include <deal.II/lac/sparsity_tools.h>
#  include <deal.II/lac/trilinos_block_sparse_matrix.h>
#  include <deal.II/lac/trilinos_parallel_block_vector.h>
#  include <deal.II/lac/trilinos_precondition.h>
#  include <deal.II/lac/trilinos_solver.h>
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/base/utilities.h>
#endif

#include <deal.II/lac/sparse_direct.h>


#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <vector>

// The first new header contains some common level set functions.
// For example, the spherical geometry that we use here.
#include <deal.II/base/function_signed_distance.h>

// We also need 3 new headers from the NonMatching namespace.
#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

// @sect3{The LaplaceSolver class Template}
// We then define the main class that solves the Laplace problem.

namespace StepTrace
{
  using namespace dealii;

  template <int dim>
  class LaplaceBeltramiSolver
  {
  public:
    LaplaceBeltramiSolver();

    void run();

  private:
#ifdef USE_TRILINOS
  using VectorType = TrilinosWrappers::MPI::Vector;
  using MatrixType = TrilinosWrappers::SparseMatrix;
#else 
    using VectorType = Vector<double>;
  using MatrixType = SparseMatrix<double> ;  
#endif
  
    void make_grid();

    void setup_discrete_level_set();

    void distribute_dofs();

    void initialize_matrices();

    void assemble_system();

    void solve();

    void output_results() const;

    double compute_L2_error() const;

    bool face_has_ghost_penalty(
      const typename Triangulation<dim>::active_cell_iterator &cell,
      const unsigned int face_index) const;

    const unsigned int fe_degree;

    const Functions::ConstantFunction<dim> rhs_function;
    const Functions::ConstantFunction<dim> boundary_condition;

#ifdef USE_TRILINOS
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif

    // We need two separate DoFHandlers. The first manages the DoFs for the
    // discrete level set function that describes the geometry of the domain.
    const FE_Q<dim> fe_level_set;
    DoFHandler<dim> level_set_dof_handler;
    VectorType  level_set;

    // The second DoFHandler manages the DoFs for the solution of the Poisson
    // equation.
    hp::FECollection<dim> fe_collection;
    DoFHandler<dim>       dof_handler;
    VectorType        solution;

    NonMatching::MeshClassifier<dim> mesh_classifier;

    SparsityPattern      sparsity_pattern;
    MatrixType stiffness_matrix;
    VectorType       rhs;

    unsigned int number_of_iterations;
  };

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
  return 3*p(0);
}


template <int dim>
LaplaceBeltramiSolver<dim>::LaplaceBeltramiSolver()
  : fe_degree(1)
  , rhs_function(4.0)
  , boundary_condition(1.0)
#ifdef USE_TRILINOS
  , triangulation(MPI_COMM_WORLD,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening),
                  parallel::distributed::Triangulation<
                    dim>::mesh_reconstruction_after_repartitioning)
#endif
  , fe_level_set(fe_degree)
  , level_set_dof_handler(triangulation)
  , dof_handler(triangulation)
  , mesh_classifier(level_set_dof_handler, level_set)
{}



// @sect3{Setting up the Background Mesh}
// We generate a background mesh with perfectly Cartesian cells. Our domain is
// a unit disc centered at the origin, so we need to make the background mesh
// a bit larger than $[-1, 1]^{\text{dim}}$ to completely cover $\Omega$.
template <int dim>
void
LaplaceBeltramiSolver<dim>::make_grid()
{
  std::cout << "Creating background mesh" << std::endl;

  GridGenerator::hyper_cube(triangulation, -1.21, 1.21);
  triangulation.refine_global(2);
  }



  // @sect3{Setting up the Discrete Level Set Function}
  // The discrete level set function is defined on the whole background mesh.
  // Thus, to set up the DoFHandler for the level set function, we distribute
  // DoFs over all elements in $\mathcal{T}_h$. We then set up the discrete
  // level set function by interpolating onto this finite element space.
  template <int dim>
  void LaplaceBeltramiSolver<dim>::setup_discrete_level_set()
  {
    std::cout << "Setting up discrete level set function" << std::endl;

    level_set_dof_handler.distribute_dofs(fe_level_set);

    std::cout<<" level set number of dofs: "<<level_set_dof_handler.n_dofs()<<std::endl;
    const Functions::SignedDistance::Sphere<dim> signed_distance_sphere;    

#ifdef USE_TRILINOS
    IndexSet level_set_locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(level_set_dof_handler, level_set_locally_relevant_dofs);

    VectorType tmp_sol;
    tmp_sol.reinit(level_set_dof_handler.locally_owned_dofs(),
                  MPI_COMM_WORLD);

    VectorTools::interpolate(level_set_dof_handler,
                             signed_distance_sphere,
                             tmp_sol);

    level_set.reinit(level_set_locally_relevant_dofs,
                     level_set_dof_handler.locally_owned_dofs(),
                     MPI_COMM_WORLD);
    level_set = tmp_sol;

#else
    level_set.reinit(level_set_dof_handler.n_dofs());
    VectorTools::interpolate(level_set_dof_handler,
                             signed_distance_sphere,
                             level_set);
#endif
  }



  // @sect3{Setting up the Finite Element Space}
  // To set up the finite element space $V_\Omega^h$, we will use 2 different
  // elements: FE_Q and FE_Nothing. For better readability we define an enum for
  // the indices in the order we store them in the hp::FECollection.
  enum ActiveFEIndex
  {
    lagrange = 0,
    nothing  = 1
  };

  // We then use the MeshClassifier to check LocationToLevelSet for each cell in
  // the mesh and tell the DoFHandler to use FE_Q on elements that are inside or
  // intersected, and FE_Nothing on the elements that are outside.
  template <int dim>
  void LaplaceBeltramiSolver<dim>::distribute_dofs()
  {
    std::cout << "Distributing degrees of freedom" << std::endl;

    fe_collection.push_back(FE_Q<dim>(fe_degree));
    fe_collection.push_back(FE_Nothing<dim>());

    for (const auto &cell : dof_handler.active_cell_iterators())
      if(cell->is_locally_owned())
      {
        const NonMatching::LocationToLevelSet cell_location =
          mesh_classifier.location_to_level_set(cell);
          
      if (cell_location == NonMatching::LocationToLevelSet::intersected)
         cell->set_active_fe_index(ActiveFEIndex::lagrange);
      else 
      	 cell->set_active_fe_index(ActiveFEIndex::nothing);
      //TraceFEM
      // if (cell_location == NonMatching::LocationToLevelSet::outside)
      //   cell->set_active_fe_index(ActiveFEIndex::nothing);
      // else
      // cell->set_active_fe_index(ActiveFEIndex::lagrange);
      }

    dof_handler.distribute_dofs(fe_collection);
    std::cout<<" number of dofs: "<<dof_handler.n_dofs()<<std::endl;
  }



  // @sect3{Sparsity Pattern}
  // The added ghost penalty results in a sparsity pattern similar to a DG
  // method with a symmetric-interior-penalty term. Thus, we can use the
  // make_flux_sparsity_pattern() function to create it. However, since the
  // ghost-penalty terms only act on the faces in $\mathcal{F}_h$, we can pass
  // in a lambda function that tells make_flux_sparsity_pattern() over which
  // faces the flux-terms appear. This gives us a sparsity pattern with minimal
  // number of entries. When passing a lambda function,
  // make_flux_sparsity_pattern requires us to also pass cell and face coupling
  // tables to it. If the problem was vector-valued, these tables would allow us
  // to couple only some of the vector components. This is discussed in step-46.
  template <int dim>
  void LaplaceBeltramiSolver<dim>::initialize_matrices()
  {
    std::cout << "Initializing matrices" << std::endl;

    const auto face_has_flux_coupling = [&](const auto &       cell,
                                            const unsigned int face_index) {
      return this->face_has_ghost_penalty(cell, face_index);
    };

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

    const unsigned int           n_components = fe_collection.n_components();
    Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
    Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);
    cell_coupling[0][0] = DoFTools::always;
    face_coupling[0][0] = DoFTools::always;

    const AffineConstraints<double> constraints;
    const bool                      keep_constrained_dofs = true;

    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         dsp,
                                         constraints,
                                         keep_constrained_dofs,
                                         cell_coupling,
                                         face_coupling,
                                         numbers::invalid_subdomain_id,
                                         face_has_flux_coupling);
#ifdef USE_TRILINOS
    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs,
                                               MPI_COMM_WORLD,
                                               locally_relevant_dofs);
    stiffness_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
#else                                         
    sparsity_pattern.copy_from(dsp);

    stiffness_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    rhs.reinit(dof_handler.n_dofs());
#endif
  }



  // The following function describes which faces are part of the set
  // $\mathcal{F}_h$. That is, it returns true if the face of the incoming cell
  // belongs to the set $\mathcal{F}_h$.
  template <int dim>
  bool LaplaceBeltramiSolver<dim>::face_has_ghost_penalty(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    const unsigned int                                       face_index) const
  {
    if (cell->at_boundary(face_index))
      return false;

    const NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);

    const NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));


	/*TraceFEM: only internal to intersected
    if (cell_location == NonMatching::LocationToLevelSet::intersected &&
        neighbor_location != NonMatching::LocationToLevelSet::outside)
      return true;

    if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
        cell_location != NonMatching::LocationToLevelSet::outside)
      return true;
      */
      if (cell_location == NonMatching::LocationToLevelSet::intersected &&
        neighbor_location == NonMatching::LocationToLevelSet::intersected)
      return true;

    return false;
  }



  // @sect3{Assembling the System}
  template <int dim>
  void LaplaceBeltramiSolver<dim>::assemble_system()
  {
    std::cout << "Assembling" << std::endl;

    RightHandSide<dim> right_hand_side;

    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);
    Vector<double>     local_rhs(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    const double ghost_parameter   = 1;
    const double norm_grad_parameter  = 0;
    const double nitsche_parameter = 5 * (fe_degree + 1) * fe_degree;

    // Since the ghost penalty is similar to a DG flux term, the simplest way to
    // assemble it is to use an FEInterfaceValues object.
    const QGauss<dim - 1>  face_quadrature(fe_degree + 1);
    FEInterfaceValues<dim> fe_interface_values(fe_collection[0],
                                               face_quadrature,
                                               update_gradients |
                                                 update_JxW_values |
                                                 update_normal_vectors);


    // As we iterate over the cells in the mesh, we would in principle have to
    // do the following on each cell, $T$,
    //
    // 1. Construct one quadrature rule to integrate over the intersection with
    // the domain, $T \cap \Omega$, and one quadrature rule to integrate over
    // the intersection with the boundary, $T \cap \Gamma$.
    // 2. Create FEValues-like objects with the new quadratures.
    // 3. Assemble the local matrix using the created FEValues-objects.
    //
    // To make the assembly easier, we use the class NonMatching::FEValues,
    // which does the above steps 1 and 2 for us. The algorithm @cite saye_2015
    // that is used to generate the quadrature rules on the intersected cells
    // uses a 1-dimensional quadrature rule as base. Thus, we pass a 1D
    // Gauss--Legendre quadrature to the constructor of NonMatching::FEValues.
    // On the non-intersected cells, a tensor product of this 1D-quadrature will
    // be used.
    //
    // As stated in the introduction, each cell has 3 different regions: inside,
    // surface, and outside, where the level set function in each region is
    // negative, zero, and positive. We need an UpdateFlags variable for each
    // such region. These are stored on an object of type
    // NonMatching::RegionUpdateFlags, which we pass to NonMatching::FEValues.
    const QGauss<1> quadrature_1D(fe_degree + 1);
    
    QGauss<dim>     quadrature_formula(fe_degree + 1);
    FEValues<dim> fe_values(fe_collection[0],
                        quadrature_formula,
                        update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);
	

  
    // As we iterate over the cells, we don't need to do anything on the cells
    // that have FENothing elements. To disregard them we use an iterator
    // filter.
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() &&
          cell->active_fe_index() == ActiveFEIndex::lagrange)
        {
          local_stiffness = 0;
          local_rhs       = 0;

          const double cell_side_length = cell->minimum_vertex_distance();

          // First, we call the reinit function of our NonMatching::FEValues
          // object. In the background, NonMatching::FEValues uses the
          // MeshClassifier passed to its constructor to check if the incoming
          // cell is intersected. If that is the case, NonMatching::FEValues
          // calls the NonMatching::QuadratureGenerator in the background to
          // create the immersed quadrature rules.
          non_matching_fe_values.reinit(cell);

          // After calling reinit, we can retrieve a dealii::FEValues object
          // with quadrature points that corresponds to integrating over the
          // inside region of the cell. This is the object we use to do the
          // local assembly. This is similar to how hp::FEValues builds
          // dealii::FEValues objects. However, one difference here is that the
          // dealii::FEValues object is returned as an optional. This is a type
          // that wraps an object that may or may not be present. This requires
          // us to add an if-statement to check if the returned optional
          // contains a value, before we use it. This might seem odd at first.
          // Why does the function not just return a reference to a const
          // FEValues<dim>? The reason is that in an immersed method, we have
          // essentially no control of how the cuts occur. Even if the cell is
          // formally intersected: $T \cap \Omega \neq \emptyset$, it might be
          // that the cut is only of floating point size $|T \cap \Omega| \sim
          // \epsilon$. When this is the case, we can not expect that the
          // algorithm that generates the quadrature rule produces anything
          // useful. It can happen that the algorithm produces 0 quadrature
          // points. When this happens, the returned optional will not contain a
          // value, even if the cell is formally intersected.

          /* TraceFEM:
        /* TraceFEM: 
          /* TraceFEM:
          const std_cxx17::optional<FEValues<dim>> &inside_fe_values =
            non_matching_fe_values.get_inside_fe_values();

          if (inside_fe_values)
            for (const unsigned int q :
                 inside_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = inside_fe_values->quadrature_point(q);
                for (const unsigned int i : inside_fe_values->dof_indices())
                  {
                    for (const unsigned int j : inside_fe_values->dof_indices())
                      {
                        local_stiffness(i, j) +=
                          inside_fe_values->shape_grad(i, q) *
                          inside_fe_values->shape_grad(j, q) *
                          inside_fe_values->JxW(q);
                      }
                    local_rhs(i) += rhs_function.value(point) *
                                    inside_fe_values->shape_value(i, q) *
                                    inside_fe_values->JxW(q);
                  }
              }
    */

          // TraceFEM: normal-gradient-over-volume stabilization...
          /*const std_cxx17::optional<FEValues<dim>> &intersected_fe_values =
            non_matching_fe_values.get_intersected_fe_values(); //Doesn't exist!
  */
          const std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
            &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

          if (surface_fe_values)
            {
              typename DoFHandler<dim>::active_cell_iterator level_set_cell(
                &(triangulation),
                cell->level(),
                cell->index(),
                &level_set_dof_handler);
              fe_values.reinit(level_set_cell);
              std::vector<Tensor<1, dim>> normal(quadrature_formula.size());
              fe_values.get_function_gradients(level_set, normal);
           	fe_values.get_function_gradients(level_set, normal);		
              fe_values.get_function_gradients(level_set, normal);

              for (const unsigned int q : fe_values.quadrature_point_indices())
                {
                  const Point<dim> &point = fe_values.quadrature_point(q);
                  normal[q] = (1.0 / normal[q].norm()) * normal[q];
                  for (const unsigned int i : fe_values.dof_indices())
                    {
                      for (const unsigned int j : fe_values.dof_indices())
                        {
                          local_stiffness(i, j) +=
                            norm_grad_parameter *
                            (normal[q] * fe_values.shape_grad(i, q)) *
                            (normal[q] * fe_values.shape_grad(j, q)) *
                            fe_values.JxW(q);
                        }
                    }
                }
            }
          //...TraceFEM*/

          // In the same way, we can use NonMatching::FEValues to retrieve an
          // FEFaceValues-like object to integrate over $T \cap \Gamma$. The
          // only thing that is new here is the type of the object. The
          // transformation from quadrature weights to JxW-values is different
          // for surfaces, so we need a new class:
          // NonMatching::FEImmersedSurfaceValues. In addition to the ordinary
          // functions shape_value(..), shape_grad(..), etc., one can use its
          // normal_vector(..)-function to get an outward normal to the immersed
          // surface, $\Gamma$. In terms of the level set function, this normal
          // reads
          // @f{equation*}
          //   n = \frac{\nabla \psi}{\| \nabla \psi \|}.
          // @f}
          // An additional benefit of std::optional is that we do not need any
          // other check for whether we are on intersected cells: In case we are
          // on an inside cell, we get an empty object here.

          /*const std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
            &surface_fe_values = non_matching_fe_values.get_surface_fe_values();
  */
          if (surface_fe_values)
            {
              for (const unsigned int q :
                   surface_fe_values->quadrature_point_indices())
                {
                  const Point<dim> &point =
                    surface_fe_values->quadrature_point(q);
                  const Tensor<1, dim> &normal =
                    surface_fe_values->normal_vector(q);
                  for (const unsigned int i : surface_fe_values->dof_indices())
                    {
                      for (const unsigned int j :
                           surface_fe_values->dof_indices())
                        {
                          local_stiffness(i, j) +=
                            (
                              /*TraceFEM -normal *
                               surface_fe_values->shape_grad(i, q) *
                                 surface_fe_values->shape_value(j, q) +
                               -normal * surface_fe_values->shape_grad(j, q) *
                                 surface_fe_values->shape_value(i, q) +
                               nitsche_parameter / cell_side_length * */
                              surface_fe_values->shape_value(i, q) *
                                surface_fe_values->shape_value(j, q) +
                              (surface_fe_values->shape_grad(i, q) -
                               (normal * surface_fe_values->shape_grad(i, q)) *
                                 normal) *
                                (surface_fe_values->shape_grad(j, q) -
                                 (normal *
                                  surface_fe_values->shape_grad(j, q)) *
                                   normal)) *
                            surface_fe_values->JxW(q);
                        }

                      local_rhs(i) += right_hand_side.value(point) *
                                      surface_fe_values->shape_value(i, q) *
                                      surface_fe_values->JxW(q);
                      /*TraceFEM
                    /*TraceFEM 
                      /*TraceFEM
                      local_rhs(i) +=
                         boundary_condition.value(point) *
                        (nitsche_parameter / cell_side_length *
                           surface_fe_values->shape_value(i, q) -
                         normal * surface_fe_values->shape_grad(i, q)) *
                        surface_fe_values->JxW(q);*/
                    }
                }
            }


          cell->get_dof_indices(local_dof_indices);

          stiffness_matrix.add(local_dof_indices, local_stiffness);
          rhs.add(local_dof_indices, local_rhs);


          // The assembly of the ghost penalty term is straight forward. As we
          // iterate over the local faces, we first check if the current face
          // belongs to the set $\mathcal{F}_h$. The actual assembly is simple
          // using FEInterfaceValues. Assembling in this we will traverse each
          // internal face in the mesh twice, so in order to get the penalty
          // constant we expect, we multiply the penalty term with a factor 1/2.

          for (unsigned int f : cell->face_indices())
            if (face_has_ghost_penalty(cell, f))
              {
                const unsigned int invalid_subface =
                  numbers::invalid_unsigned_int;


                fe_interface_values.reinit(cell,
                                           f,
                                           invalid_subface,
                                           cell->neighbor(f),
                                           cell->neighbor_of_neighbor(f),
                                           invalid_subface);

                const unsigned int n_interface_dofs =
                  fe_interface_values.n_current_interface_dofs();
                FullMatrix<double> local_stabilization(n_interface_dofs,
                                                       n_interface_dofs);
                for (unsigned int q = 0;
                     q < fe_interface_values.n_quadrature_points;
                     ++q)
                  {
                    const Tensor<1, dim> normal = fe_interface_values.normal(q);
                    for (unsigned int i = 0; i < n_interface_dofs; ++i)
                      for (unsigned int j = 0; j < n_interface_dofs; ++j)
                        {
                          local_stabilization(i, j) +=
                            .5 *
                            ghost_parameter
                            // TraceFEM: * cell_side_length
                            * normal *
                            fe_interface_values.jump_in_shape_gradients(i, q) *
                            normal *
                            fe_interface_values.jump_in_shape_gradients(j, q) *
                            fe_interface_values.JxW(q);
                        }
                  }

                const std::vector<types::global_dof_index>
                  local_interface_dof_indices =
                    fe_interface_values.get_interface_dof_indices();

                // stiffness_matrix.add(local_interface_dof_indices,
                //                      local_stabilization);
              }
        }
#ifdef USE_TRILINOS        
      stiffness_matrix.compress(VectorOperation::add);
      rhs.compress(VectorOperation::add);
#endif
  }


  // @sect3{Solving the System}
  template <int dim>
  void LaplaceBeltramiSolver<dim>::solve()
  {
    std::string flag="Direct";
    if (flag=="Direct")
    {	
    	std::cout << "Solving directly... ";
    	Timer timer;
#ifdef USE_TRILINOS      
	    SolverControl                  solver_control(2000, 1e-8);
            TrilinosWrappers::SolverDirect::AdditionalData data;
            TrilinosWrappers::SolverDirect solver(solver_control, data);
            solver.solve(stiffness_matrix, solution, rhs);
#else
	    SparseDirectUMFPACK A_direct;
	    A_direct.initialize(stiffness_matrix);
	 
	    A_direct.vmult(solution, rhs);
#endif      
 
	    timer.stop();
	    std::cout << "took (" << timer.cpu_time() << "s)" << std::endl;
    }
    else
     {
      Timer timer;
	    std::cout << "Solving system" << std::endl;
#ifdef USE_TRILINOS
	    const unsigned int max_iterations = solution.size();
	    SolverControl      solver_control(max_iterations);
      // copied from ASPECT
      // Make the AMG preconditioner
      std::vector<std::vector<bool>> constant_modes;
      DoFTools::extract_constant_modes (dof_handler,
                                        ComponentMask(),
                                        constant_modes);      
      TrilinosWrappers::PreconditionAMG                 preconditioner_stiffness;
      TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
      Amg_data.constant_modes        = constant_modes;
      Amg_data.elliptic              = true;
      Amg_data.higher_order_elements = false;
      Amg_data.smoother_sweeps       = 2;
      Amg_data.aggregation_threshold = 0.02;
      Amg_data.output_details        = true;
      preconditioner_stiffness.initialize(stiffness_matrix);

      SolverCG<VectorType> cg(solver_control);
      cg.solve (stiffness_matrix, solution, rhs, preconditioner_stiffness);      
      std::cout << "   Solving system using AMG... " << solver_control.last_step() <<" iterations."<< std::endl;      
      number_of_iterations = solver_control.last_step();

#else
	    const unsigned int max_iterations = solution.size();
	    SolverControl      solver_control(max_iterations);
	    SolverCG<>         solver(solver_control);
	    SparseILU<double> ILU;
	    ILU.initialize(stiffness_matrix);
	    solver.solve(stiffness_matrix, solution, rhs, ILU);
	    std::cout<<" Number of iter:\t" << solver_control.last_step() << "\n";
#endif
	    timer.stop();
	    std::cout << "took (" << timer.cpu_time() << "s)" << std::endl;
    }
  }


  // @sect3{Data Output}
  // Since both DoFHandler instances use the same triangulation, we can add both
  // the level set function and the solution to the same vtu-file. Further, we
  // do not want to output the cells that have LocationToLevelSet value outside.
  // To disregard them, we write a small lambda function and use the
  // set_cell_selection function of the DataOut class.
  template <int dim>
  void LaplaceBeltramiSolver<dim>::output_results() const
  {
    std::cout << "Writing vtu file" << std::endl;

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

    data_out.set_cell_selection(
      [this](const typename Triangulation<dim>::cell_iterator &cell) {
        return cell->is_active() &&
               mesh_classifier.location_to_level_set(cell) ==
                 /* TraceFEM != NonMatching::LocationToLevelSet::outside;*/
                  NonMatching::LocationToLevelSet::intersected;
      });

#ifdef USE_TRILONOS
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    // have to create the directory output
    data_out.write_vtu_with_pvtu_record(
        output_dir, "solution", cycle, MPI_COMM_WORLD, 5, 1);
#else    
    data_out.build_patches();
    std::ofstream output("step-Trace.vtu");
    data_out.write_vtu(output);

#endif
  }



  // @sect3{$L^2$-Error}
  // To test that the implementation works as expected, we want to compute the
  // error in the solution in the $L^2$-norm. The analytical solution to the
  // Poisson problem stated in the introduction reads
  // @f{align*}
  //  u(x) = 1 - \frac{2}{\text{dim}}(\| x \|^2 - 1) , \qquad x \in
  //  \overline{\Omega}.
  // @f}
  // We first create a function corresponding to the analytical solution:
  template <int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
    double value(const Point<dim> & point,
                 const unsigned int component = 0) const override;
  };



  template <int dim>
  double AnalyticalSolution<dim>::value(const Point<dim> & point,
                                        const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);
    (void)component;

    return point[0];
    //return 1. - 2. / dim * (point.norm_square() - 1.);
  }



  // Of course, the analytical solution, and thus also the error, is only
  // defined in $\overline{\Omega}$. Thus, to compute the $L^2$-error we must
  // proceed in the same way as when we assembled the linear system. We first
  // create an NonMatching::FEValues object.
  template <int dim>
  double LaplaceBeltramiSolver<dim>::compute_L2_error() const
  {
    std::cout << "Computing L2 error..." << std::endl;

    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside =
      update_values | update_JxW_values | update_quadrature_points;
        region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;


    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);
// std::cout << "Computing L2 error...1..." << std::endl;
    // We then iterate iterate over the cells that have LocationToLevelSetValue
    // value inside or intersected again. For each quadrature point, we compute
    // the pointwise error and use this to compute the integral.
    const AnalyticalSolution<dim> analytical_solution;
    double                        error_L2_squared = 0;

    for (const auto &cell :
         dof_handler.active_cell_iterators() )
      if(cell->is_locally_owned() &&
         cell->active_fe_index() == ActiveFEIndex(lagrange))
      {
      // std::cout << "Computing L2 error...2" << std::endl;
        non_matching_fe_values.reinit(cell);
        
        /*TraceFEM
        const std_cxx17::optional<FEValues<dim>> &fe_values =
          non_matching_fe_values.get_inside_fe_values();

        if (fe_values)
          {
            std::VectorType solution_values(fe_values->n_quadrature_points);
            fe_values->get_function_values(solution, solution_values);

            for (const unsigned int q : fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = fe_values->quadrature_point(q);
                const double      error_at_point =
                  solution_values.at(q) - analytical_solution.value(point);
                error_L2_squared +=
                  std::pow(error_at_point, 2) * fe_values->JxW(q);
              }
          }
          */
          
           const std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

        if (surface_fe_values)
          {
          
          std::vector<double> solution_values(surface_fe_values->n_quadrature_points);
            surface_fe_values->get_function_values(solution, solution_values);

            for (const unsigned int q : surface_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = surface_fe_values->quadrature_point(q);
                const double      error_at_point =
                  solution_values.at(q) - analytical_solution.value(point);
                error_L2_squared +=
                  std::pow(error_at_point, 2) * surface_fe_values->JxW(q);
              }
              // std::cout<<" l2 error: "<< error_L2_squared << std::endl;
          }
      }
// std::cout << "Computing L2 error...3" << std::endl;
#ifdef USE_TRILINOS
// std::cout<<" l2 error: "<< error_L2_squared << std::endl;
    const double sum_global = Utilities::MPI::sum(error_L2_squared,
                             MPI_COMM_WORLD);
// std::cout << "Computing L2 error...4" << std::endl;                             
    return std::sqrt(sum_global);

#else
    return std::sqrt(error_L2_squared);
#endif
    std::cout<<" done!"<<std::endl;
  }



  // @sect3{A Convergence Study}
  // Finally, we do a convergence study to check that the $L^2$-error decreases
  // with the expected rate. We refine the background mesh a few times. In each
  // refinement cycle, we solve the problem, compute the error, and add the
  // $L^2$-error and the mesh size to a ConvergenceTable.
  template <int dim>
  void LaplaceBeltramiSolver<dim>::run()
  {
    ConvergenceTable   convergence_table;
    const unsigned int n_refinements = 5;

    make_grid();
    for (unsigned int cycle = 0; cycle <= n_refinements; cycle++)
      {
        std::cout << "Refinement cycle " << cycle << std::endl;
        triangulation.refine_global(1);
        setup_discrete_level_set();
        std::cout << "Classifying cells" << std::endl;
        mesh_classifier.reclassify();
        distribute_dofs();
        initialize_matrices();
        assemble_system();
        solve();
        if (cycle == n_refinements)
          output_results();
        const double error_L2 = compute_L2_error();
        const double cell_side_length =
          triangulation.begin_active()->minimum_vertex_distance();

        convergence_table.add_value("Cycle", cycle);
        convergence_table.add_value("iterations", number_of_iterations);
#ifdef USE_TRILINOS
        convergence_table.add_value("Mesh size", Utilities::MPI::min(cell_side_length, MPI_COMM_WORLD));
#else        
        convergence_table.add_value("Mesh size", cell_side_length);
#endif        
        convergence_table.add_value("L2-Error", error_L2);

        convergence_table.evaluate_convergence_rates(
          "L2-Error", ConvergenceTable::reduction_rate_log2);
        convergence_table.set_scientific("L2-Error", true);

        std::cout << std::endl;
#ifdef USE_TRILINOS
        ConditionalOStream pcout(
        std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
        convergence_table.write_text(pcout.get_stream());      
#else        
        convergence_table.write_text(std::cout);
#endif        
        std::cout << std::endl;
      }
  }

} // namespace StepTrace



// @sect3{The main() function}
int main(int argc, char *argv[])
{
#ifdef USE_TRILINOS
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, numbers::invalid_unsigned_int);
#endif  
  const int dim = 3;

  StepTrace::LaplaceBeltramiSolver<dim> LB_solver;
  LB_solver.run();
}
