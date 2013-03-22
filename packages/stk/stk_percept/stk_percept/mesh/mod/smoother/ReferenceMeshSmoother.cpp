#include <stk_percept/Percept.hpp>
#if !defined(__IBMCPP__) 

#include <stk_percept/mesh/mod/smoother/ReferenceMeshSmoother.hpp>
#include <stk_percept/mesh/mod/smoother/SmootherMetric.hpp>
#include <stk_percept/mesh/mod/smoother/MeshSmoother.hpp>
#include <stk_percept/mesh/mod/smoother/SpacingFieldUtil.hpp>

#include <stk_mesh/base/FieldParallel.hpp>
#include <stk_util/parallel/ParallelReduce.hpp>

#include <stdio.h>

#include "mpi.h"

namespace stk {
  namespace percept {

    void ReferenceMeshSmoother::sync_fields(int iter)
    {
      std::vector< const stk::mesh::FieldBase *> fields;
      fields.push_back(m_eMesh->get_coordinates_field());

      // only the aura = !locally_owned_part && !globally_shared_part (outer layer)
      stk::mesh::communicate_field_data(m_eMesh->get_bulk_data()->shared_aura(), fields); 
      // the shared part (just the shared boundary)
      //stk::mesh::communicate_field_data(*m_eMesh->get_bulk_data()->ghostings()[0], fields);
    }


    bool ReferenceMeshSmoother::check_convergence()
    {
      throw std::runtime_error("not implemented");
#if 0
      stk::all_reduce( m_eMesh->get_bulk_data()->parallel() , stk::ReduceMax<1>( & m_dmax ) );
      bool cond = (m_num_invalid == 0 && m_dmax < gradNorm);
      return cond;
#endif
      return false;
    }

    double ReferenceMeshSmoother::run_one_iteration()
    {
      throw std::runtime_error("not implemented");
      return 0.0;
    }

    static void print_comm_list( const BulkData & mesh , bool doit )
    {
      if ( doit ) {
        std::ostringstream msg ;

        msg << std::endl ;

        for ( std::vector<EntityCommListInfo>::const_iterator
                i =  mesh.comm_list().begin() ;
              i != mesh.comm_list().end() ; ++i ) {

          Entity entity = i->entity;
          msg << "P" << mesh.parallel_rank() << ": " ;

          print_entity_key( msg , MetaData::get(mesh) , i->key );

          msg << " owner(" << entity.owner_rank() << ")" ;

          if ( Modified == entity.state() ) { msg << " mod" ; }
          else if ( Deleted == entity.state() ) { msg << " del" ; }
          else { msg << "    " ; }

          for ( PairIterEntityComm ec = mesh.entity_comm(i->key); ! ec.empty() ; ++ec ) {
            msg << " gid, proc (" << ec->ghost_id << "," << ec->proc << ")" ;
          }
          msg << std::endl ;
        }

        std::cout << msg.str();
      }
    }

    void ReferenceMeshSmoother::run_algorithm()
    {
      //std::cout << "\nP[" << m_eMesh->get_rank() << "] tmp srk ReferenceMeshSmoother innerIter= " << innerIter << " parallelIterations= " << parallelIterations << std::endl;

      //if (!m_eMesh->get_rank()) 
      //std::cout << "\nP[" << m_eMesh->get_rank() << "] tmp srk ReferenceMeshSmoother: running shape improver... \n" << std::endl;

      PerceptMesh *eMesh = m_eMesh;
      m_num_nodes = m_eMesh->get_number_nodes();

      print_comm_list(*eMesh->get_bulk_data(), false);

      stk::mesh::FieldBase *coord_field           = eMesh->get_coordinates_field();
      stk::mesh::FieldBase *coord_field_current   = coord_field;
      stk::mesh::FieldBase *coord_field_projected = eMesh->get_field("coordinates_N"); 
      stk::mesh::FieldBase *coord_field_original  = eMesh->get_field("coordinates_NM1");
      stk::mesh::FieldBase *coord_field_lagged    = eMesh->get_field("coordinates_lagged");

      m_coord_field_original  = coord_field_original;
      m_coord_field_projected = coord_field_projected;
      m_coord_field_lagged    = coord_field_lagged;
      m_coord_field_current   = coord_field_current;

      eMesh->copy_field(coord_field_lagged, coord_field_original);

      // untangle
      SmootherMetricUntangle untangle_metric(eMesh);

      // shape-size-orient smooth
      SmootherMetricShapeSizeOrient shape_size_orient_metric(eMesh);

      // shape
      SmootherMetricShapeB1 shape_b1_metric(eMesh, m_use_ref_mesh);

      SmootherMetricShapeC1 shape_c1_metric(eMesh);

      SmootherMetricShapeSizeOrientB1 shape_size_orient_b1_metric(eMesh);

      // laplace
      SmootherMetricLaplace laplace_metric(eMesh);

      // scaled jacobian
      SmootherMetricScaledJacobianElemental scaled_jac_metric(eMesh);

      // scaled jacobian - nodal
      SmootherMetricScaledJacobianNodal scaled_jac_metric_nodal(eMesh);

      //double omegas[] = {0.0, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0};
      //double omegas[] = {0.001, 1.0};
      //double omegas[] = { 0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.4, 0.45,0.46,0.47,0.48,0.49,0.5,0.52,0.54,0.56,0.59, 0.6, 0.8, 1.0};
      double omegas[] = { 1.0};
      //double omegas[] = { 0.001, 0.01, 0.1, 0.2, 0.4, 1.0};
      //double omegas[] = {0.0, 0.001, 0.01, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.4, 0.6, 0.8, 1.0};
      int nomega = sizeof(omegas)/sizeof(omegas[0]);
      
      for (int outer = 0; outer < nomega; outer++)
        {
          double omega = (outer < nomega ? omegas[outer] : 1.0);
          m_omega = omega;
          m_omega_prev = omega;
          if (outer > 0) m_omega_prev = omegas[outer-1];

          // set current state and evaluate mesh validity (current = omega*project + (1-omega)*original)
          eMesh->nodal_field_axpbypgz(omega, coord_field_projected, (1.0-omega), coord_field_original, 0.0, coord_field_current);

          int num_invalid = MeshSmoother::parallel_count_invalid_elements(m_eMesh);

          if (!m_eMesh->get_rank()) 
            std::cout << "\ntmp srk ReferenceMeshSmoother num_invalid current= " << num_invalid << " for outer_iter= " << outer 
                      << " omega= " << omega
                      << (num_invalid ? " WARNING: invalid elements exist before smoothing" : " OK")
                      << std::endl;
          //if (num_invalid) return;
          m_num_invalid = num_invalid;
          m_untangled = (m_num_invalid == 0);

          eMesh->save_as("outer_iter_"+toString(outer)+"_m1_mesh.e");

          int iter_all=0;

          int do_anim = 0; // = frequency of anim writes
          if (do_anim)
            {
              eMesh->save_as("anim_all."+toString(iter_all)+".e");
            }

          int nstage=2;
          for (int stage = 0; stage < nstage; stage++)
            {
              m_stage = stage;
              if (stage==0) 
                {
                  m_metric = &untangle_metric;
                  //m_metric = &scaled_jac_metric_nodal;
                }
              else 
                {
                  int num_invalid_1 = MeshSmoother::parallel_count_invalid_elements(m_eMesh);
                  VERIFY_OP_ON(num_invalid_1, ==, 0, "Invalid elements exist for start of stage 2, aborting");

                  //m_metric = &shape_size_orient_metric;
                  //m_metric = &shape_size_orient_b1_metric;
                  m_metric = &shape_b1_metric;
                  //m_metric = &shape_c1_metric;
                  //m_metric = &laplace_metric;
                  //m_metric = &scaled_jac_metric;

                }

              for (int iter = 0; iter < innerIter; ++iter, ++iter_all)
                {
                  m_iter = iter;
                  m_num_invalid = MeshSmoother::parallel_count_invalid_elements(m_eMesh);
                  int num_invalid_0 = m_num_invalid;
                  if (!m_untangled && m_num_invalid == 0)
                    {
                      m_untangled = true;
                    }

                  //               if (!m_eMesh->get_rank() && num_invalid_0) 
                  //                 std::cout << "\ntmp srk ReferenceMeshSmoother num_invalid current= " << num_invalid_0 
                  //                           << (num_invalid ? " WARNING: invalid elements exist before smoothing" : "OK")
                  //                           << std::endl;

                  m_global_metric = run_one_iteration();

                  sync_fields(iter);
                  //num_invalid_0 = MeshSmoother::parallel_count_invalid_elements(m_eMesh);
                  //m_num_invalid = num_invalid_0;
                  bool conv = check_convergence();
                  if (!m_eMesh->get_rank())
                  {
                    std::cout << "P[" << m_eMesh->get_rank() << "] " << "INFO: iter= " << iter << " dmax= " << m_dmax << " m_dnew= " << m_dnew 
                              << " m_d0= " << m_d0 << " m_alpha= " << m_alpha << " m_scale= " << m_scale << " m_grad_norm= " << m_grad_norm << " m_grad_norm_scaled = " << m_grad_norm_scaled
                              << " num_invalid= " << num_invalid_0 
                              << " m_global_metric= " << m_global_metric << " stage= " << stage << " m_untangled= " << m_untangled
                              << std::endl;
                  }

                  if (do_anim)
                    {
                       eMesh->save_as("iter_"+toString(outer)+"_"+toString(stage)+"."+toString(iter+1)+".e");
                       if (iter_all % do_anim == 0) eMesh->save_as("anim_all."+toString(iter_all+1)+".e");
                    }

                  if (!m_untangled && m_num_invalid == 0)
                    {
                      m_untangled = true;
                    }
                  if (conv && m_untangled) 
                    {
                      //std::cout << "P[" << m_eMesh->get_rank() << "] tmp srk a1 converged, stage= " << m_stage << std::endl;
                      break;
                    }
                }

              eMesh->save_as("outer_iter_"+toString(outer)+"_"+toString(stage)+"_mesh.e");
            }

          eMesh->copy_field(coord_field_lagged, coord_field);

        }

      //if (!m_eMesh->get_rank()) 

      MPI_Barrier( MPI_COMM_WORLD );

      if (!m_eMesh->get_rank()) 
        std::cout << "\nP[" << m_eMesh->get_rank() << "] tmp srk ReferenceMeshSmoother: running shape improver... done \n" << std::endl;

    }

  }
}


#endif
