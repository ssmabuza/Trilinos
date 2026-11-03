// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_BernsteinBasis_HGRAD_TET.hpp
    \brief  H(grad) Bernstein basis on the reference tetrahedron.
 */

#ifndef Intrepid2_BernsteinBasis_HGRAD_TET_h
#define Intrepid2_BernsteinBasis_HGRAD_TET_h

#include <Kokkos_DynRankView.hpp>

#include <Intrepid2_config.h>

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_LINE.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_TRI.hpp"
#include "Intrepid2_Polynomials.hpp"
#include "Intrepid2_Utils.hpp"

namespace Intrepid2
{
  namespace Impl
  {
    /** \brief Evaluate trivariate Bernstein polynomials and their partial derivatives at a single point
               using precomputed barycentric coordinate powers pow_lambda(0..3, 0..p).
     */
    template<typename OutputScalar, typename PowViewType>
    KOKKOS_INLINE_FUNCTION
    OutputScalar evalBernsteinTetValue(const PowViewType &pow_lambda,
                                       const ordinal_type q,
                                       const ordinal_type g0,
                                       const ordinal_type g1,
                                       const ordinal_type g2,
                                       const ordinal_type g3)
    {
      const double coeff = Polynomials::binomialCoefficient(q, g3)
                         * Polynomials::binomialCoefficient(q - g3, g2)
                         * Polynomials::binomialCoefficient(q - g3 - g2, g1);
      return coeff * pow_lambda(0, g0) * pow_lambda(1, g1) * pow_lambda(2, g2) * pow_lambda(3, g3);
    }

    template<typename OutputScalar, typename PowViewType>
    KOKKOS_INLINE_FUNCTION
    OutputScalar evalBernsteinTetDeriv(const PowViewType &pow_lambda,
                                       const ordinal_type p,
                                       const ordinal_type a0,
                                       const ordinal_type a1,
                                       const ordinal_type a2,
                                       const ordinal_type a3,
                                       const ordinal_type kx,
                                       const ordinal_type ky,
                                       const ordinal_type kz)
    {
      const ordinal_type k = kx + ky + kz;
      if (k == 0) {
        return evalBernsteinTetValue<OutputScalar>(pow_lambda, p, a0, a1, a2, a3);
      }
      if (k > p) {
        return OutputScalar(0.0);
      }

      const ordinal_type q = p - k;
      double scale = 1.0;
      for (ordinal_type m = 0; m < k; ++m) {
        scale *= static_cast<double>(p - m);
      }

      const ordinal_type r1_max = (kx < a1) ? kx : a1;
      const ordinal_type r2_max = (ky < a2) ? ky : a2;
      const ordinal_type r3_max = (kz < a3) ? kz : a3;

      OutputScalar sum = 0.0;
      for (ordinal_type r1 = 0; r1 <= r1_max; ++r1) {
        const double binom_x = Polynomials::binomialCoefficient(kx, r1);
        for (ordinal_type r2 = 0; r2 <= r2_max; ++r2) {
          const double binom_xy = binom_x * Polynomials::binomialCoefficient(ky, r2);
          const ordinal_type r3_min_raw = (k - a0) - r1 - r2;
          const ordinal_type r3_min = (r3_min_raw > 0) ? r3_min_raw : 0;
          for (ordinal_type r3 = r3_min; r3 <= r3_max; ++r3) {
            const ordinal_type g0 = a0 - (k - r1 - r2 - r3);
            const ordinal_type g1 = a1 - r1;
            const ordinal_type g2 = a2 - r2;
            const ordinal_type g3 = a3 - r3;
            const double sign = ((k - r1 - r2 - r3) % 2 == 0) ? 1.0 : -1.0;
            const double w = sign * binom_xy * Polynomials::binomialCoefficient(kz, r3);
            sum += w * evalBernsteinTetValue<OutputScalar>(pow_lambda, q, g0, g1, g2, g3);
          }
        }
      }
      return scale * sum;
    }

    template<typename OutputScalar, typename PointScalar, typename OutputFieldType, typename InputPointsType, typename PowViewType>
    KOKKOS_INLINE_FUNCTION
    void evalBernsteinTetPoint(const EOperator opType,
                               const OutputFieldType &output,
                               const InputPointsType &inputPoints,
                               const ordinal_type pointOrdinal,
                               const ordinal_type polyOrder,
                               PowViewType &pow_lambda)
    {
      const PointScalar x = inputPoints(pointOrdinal, 0);
      const PointScalar y = inputPoints(pointOrdinal, 1);
      const PointScalar z = inputPoints(pointOrdinal, 2);

      const OutputScalar lam[4] = {
        OutputScalar(1.0 - x - y - z),
        OutputScalar(x),
        OutputScalar(y),
        OutputScalar(z)
      };

      for (int m = 0; m < 4; ++m) {
        pow_lambda(m, 0) = 1.0;
        for (ordinal_type a = 1; a <= polyOrder; ++a) {
          pow_lambda(m, a) = pow_lambda(m, a - 1) * lam[m];
        }
      }

      switch (opType)
      {
        case OPERATOR_VALUE:
        {
          ordinal_type fieldOrd = 0;
          for (ordinal_type a3 = 0; a3 <= polyOrder; ++a3) {
            for (ordinal_type a2 = 0; a2 <= polyOrder - a3; ++a2) {
              for (ordinal_type a1 = 0; a1 <= polyOrder - a3 - a2; ++a1) {
                const ordinal_type a0 = polyOrder - a1 - a2 - a3;
                output.access(fieldOrd, pointOrdinal) = evalBernsteinTetValue<OutputScalar>(pow_lambda, polyOrder, a0, a1, a2, a3);
                ++fieldOrd;
              }
            }
          }
          break;
        }
        case OPERATOR_GRAD:
        case OPERATOR_D1:
        {
          ordinal_type fieldOrd = 0;
          for (ordinal_type a3 = 0; a3 <= polyOrder; ++a3) {
            for (ordinal_type a2 = 0; a2 <= polyOrder - a3; ++a2) {
              for (ordinal_type a1 = 0; a1 <= polyOrder - a3 - a2; ++a1) {
                const ordinal_type a0 = polyOrder - a1 - a2 - a3;
                output.access(fieldOrd, pointOrdinal, 0) = evalBernsteinTetDeriv<OutputScalar>(pow_lambda, polyOrder, a0, a1, a2, a3, 1, 0, 0);
                output.access(fieldOrd, pointOrdinal, 1) = evalBernsteinTetDeriv<OutputScalar>(pow_lambda, polyOrder, a0, a1, a2, a3, 0, 1, 0);
                output.access(fieldOrd, pointOrdinal, 2) = evalBernsteinTetDeriv<OutputScalar>(pow_lambda, polyOrder, a0, a1, a2, a3, 0, 0, 1);
                ++fieldOrd;
              }
            }
          }
          break;
        }
        case OPERATOR_D2:
        case OPERATOR_D3:
        case OPERATOR_D4:
        case OPERATOR_D5:
        case OPERATOR_D6:
        case OPERATOR_D7:
        case OPERATOR_D8:
        case OPERATOR_D9:
        case OPERATOR_D10:
        {
          const ordinal_type k = getOperatorOrder(opType);
          ordinal_type fieldOrd = 0;
          for (ordinal_type a3 = 0; a3 <= polyOrder; ++a3) {
            for (ordinal_type a2 = 0; a2 <= polyOrder - a3; ++a2) {
              for (ordinal_type a1 = 0; a1 <= polyOrder - a3 - a2; ++a1) {
                const ordinal_type a0 = polyOrder - a1 - a2 - a3;
                for (ordinal_type kx = k; kx >= 0; --kx) {
                  for (ordinal_type ky = k - kx; ky >= 0; --ky) {
                    const ordinal_type kz = k - kx - ky;
                    const ordinal_type dkIdx = getDkEnumeration<3>(kx, ky, kz);
                    output.access(fieldOrd, pointOrdinal, dkIdx) = evalBernsteinTetDeriv<OutputScalar>(pow_lambda, polyOrder, a0, a1, a2, a3, kx, ky, kz);
                  }
                }
                ++fieldOrd;
              }
            }
          }
          break;
        }
        case OPERATOR_MAX:
        {
          const ordinal_type numFields = output.extent_int(0);
          const ordinal_type numComp = (output.rank() >= 3) ? output.extent_int(2) : 1;
          for (ordinal_type f = 0; f < numFields; ++f) {
            for (ordinal_type c = 0; c < numComp; ++c) {
              output.access(f, pointOrdinal, c) = 0.0;
            }
          }
          break;
        }
        default:
          INTREPID2_TEST_FOR_ABORT(true, ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_TET) Unsupported operator type");
      }
    }
  } // namespace Impl

  /** \class  Intrepid2::Bernstein_HGRAD_TET_Functor
      \brief  Functor for computing values for the BernsteinBasis_HGRAD_TET and BernsteinBasis_HVOL_TET classes.
   */
  template<class DeviceType, class OutputScalar, class PointScalar,
           class OutputFieldType, class InputPointsType>
  struct Bernstein_HGRAD_TET_Functor
  {
    using ExecutionSpace      = typename DeviceType::execution_space;
    using ScratchSpace        = typename ExecutionSpace::scratch_memory_space;
    using OutputScratchView2D = Kokkos::View<OutputScalar**,ScratchSpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using TeamMember = typename TeamPolicy::member_type;

    EOperator opType_;

    OutputFieldType output_;
    InputPointsType inputPoints_;

    ordinal_type polyOrder_;
    ordinal_type numFields_, numPoints_;

    size_t fad_size_output_;

    Bernstein_HGRAD_TET_Functor(EOperator opType, OutputFieldType output, InputPointsType inputPoints,
                                ordinal_type polyOrder)
    : opType_(opType), output_(output), inputPoints_(inputPoints),
      polyOrder_(polyOrder),
      fad_size_output_(getScalarDimensionForView(output))
    {
      numFields_ = output.extent_int(0);
      numPoints_ = output.extent_int(1);
      INTREPID2_TEST_FOR_EXCEPTION(numPoints_ != inputPoints.extent_int(0), std::invalid_argument, "point counts need to match!");
      INTREPID2_TEST_FOR_EXCEPTION(numFields_ != (polyOrder_ + 1) * (polyOrder_ + 2) * (polyOrder_ + 3) / 6, std::invalid_argument, "output field size does not match basis cardinality");
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const TeamMember & teamMember ) const
    {
      auto pointOrdinal = teamMember.league_rank();
      OutputScratchView2D pow_lambda;
      if (fad_size_output_ > 0) {
        pow_lambda = OutputScratchView2D(teamMember.team_shmem(), 4, polyOrder_ + 1, fad_size_output_);
      } else {
        pow_lambda = OutputScratchView2D(teamMember.team_shmem(), 4, polyOrder_ + 1);
      }

      Impl::evalBernsteinTetPoint<OutputScalar, PointScalar>(opType_, output_, inputPoints_, pointOrdinal, polyOrder_, pow_lambda);
    }

    size_t team_shmem_size (int /*team_size*/) const
    {
      if (fad_size_output_ > 0)
        return OutputScratchView2D::shmem_size(4, polyOrder_ + 1, fad_size_output_);
      else
        return OutputScratchView2D::shmem_size(4, polyOrder_ + 1);
    }
  };

  /** \class  Intrepid2::BernsteinBasis_HGRAD_TET
      \brief  Basis defining the trivariate Bernstein polynomial basis of degree p on the reference tetrahedron,
              a polynomial subspace of H(grad).

              On shards::Tetrahedron<4> with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1), the barycentric coordinates are
                lambda_0 = 1 - x - y - z,  lambda_1 = x,  lambda_2 = y,  lambda_3 = z,
              and the basis functions are
                B_{(a0,a1,a2,a3)}^p(x,y,z) = (p! / (a0! a1! a2! a3!)) lambda_0^{a0} lambda_1^{a1} lambda_2^{a2} lambda_3^{a3}
              enumerated in lattice order (a3 = 0..p, a2 = 0..p-a3, a1 = 0..p-a3-a2, a0 = p - a1 - a2 - a3).
  */
  template<typename DeviceType,
           typename OutputScalar = double,
           typename PointScalar  = double,
           bool defineVertexFunctions = true>
  class BernsteinBasis_HGRAD_TET
  : public Basis<DeviceType,OutputScalar,PointScalar>
  {
  public:
    using BasisBase = Basis<DeviceType,OutputScalar,PointScalar>;
    using HostBasis = BernsteinBasis_HGRAD_TET<typename Kokkos::HostSpace::device_type,OutputScalar,PointScalar,defineVertexFunctions>;

    using typename BasisBase::OrdinalTypeArray1DHost;
    using typename BasisBase::OrdinalTypeArray2DHost;

    using typename BasisBase::OutputViewType;
    using typename BasisBase::PointViewType;
    using typename BasisBase::ScalarViewType;

    using typename BasisBase::ExecutionSpace;
    using typename BasisBase::OutputValueType;
    using typename BasisBase::PointValueType;

  protected:
    ordinal_type polyOrder_;
    EPointType pointType_;

    static ordinal_type simplexFaceInteriorOrdinal(const ordinal_type p,
                                                   const ordinal_type g1,
                                                   const ordinal_type g2)
    {
      // Matches BernsteinBasis_HGRAD_TRI interior enumeration (g2 = 1..p-2, g1 = 1..p-1-g2)
      return (g2 - 1) * (p - 1) - ((g2 - 1) * g2) / 2 + (g1 - 1);
    }

  public:
    /** \brief  Constructor.
        \param [in] polyOrder - the polynomial order of the basis.
        \param [in] pointType - point type for DoF coordinates (POINTTYPE_DEFAULT or POINTTYPE_EQUISPACED).
     */
    BernsteinBasis_HGRAD_TET(ordinal_type polyOrder, const EPointType pointType = POINTTYPE_DEFAULT)
    : polyOrder_(polyOrder),
      pointType_(pointType)
    {
      INTREPID2_TEST_FOR_EXCEPTION(defineVertexFunctions && polyOrder < 1, std::invalid_argument,
                                   ">>> ERROR (Intrepid2::BernsteinBasis_HGRAD_TET): polyOrder must be >= 1 when defineVertexFunctions is true.");
      INTREPID2_TEST_FOR_EXCEPTION(!defineVertexFunctions && polyOrder < 0, std::invalid_argument,
                                   ">>> ERROR (Intrepid2::BernsteinBasis_HGRAD_TET): polyOrder must be >= 0.");
      INTREPID2_TEST_FOR_EXCEPTION(pointType != POINTTYPE_DEFAULT && pointType != POINTTYPE_EQUISPACED,
                                   std::invalid_argument,
                                   ">>> ERROR (Intrepid2::BernsteinBasis_HGRAD_TET): PointType must be POINTTYPE_DEFAULT or POINTTYPE_EQUISPACED.");

      this->basisCardinality_     = ((polyOrder + 1) * (polyOrder + 2) * (polyOrder + 3)) / 6;
      this->basisDegree_          = polyOrder;
      this->basisCellTopologyKey_ = shards::Tetrahedron<4>::key;
      this->basisType_            = BASIS_FEM_DEFAULT;
      this->basisCoordinates_     = COORDINATES_CARTESIAN;
      this->functionSpace_        = FUNCTION_SPACE_HGRAD;

      const ordinal_type card = this->basisCardinality_;

      // Populate Bezier domain points in reference tetrahedron coordinates
      Kokkos::DynRankView<typename ScalarViewType::value_type, typename DeviceType::execution_space::array_layout, Kokkos::HostSpace>
        dofCoordsHost("BernsteinBasis::HGRAD::TET::dofCoords", card, 3);

      if (polyOrder == 0) {
        dofCoordsHost(0, 0) = 0.25;
        dofCoordsHost(0, 1) = 0.25;
        dofCoordsHost(0, 2) = 0.25;
      } else {
        ordinal_type f = 0;
        for (ordinal_type a3 = 0; a3 <= polyOrder; ++a3) {
          for (ordinal_type a2 = 0; a2 <= polyOrder - a3; ++a2) {
            for (ordinal_type a1 = 0; a1 <= polyOrder - a3 - a2; ++a1) {
              dofCoordsHost(f, 0) = static_cast<double>(a1) / static_cast<double>(polyOrder);
              dofCoordsHost(f, 1) = static_cast<double>(a2) / static_cast<double>(polyOrder);
              dofCoordsHost(f, 2) = static_cast<double>(a3) / static_cast<double>(polyOrder);
              ++f;
            }
          }
        }
      }
      this->dofCoords_ = Kokkos::create_mirror_view(typename DeviceType::memory_space(), dofCoordsHost);
      Kokkos::deep_copy(this->dofCoords_, dofCoordsHost);

      // Initialize subcell DoF tags aligned with Shards Tetrahedron<4> subcell orientations:
      // Edges: 0:(0,1), 1:(1,2), 2:(2,0), 3:(0,3), 4:(1,3), 5:(2,3)
      // Faces: 0:(0,1,3), 1:(1,2,3), 2:(0,3,2), 3:(0,2,1)
      {
        const ordinal_type tagSize  = 4;
        const ordinal_type posScDim = 0;
        const ordinal_type posScOrd = 1;
        const ordinal_type posDfOrd = 2;

        OrdinalTypeArray1DHost tagView("tag view", card * tagSize);

        if (defineVertexFunctions && polyOrder >= 1) {
          const ordinal_type numEdgeDof = polyOrder - 1;
          const ordinal_type numFaceDof = (polyOrder > 2) ? ((polyOrder - 1) * (polyOrder - 2)) / 2 : 0;
          const ordinal_type numVolDof  = (polyOrder > 3) ? ((polyOrder - 1) * (polyOrder - 2) * (polyOrder - 3)) / 6 : 0;

          const ordinal_type edgeVerts[6][2] = {
            {0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}
          };

          ordinal_type f = 0;
          ordinal_type volLocalDof = 0;
          for (ordinal_type a3 = 0; a3 <= polyOrder; ++a3) {
            for (ordinal_type a2 = 0; a2 <= polyOrder - a3; ++a2) {
              for (ordinal_type a1 = 0; a1 <= polyOrder - a3 - a2; ++a1) {
                const ordinal_type a0 = polyOrder - a1 - a2 - a3;
                const ordinal_type a[4] = {a0, a1, a2, a3};

                int nnz = 0;
                for (int m = 0; m < 4; ++m) {
                  if (a[m] > 0) ++nnz;
                }

                if (nnz == 1) {
                  // Vertex DoF
                  ordinal_type vOrd = -1;
                  for (int m = 0; m < 4; ++m) {
                    if (a[m] == polyOrder) { vOrd = m; break; }
                  }
                  tagView(f * tagSize + 0) = 0;
                  tagView(f * tagSize + 1) = vOrd;
                  tagView(f * tagSize + 2) = 0;
                  tagView(f * tagSize + 3) = 1;
                } else if (nnz == 2) {
                  // Edge interior DoF
                  ordinal_type eOrd = -1;
                  for (ordinal_type e = 0; e < 6; ++e) {
                    if (a[edgeVerts[e][0]] > 0 && a[edgeVerts[e][1]] > 0) {
                      eOrd = e;
                      break;
                    }
                  }
                  const ordinal_type localDof = a[edgeVerts[eOrd][1]] - 1;
                  tagView(f * tagSize + 0) = 1;
                  tagView(f * tagSize + 1) = eOrd;
                  tagView(f * tagSize + 2) = localDof;
                  tagView(f * tagSize + 3) = numEdgeDof;
                } else if (nnz == 3) {
                  // Face interior DoF
                  ordinal_type faceOrd = -1;
                  ordinal_type g1 = 0, g2 = 0;
                  if (a[2] == 0) {
                    // Face 0: (0, 1, 3)
                    faceOrd = 0; g1 = a[1]; g2 = a[3];
                  } else if (a[0] == 0) {
                    // Face 1: (1, 2, 3)
                    faceOrd = 1; g1 = a[2]; g2 = a[3];
                  } else if (a[1] == 0) {
                    // Face 2: (0, 3, 2)
                    faceOrd = 2; g1 = a[3]; g2 = a[2];
                  } else {
                    // Face 3: (0, 2, 1) (a[3] == 0)
                    faceOrd = 3; g1 = a[2]; g2 = a[1];
                  }
                  const ordinal_type localDof = simplexFaceInteriorOrdinal(polyOrder, g1, g2);
                  tagView(f * tagSize + 0) = 2;
                  tagView(f * tagSize + 1) = faceOrd;
                  tagView(f * tagSize + 2) = localDof;
                  tagView(f * tagSize + 3) = numFaceDof;
                } else {
                  // Volume interior DoF (nnz == 4)
                  tagView(f * tagSize + 0) = 3;
                  tagView(f * tagSize + 1) = 0;
                  tagView(f * tagSize + 2) = volLocalDof++;
                  tagView(f * tagSize + 3) = numVolDof;
                }
                ++f;
              }
            }
          }
          INTREPID2_TEST_FOR_EXCEPTION(f != card || volLocalDof != numVolDof, std::logic_error,
                                       ">>> ERROR (Intrepid2::BernsteinBasis_HGRAD_TET): internal tag enumeration error.");
        } else {
          for (ordinal_type i = 0; i < card; ++i) {
            tagView(i * tagSize + 0) = 3;
            tagView(i * tagSize + 1) = 0;
            tagView(i * tagSize + 2) = i;
            tagView(i * tagSize + 3) = card;
          }
        }

        this->setOrdinalTagData(this->tagToOrdinal_,
                                this->ordinalToTag_,
                                tagView,
                                this->basisCardinality_,
                                tagSize,
                                posScDim,
                                posScOrd,
                                posDfOrd);
      }
    }

    const char* getName() const override {
      return "Intrepid2_BernsteinBasis_HGRAD_TET";
    }

    virtual bool requireOrientation() const override {
      return (defineVertexFunctions && this->getDegree() > 2);
    }

    using BasisBase::getValues;

    virtual void getValues( const ExecutionSpace& space,
                                  OutputViewType  outputValues,
                            const PointViewType   inputPoints,
                            const EOperator       operatorType = OPERATOR_VALUE ) const override
    {
#ifdef HAVE_INTREPID2_DEBUG
      Intrepid2::getValues_HGRAD_Args(outputValues,
                                      inputPoints,
                                      operatorType,
                                      this->getBaseCellTopology(),
                                      this->getCardinality());
#endif
      auto numPoints = inputPoints.extent_int(0);

      using FunctorType = Bernstein_HGRAD_TET_Functor<DeviceType, OutputScalar, PointScalar, OutputViewType, PointViewType>;
      FunctorType functor(operatorType, outputValues, inputPoints, polyOrder_);

      const int outputVectorSize = getVectorSizeForHierarchicalParallelism<OutputScalar>();
      const int pointVectorSize  = getVectorSizeForHierarchicalParallelism<PointScalar>();
      const int vectorSize = std::max(outputVectorSize, pointVectorSize);
      const int teamSize = 1;

      auto policy = Kokkos::TeamPolicy<ExecutionSpace>(space, numPoints, teamSize, vectorSize);
      Kokkos::parallel_for("Bernstein_HGRAD_TET_Functor", policy, functor);
    }

    virtual void getValues( OutputViewType outputValues,
                            const PointViewType inputPoints,
                            const EOperator operatorType = OPERATOR_VALUE ) const override
    {
      this->getValues(ExecutionSpace{}, outputValues, inputPoints, operatorType);
    }

    virtual void getScratchSpaceSize( ordinal_type& perThreadSpaceSize,
                                      const PointViewType inputPoints,
                                      const EOperator /*operatorType*/ = OPERATOR_VALUE ) const override
    {
      using ScratchView2D = Kokkos::View<OutputScalar**, typename ExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      const size_t fadSize = getScalarDimensionForView(inputPoints);
      if (fadSize > 0)
        perThreadSpaceSize = ScratchView2D::shmem_size(4, polyOrder_ + 1, fadSize);
      else
        perThreadSpaceSize = ScratchView2D::shmem_size(4, polyOrder_ + 1);
    }

    KOKKOS_INLINE_FUNCTION
    virtual void getValues( OutputViewType outputValues,
                            const PointViewType inputPoints,
                            const EOperator operatorType,
                            const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type& team_member,
                            const int threadScratchLevel,
                            const ordinal_type subcellDim = -1,
                            const ordinal_type subcellOrdinal = -1 ) const override
    {
      INTREPID2_TEST_FOR_ABORT(!((subcellDim == -1) && (subcellOrdinal == -1)),
        ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_TET::getValues) Selecting subsets of basis functions is not supported.");

      using ScratchView2D = Kokkos::View<OutputScalar**, typename ExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      const size_t fadSize = getScalarDimensionForView(outputValues);
      ScratchView2D pow_lambda = (fadSize > 0)
        ? ScratchView2D(team_member.thread_scratch(threadScratchLevel), 4, this->polyOrder_ + 1, fadSize)
        : ScratchView2D(team_member.thread_scratch(threadScratchLevel), 4, this->polyOrder_ + 1);

      const int numPoints = inputPoints.extent_int(0);
      const ordinal_type polyOrder = this->polyOrder_;

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, numPoints), [=](const int pt) {
        auto pow_local = pow_lambda;
        Impl::evalBernsteinTetPoint<OutputScalar, PointScalar>(operatorType, outputValues, inputPoints, pt, polyOrder, pow_local);
      });
    }

    virtual void getDofCoords( ScalarViewType dofCoords ) const override
    {
#ifdef HAVE_INTREPID2_DEBUG
      INTREPID2_TEST_FOR_EXCEPTION( rank(dofCoords) != 2, std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_TET::getDofCoords) rank = 2 required for dofCoords array");
      INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoords.extent(0)) != this->getCardinality(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_TET::getDofCoords) mismatch in number of dof and 0th dimension of dofCoords array");
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.extent(1) != this->getBaseCellTopology().getDimension(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_TET::getDofCoords) incorrect reference cell (1st) dimension in dofCoords array");
#endif
      Kokkos::deep_copy(dofCoords, this->dofCoords_);
    }

    BasisPtr<DeviceType,OutputScalar,PointScalar>
    getSubCellRefBasis(const ordinal_type subCellDim, const ordinal_type subCellOrd) const override
    {
      if (subCellDim == 1 && subCellOrd >= 0 && subCellOrd < 6) {
        return Teuchos::rcp(new BernsteinBasis_HGRAD_LINE<DeviceType,OutputScalar,PointScalar>(this->basisDegree_, pointType_));
      } else if (subCellDim == 2 && subCellOrd >= 0 && subCellOrd < 4) {
        return Teuchos::rcp(new BernsteinBasis_HGRAD_TRI<DeviceType,OutputScalar,PointScalar>(this->basisDegree_, pointType_));
      }
      INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Input parameters out of bounds");
    }

    virtual HostBasisPtr<OutputScalar, PointScalar>
    getHostBasis() const override {
      return Teuchos::rcp( new HostBasis(polyOrder_, pointType_) );
    }
  };
} // end namespace Intrepid2

#endif /* Intrepid2_BernsteinBasis_HGRAD_TET_h */

