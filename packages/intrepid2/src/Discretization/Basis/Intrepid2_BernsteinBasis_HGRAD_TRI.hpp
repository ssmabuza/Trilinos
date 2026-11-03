// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_BernsteinBasis_HGRAD_TRI.hpp
    \brief  H(grad) Bernstein basis on the reference triangle.
 */

#ifndef Intrepid2_BernsteinBasis_HGRAD_TRI_h
#define Intrepid2_BernsteinBasis_HGRAD_TRI_h

#include <Kokkos_DynRankView.hpp>

#include <Intrepid2_config.h>

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_LINE.hpp"
#include "Intrepid2_Polynomials.hpp"
#include "Intrepid2_Utils.hpp"

namespace Intrepid2
{
  namespace Impl
  {
    /** \brief Evaluate bivariate Bernstein polynomials and their partial derivatives at a single point
               using precomputed barycentric coordinate powers pow_lambda(0..2, 0..p).
     */
    template<typename OutputScalar, typename PowViewType>
    KOKKOS_INLINE_FUNCTION
    OutputScalar evalBernsteinTriValue(const PowViewType &pow_lambda,
                                       const ordinal_type q,
                                       const ordinal_type g0,
                                       const ordinal_type g1,
                                       const ordinal_type g2)
    {
      const double coeff = Polynomials::binomialCoefficient(q, g2) * Polynomials::binomialCoefficient(q - g2, g1);
      return coeff * pow_lambda(0, g0) * pow_lambda(1, g1) * pow_lambda(2, g2);
    }

    template<typename OutputScalar, typename PowViewType>
    KOKKOS_INLINE_FUNCTION
    OutputScalar evalBernsteinTriDeriv(const PowViewType &pow_lambda,
                                       const ordinal_type p,
                                       const ordinal_type a0,
                                       const ordinal_type a1,
                                       const ordinal_type a2,
                                       const ordinal_type kx,
                                       const ordinal_type ky)
    {
      const ordinal_type k = kx + ky;
      if (k == 0) {
        return evalBernsteinTriValue<OutputScalar>(pow_lambda, p, a0, a1, a2);
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

      OutputScalar sum = 0.0;
      for (ordinal_type r1 = 0; r1 <= r1_max; ++r1) {
        const double binom_x = Polynomials::binomialCoefficient(kx, r1);
        const ordinal_type r2_min_raw = (k - a0) - r1;
        const ordinal_type r2_min = (r2_min_raw > 0) ? r2_min_raw : 0;
        for (ordinal_type r2 = r2_min; r2 <= r2_max; ++r2) {
          const ordinal_type g0 = a0 - (k - r1 - r2);
          const ordinal_type g1 = a1 - r1;
          const ordinal_type g2 = a2 - r2;
          const double sign = ((k - r1 - r2) % 2 == 0) ? 1.0 : -1.0;
          const double w = sign * binom_x * Polynomials::binomialCoefficient(ky, r2);
          sum += w * evalBernsteinTriValue<OutputScalar>(pow_lambda, q, g0, g1, g2);
        }
      }
      return scale * sum;
    }

    template<typename OutputScalar, typename PointScalar, typename OutputFieldType, typename InputPointsType, typename PowViewType>
    KOKKOS_INLINE_FUNCTION
    void evalBernsteinTriPoint(const EOperator opType,
                               const OutputFieldType &output,
                               const InputPointsType &inputPoints,
                               const ordinal_type pointOrdinal,
                               const ordinal_type polyOrder,
                               PowViewType &pow_lambda)
    {
      const PointScalar x = inputPoints(pointOrdinal, 0);
      const PointScalar y = inputPoints(pointOrdinal, 1);

      const OutputScalar lam[3] = { OutputScalar(1.0 - x - y), OutputScalar(x), OutputScalar(y) };

      for (int m = 0; m < 3; ++m) {
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
          for (ordinal_type a2 = 0; a2 <= polyOrder; ++a2) {
            for (ordinal_type a1 = 0; a1 <= polyOrder - a2; ++a1) {
              const ordinal_type a0 = polyOrder - a1 - a2;
              output.access(fieldOrd, pointOrdinal) = evalBernsteinTriValue<OutputScalar>(pow_lambda, polyOrder, a0, a1, a2);
              ++fieldOrd;
            }
          }
          break;
        }
        case OPERATOR_GRAD:
        case OPERATOR_D1:
        {
          ordinal_type fieldOrd = 0;
          for (ordinal_type a2 = 0; a2 <= polyOrder; ++a2) {
            for (ordinal_type a1 = 0; a1 <= polyOrder - a2; ++a1) {
              const ordinal_type a0 = polyOrder - a1 - a2;
              output.access(fieldOrd, pointOrdinal, 0) = evalBernsteinTriDeriv<OutputScalar>(pow_lambda, polyOrder, a0, a1, a2, 1, 0);
              output.access(fieldOrd, pointOrdinal, 1) = evalBernsteinTriDeriv<OutputScalar>(pow_lambda, polyOrder, a0, a1, a2, 0, 1);
              ++fieldOrd;
            }
          }
          break;
        }
        case OPERATOR_CURL:
        {
          ordinal_type fieldOrd = 0;
          for (ordinal_type a2 = 0; a2 <= polyOrder; ++a2) {
            for (ordinal_type a1 = 0; a1 <= polyOrder - a2; ++a1) {
              const ordinal_type a0 = polyOrder - a1 - a2;
              output.access(fieldOrd, pointOrdinal, 0) =  evalBernsteinTriDeriv<OutputScalar>(pow_lambda, polyOrder, a0, a1, a2, 0, 1);
              output.access(fieldOrd, pointOrdinal, 1) = -evalBernsteinTriDeriv<OutputScalar>(pow_lambda, polyOrder, a0, a1, a2, 1, 0);
              ++fieldOrd;
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
          for (ordinal_type a2 = 0; a2 <= polyOrder; ++a2) {
            for (ordinal_type a1 = 0; a1 <= polyOrder - a2; ++a1) {
              const ordinal_type a0 = polyOrder - a1 - a2;
              for (ordinal_type ky = 0; ky <= k; ++ky) {
                const ordinal_type kx = k - ky;
                output.access(fieldOrd, pointOrdinal, ky) = evalBernsteinTriDeriv<OutputScalar>(pow_lambda, polyOrder, a0, a1, a2, kx, ky);
              }
              ++fieldOrd;
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
          INTREPID2_TEST_FOR_ABORT(true, ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_TRI) Unsupported operator type");
      }
    }
  } // namespace Impl

  /** \class  Intrepid2::Bernstein_HGRAD_TRI_Functor
      \brief  Functor for computing values for the BernsteinBasis_HGRAD_TRI and BernsteinBasis_HVOL_TRI classes.
   */
  template<class DeviceType, class OutputScalar, class PointScalar,
           class OutputFieldType, class InputPointsType>
  struct Bernstein_HGRAD_TRI_Functor
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

    Bernstein_HGRAD_TRI_Functor(EOperator opType, OutputFieldType output, InputPointsType inputPoints,
                                ordinal_type polyOrder)
    : opType_(opType), output_(output), inputPoints_(inputPoints),
      polyOrder_(polyOrder),
      fad_size_output_(getScalarDimensionForView(output))
    {
      numFields_ = output.extent_int(0);
      numPoints_ = output.extent_int(1);
      INTREPID2_TEST_FOR_EXCEPTION(numPoints_ != inputPoints.extent_int(0), std::invalid_argument, "point counts need to match!");
      INTREPID2_TEST_FOR_EXCEPTION(numFields_ != (polyOrder_ + 1) * (polyOrder_ + 2) / 2, std::invalid_argument, "output field size does not match basis cardinality");
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const TeamMember & teamMember ) const
    {
      auto pointOrdinal = teamMember.league_rank();
      OutputScratchView2D pow_lambda;
      if (fad_size_output_ > 0) {
        pow_lambda = OutputScratchView2D(teamMember.team_shmem(), 3, polyOrder_ + 1, fad_size_output_);
      } else {
        pow_lambda = OutputScratchView2D(teamMember.team_shmem(), 3, polyOrder_ + 1);
      }

      Impl::evalBernsteinTriPoint<OutputScalar, PointScalar>(opType_, output_, inputPoints_, pointOrdinal, polyOrder_, pow_lambda);
    }

    size_t team_shmem_size (int /*team_size*/) const
    {
      if (fad_size_output_ > 0)
        return OutputScratchView2D::shmem_size(3, polyOrder_ + 1, fad_size_output_);
      else
        return OutputScratchView2D::shmem_size(3, polyOrder_ + 1);
    }
  };

  /** \class  Intrepid2::BernsteinBasis_HGRAD_TRI
      \brief  Basis defining the bivariate Bernstein polynomial basis of degree p on the reference triangle,
              a polynomial subspace of H(grad).

              On shards::Triangle<3> with vertices (0,0), (1,0), (0,1), the barycentric coordinates are
                lambda_0 = 1 - x - y,  lambda_1 = x,  lambda_2 = y,
              and the basis functions are
                B_{(a0,a1,a2)}^p(x,y) = (p! / (a0! a1! a2!)) lambda_0^{a0} lambda_1^{a1} lambda_2^{a2}
              enumerated in lattice order (a2 = 0..p, a1 = 0..p-a2, a0 = p - a1 - a2).
  */
  template<typename DeviceType,
           typename OutputScalar = double,
           typename PointScalar  = double,
           bool defineVertexFunctions = true>
  class BernsteinBasis_HGRAD_TRI
  : public Basis<DeviceType,OutputScalar,PointScalar>
  {
  public:
    using BasisBase = Basis<DeviceType,OutputScalar,PointScalar>;
    using HostBasis = BernsteinBasis_HGRAD_TRI<typename Kokkos::HostSpace::device_type,OutputScalar,PointScalar,defineVertexFunctions>;

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

  public:
    /** \brief  Constructor.
        \param [in] polyOrder - the polynomial order of the basis.
        \param [in] pointType - point type for DoF coordinates (POINTTYPE_DEFAULT or POINTTYPE_EQUISPACED).
     */
    BernsteinBasis_HGRAD_TRI(ordinal_type polyOrder, const EPointType pointType = POINTTYPE_DEFAULT)
    : polyOrder_(polyOrder),
      pointType_(pointType)
    {
      INTREPID2_TEST_FOR_EXCEPTION(defineVertexFunctions && polyOrder < 1, std::invalid_argument,
                                   ">>> ERROR (Intrepid2::BernsteinBasis_HGRAD_TRI): polyOrder must be >= 1 when defineVertexFunctions is true.");
      INTREPID2_TEST_FOR_EXCEPTION(!defineVertexFunctions && polyOrder < 0, std::invalid_argument,
                                   ">>> ERROR (Intrepid2::BernsteinBasis_HGRAD_TRI): polyOrder must be >= 0.");
      INTREPID2_TEST_FOR_EXCEPTION(pointType != POINTTYPE_DEFAULT && pointType != POINTTYPE_EQUISPACED,
                                   std::invalid_argument,
                                   ">>> ERROR (Intrepid2::BernsteinBasis_HGRAD_TRI): PointType must be POINTTYPE_DEFAULT or POINTTYPE_EQUISPACED.");

      this->basisCardinality_     = ((polyOrder + 2) * (polyOrder + 1)) / 2;
      this->basisDegree_          = polyOrder;
      this->basisCellTopologyKey_ = shards::Triangle<3>::key;
      this->basisType_            = BASIS_FEM_DEFAULT;
      this->basisCoordinates_     = COORDINATES_CARTESIAN;
      this->functionSpace_        = FUNCTION_SPACE_HGRAD;

      const ordinal_type card = this->basisCardinality_;

      // Populate Bezier domain points in reference triangle coordinates
      Kokkos::DynRankView<typename ScalarViewType::value_type, typename DeviceType::execution_space::array_layout, Kokkos::HostSpace>
        dofCoordsHost("BernsteinBasis::HGRAD::TRI::dofCoords", card, 2);

      if (polyOrder == 0) {
        dofCoordsHost(0, 0) = 1.0 / 3.0;
        dofCoordsHost(0, 1) = 1.0 / 3.0;
      } else {
        ordinal_type f = 0;
        for (ordinal_type a2 = 0; a2 <= polyOrder; ++a2) {
          for (ordinal_type a1 = 0; a1 <= polyOrder - a2; ++a1) {
            dofCoordsHost(f, 0) = static_cast<double>(a1) / static_cast<double>(polyOrder);
            dofCoordsHost(f, 1) = static_cast<double>(a2) / static_cast<double>(polyOrder);
            ++f;
          }
        }
      }
      this->dofCoords_ = Kokkos::create_mirror_view(typename DeviceType::memory_space(), dofCoordsHost);
      Kokkos::deep_copy(this->dofCoords_, dofCoordsHost);

      // Initialize subcell DoF tags aligned with Shards Triangle<3> subcell orientations:
      // Edge 0: 0->1, Edge 1: 1->2, Edge 2: 2->0
      {
        const ordinal_type tagSize  = 4;
        const ordinal_type posScDim = 0;
        const ordinal_type posScOrd = 1;
        const ordinal_type posDfOrd = 2;

        OrdinalTypeArray1DHost tagView("tag view", card * tagSize);

        if (defineVertexFunctions && polyOrder >= 1) {
          const ordinal_type numEdgeDof = polyOrder - 1;
          const ordinal_type numFaceDof = (polyOrder > 2) ? ((polyOrder - 1) * (polyOrder - 2)) / 2 : 0;

          ordinal_type f = 0;
          ordinal_type faceLocalDof = 0;
          for (ordinal_type a2 = 0; a2 <= polyOrder; ++a2) {
            for (ordinal_type a1 = 0; a1 <= polyOrder - a2; ++a1) {
              const ordinal_type a0 = polyOrder - a1 - a2;

              if (a0 == polyOrder) {
                // Vertex 0: (0,0)
                tagView(f * tagSize + 0) = 0;
                tagView(f * tagSize + 1) = 0;
                tagView(f * tagSize + 2) = 0;
                tagView(f * tagSize + 3) = 1;
              } else if (a1 == polyOrder) {
                // Vertex 1: (1,0)
                tagView(f * tagSize + 0) = 0;
                tagView(f * tagSize + 1) = 1;
                tagView(f * tagSize + 2) = 0;
                tagView(f * tagSize + 3) = 1;
              } else if (a2 == polyOrder) {
                // Vertex 2: (0,1)
                tagView(f * tagSize + 0) = 0;
                tagView(f * tagSize + 1) = 2;
                tagView(f * tagSize + 2) = 0;
                tagView(f * tagSize + 3) = 1;
              } else if (a2 == 0) {
                // Edge 0 (0 -> 1): local DoF increases with a1 (1 .. p-1)
                tagView(f * tagSize + 0) = 1;
                tagView(f * tagSize + 1) = 0;
                tagView(f * tagSize + 2) = a1 - 1;
                tagView(f * tagSize + 3) = numEdgeDof;
              } else if (a0 == 0) {
                // Edge 1 (1 -> 2): local DoF increases with a2 (1 .. p-1)
                tagView(f * tagSize + 0) = 1;
                tagView(f * tagSize + 1) = 1;
                tagView(f * tagSize + 2) = a2 - 1;
                tagView(f * tagSize + 3) = numEdgeDof;
              } else if (a1 == 0) {
                // Edge 2 (2 -> 0): local DoF increases with a0 (1 .. p-1)
                tagView(f * tagSize + 0) = 1;
                tagView(f * tagSize + 1) = 2;
                tagView(f * tagSize + 2) = a0 - 1;
                tagView(f * tagSize + 3) = numEdgeDof;
              } else {
                // Face 0 interior (a0 > 0, a1 > 0, a2 > 0)
                tagView(f * tagSize + 0) = 2;
                tagView(f * tagSize + 1) = 0;
                tagView(f * tagSize + 2) = faceLocalDof++;
                tagView(f * tagSize + 3) = numFaceDof;
              }
              ++f;
            }
          }
          INTREPID2_TEST_FOR_EXCEPTION(f != card || faceLocalDof != numFaceDof, std::logic_error,
                                       ">>> ERROR (Intrepid2::BernsteinBasis_HGRAD_TRI): internal tag enumeration error.");
        } else {
          for (ordinal_type i = 0; i < card; ++i) {
            tagView(i * tagSize + 0) = 2;
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
      return "Intrepid2_BernsteinBasis_HGRAD_TRI";
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

      using FunctorType = Bernstein_HGRAD_TRI_Functor<DeviceType, OutputScalar, PointScalar, OutputViewType, PointViewType>;
      FunctorType functor(operatorType, outputValues, inputPoints, polyOrder_);

      const int outputVectorSize = getVectorSizeForHierarchicalParallelism<OutputScalar>();
      const int pointVectorSize  = getVectorSizeForHierarchicalParallelism<PointScalar>();
      const int vectorSize = std::max(outputVectorSize, pointVectorSize);
      const int teamSize = 1;

      auto policy = Kokkos::TeamPolicy<ExecutionSpace>(space, numPoints, teamSize, vectorSize);
      Kokkos::parallel_for("Bernstein_HGRAD_TRI_Functor", policy, functor);
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
        perThreadSpaceSize = ScratchView2D::shmem_size(3, polyOrder_ + 1, fadSize);
      else
        perThreadSpaceSize = ScratchView2D::shmem_size(3, polyOrder_ + 1);
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
        ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_TRI::getValues) Selecting subsets of basis functions is not supported.");

      using ScratchView2D = Kokkos::View<OutputScalar**, typename ExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      const size_t fadSize = getScalarDimensionForView(outputValues);
      ScratchView2D pow_lambda = (fadSize > 0)
        ? ScratchView2D(team_member.thread_scratch(threadScratchLevel), 3, this->polyOrder_ + 1, fadSize)
        : ScratchView2D(team_member.thread_scratch(threadScratchLevel), 3, this->polyOrder_ + 1);

      const int numPoints = inputPoints.extent_int(0);
      const ordinal_type polyOrder = this->polyOrder_;

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, numPoints), [=](const int pt) {
        auto pow_local = pow_lambda;
        Impl::evalBernsteinTriPoint<OutputScalar, PointScalar>(operatorType, outputValues, inputPoints, pt, polyOrder, pow_local);
      });
    }

    virtual void getDofCoords( ScalarViewType dofCoords ) const override
    {
#ifdef HAVE_INTREPID2_DEBUG
      INTREPID2_TEST_FOR_EXCEPTION( rank(dofCoords) != 2, std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_TRI::getDofCoords) rank = 2 required for dofCoords array");
      INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoords.extent(0)) != this->getCardinality(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_TRI::getDofCoords) mismatch in number of dof and 0th dimension of dofCoords array");
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.extent(1) != this->getBaseCellTopology().getDimension(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_TRI::getDofCoords) incorrect reference cell (1st) dimension in dofCoords array");
#endif
      Kokkos::deep_copy(dofCoords, this->dofCoords_);
    }

    BasisPtr<DeviceType,OutputScalar,PointScalar>
    getSubCellRefBasis(const ordinal_type subCellDim, const ordinal_type subCellOrd) const override
    {
      if (subCellDim == 1 && subCellOrd >= 0 && subCellOrd < 3) {
        return Teuchos::rcp(new BernsteinBasis_HGRAD_LINE<DeviceType,OutputScalar,PointScalar>(this->basisDegree_, pointType_));
      }
      INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Input parameters out of bounds");
    }

    virtual HostBasisPtr<OutputScalar, PointScalar>
    getHostBasis() const override {
      return Teuchos::rcp( new HostBasis(polyOrder_, pointType_) );
    }
  };
} // end namespace Intrepid2

#endif /* Intrepid2_BernsteinBasis_HGRAD_TRI_h */

