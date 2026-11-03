// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_BernsteinBasis_HGRAD_LINE.hpp
    \brief  H(grad) Bernstein basis on the reference line.
 */

#ifndef Intrepid2_BernsteinBasis_HGRAD_LINE_h
#define Intrepid2_BernsteinBasis_HGRAD_LINE_h

#include <Kokkos_DynRankView.hpp>

#include <Intrepid2_config.h>

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_Polynomials.hpp"
#include "Intrepid2_Utils.hpp"

namespace Intrepid2
{
  /** \class  Intrepid2::Bernstein_HGRAD_LINE_Functor
      \brief  Functor for computing values for the BernsteinBasis_HGRAD_LINE class.
   */
  template<class DeviceType, class OutputScalar, class PointScalar,
           class OutputFieldType, class InputPointsType,
           bool useMinusOneToOneReferenceElement = true>
  struct Bernstein_HGRAD_LINE_Functor
  {
    using ExecutionSpace    = typename DeviceType::execution_space;
    using ScratchSpace      = typename ExecutionSpace::scratch_memory_space;
    using OutputScratchView = Kokkos::View<OutputScalar*,ScratchSpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using TeamMember = typename TeamPolicy::member_type;

    EOperator opType_;

    OutputFieldType output_;      // (F,P) or (F,P,Dk)
    InputPointsType inputPoints_; // (P,1)

    ordinal_type polyOrder_;
    ordinal_type numFields_, numPoints_;

    size_t fad_size_output_;

    Bernstein_HGRAD_LINE_Functor(EOperator opType, OutputFieldType output, InputPointsType inputPoints,
                                 ordinal_type polyOrder)
    : opType_(opType), output_(output), inputPoints_(inputPoints),
      polyOrder_(polyOrder),
      fad_size_output_(getScalarDimensionForView(output))
    {
      numFields_ = output.extent_int(0);
      numPoints_ = output.extent_int(1);
      INTREPID2_TEST_FOR_EXCEPTION(numPoints_ != inputPoints.extent_int(0), std::invalid_argument, "point counts need to match!");
      INTREPID2_TEST_FOR_EXCEPTION(numFields_ != polyOrder_ + 1, std::invalid_argument, "output field size does not match basis cardinality");
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const TeamMember & teamMember ) const
    {
      auto pointOrdinal = teamMember.league_rank();
      OutputScratchView field_values_at_point;
      if (fad_size_output_ > 0) {
        field_values_at_point = OutputScratchView(teamMember.team_shmem(), numFields_, fad_size_output_);
      }
      else {
        field_values_at_point = OutputScratchView(teamMember.team_shmem(), numFields_);
      }

      const PointScalar & x = inputPoints_(pointOrdinal, 0);

      switch (opType_)
      {
        case OPERATOR_VALUE:
        {
          if constexpr (useMinusOneToOneReferenceElement) {
            Polynomials::bernsteinValues(field_values_at_point, polyOrder_, x);
          } else {
            Polynomials::shiftedBernsteinValues(field_values_at_point, polyOrder_, x);
          }
          break;
        }
        case OPERATOR_GRAD:
        case OPERATOR_D1:
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
          const ordinal_type dn = getOperatorOrder(opType_);
          if constexpr (useMinusOneToOneReferenceElement) {
            Polynomials::bernsteinDerivativeValues(field_values_at_point, polyOrder_, x, dn);
          } else {
            Polynomials::shiftedBernsteinDerivativeValues(field_values_at_point, polyOrder_, x, dn);
          }
          break;
        }
        case OPERATOR_MAX:
        {
          for (ordinal_type f = 0; f < numFields_; ++f) {
            field_values_at_point(f) = 0.0;
          }
          break;
        }
        default:
          INTREPID2_TEST_FOR_ABORT(true, ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_LINE) Unsupported operator type");
      }

      for (ordinal_type fieldOrdinal = 0; fieldOrdinal < numFields_; ++fieldOrdinal)
      {
        output_.access(fieldOrdinal, pointOrdinal, 0) = field_values_at_point(fieldOrdinal);
      }
    }

    size_t team_shmem_size (int /*team_size*/) const
    {
      size_t shmem_size = 0;
      if (fad_size_output_ > 0)
        shmem_size += OutputScratchView::shmem_size(numFields_, fad_size_output_);
      else
        shmem_size += OutputScratchView::shmem_size(numFields_);
      return shmem_size;
    }
  };

  /** \class  Intrepid2::BernsteinBasis_HGRAD_LINE
      \brief  Basis defining the Bernstein polynomial basis of degree p on the reference line, a polynomial subspace of H(grad).

              On the standard Intrepid2 reference line [-1,1] (useMinusOneToOneReferenceElement = true),
              the (i)-th basis function (0 <= i <= p) is
                B_{i,p}(x) = C(p, i) ((1 - x)/2)^{p - i} ((1 + x)/2)^i.
              On [0,1] (useMinusOneToOneReferenceElement = false),
                B_{i,p}(t) = C(p, i) (1 - t)^{p - i} t^i.

              When defineVertexFunctions is true (continuous Galerkin), i = 0 and i = p are associated
              with vertices 0 and 1, and i = 1..p-1 are associated with the edge interior.
              When defineVertexFunctions is false (discontinuous Galerkin), all p+1 basis functions
              are associated with the cell interior.
  */
  template<typename DeviceType,
           typename OutputScalar = double,
           typename PointScalar  = double,
           bool defineVertexFunctions = true,
           bool useMinusOneToOneReferenceElement = true>
  class BernsteinBasis_HGRAD_LINE
  : public Basis<DeviceType,OutputScalar,PointScalar>
  {
  public:
    using BasisBase = Basis<DeviceType,OutputScalar,PointScalar>;
    using HostBasis = BernsteinBasis_HGRAD_LINE<typename Kokkos::HostSpace::device_type,OutputScalar,PointScalar,defineVertexFunctions,useMinusOneToOneReferenceElement>;

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
    BernsteinBasis_HGRAD_LINE(ordinal_type polyOrder, const EPointType pointType = POINTTYPE_DEFAULT)
    : polyOrder_(polyOrder),
      pointType_(pointType)
    {
      INTREPID2_TEST_FOR_EXCEPTION(defineVertexFunctions && polyOrder < 1, std::invalid_argument,
                                   ">>> ERROR (Intrepid2::BernsteinBasis_HGRAD_LINE): polyOrder must be >= 1 when defineVertexFunctions is true.");
      INTREPID2_TEST_FOR_EXCEPTION(!defineVertexFunctions && polyOrder < 0, std::invalid_argument,
                                   ">>> ERROR (Intrepid2::BernsteinBasis_HGRAD_LINE): polyOrder must be >= 0.");
      INTREPID2_TEST_FOR_EXCEPTION(pointType != POINTTYPE_DEFAULT && pointType != POINTTYPE_EQUISPACED,
                                   std::invalid_argument,
                                   ">>> ERROR (Intrepid2::BernsteinBasis_HGRAD_LINE): PointType must be POINTTYPE_DEFAULT or POINTTYPE_EQUISPACED.");

      this->basisCardinality_     = polyOrder + 1;
      this->basisDegree_          = polyOrder;
      this->basisCellTopologyKey_ = shards::Line<2>::key;
      this->basisType_            = BASIS_FEM_DEFAULT;
      this->basisCoordinates_     = COORDINATES_CARTESIAN;
      this->functionSpace_        = FUNCTION_SPACE_HGRAD;

      const ordinal_type card = this->basisCardinality_;

      // Populate Bezier domain points (equispaced nodes on the reference line)
      Kokkos::DynRankView<typename ScalarViewType::value_type, typename DeviceType::execution_space::array_layout, Kokkos::HostSpace>
        dofCoordsHost("BernsteinBasis::HGRAD::LINE::dofCoords", card, 1);

      if (polyOrder == 0) {
        dofCoordsHost(0, 0) = useMinusOneToOneReferenceElement ? 0.0 : 0.5;
      } else {
        for (ordinal_type i = 0; i <= polyOrder; ++i) {
          const double t = static_cast<double>(i) / static_cast<double>(polyOrder);
          dofCoordsHost(i, 0) = useMinusOneToOneReferenceElement ? (-1.0 + 2.0 * t) : t;
        }
      }
      this->dofCoords_ = Kokkos::create_mirror_view(typename DeviceType::memory_space(), dofCoordsHost);
      Kokkos::deep_copy(this->dofCoords_, dofCoordsHost);

      // Initialize subcell DoF tags in natural lattice order
      {
        const ordinal_type tagSize  = 4;
        const ordinal_type posScDim = 0;
        const ordinal_type posScOrd = 1;
        const ordinal_type posDfOrd = 2;

        OrdinalTypeArray1DHost tagView("tag view", card * tagSize);

        if (defineVertexFunctions && polyOrder >= 1) {
          // Left vertex: i = 0
          tagView(0 * tagSize + 0) = 0;
          tagView(0 * tagSize + 1) = 0;
          tagView(0 * tagSize + 2) = 0;
          tagView(0 * tagSize + 3) = 1;

          // Interior edge DoFs: i = 1 .. p-1
          const ordinal_type numEdgeDofs = card - 2;
          for (ordinal_type i = 0; i < numEdgeDofs; ++i) {
            const ordinal_type f = i + 1;
            tagView(f * tagSize + 0) = 1;
            tagView(f * tagSize + 1) = 0;
            tagView(f * tagSize + 2) = i;
            tagView(f * tagSize + 3) = numEdgeDofs;
          }

          // Right vertex: i = p
          const ordinal_type v1 = card - 1;
          tagView(v1 * tagSize + 0) = 0;
          tagView(v1 * tagSize + 1) = 1;
          tagView(v1 * tagSize + 2) = 0;
          tagView(v1 * tagSize + 3) = 1;
        } else {
          for (ordinal_type i = 0; i < card; ++i) {
            tagView(i * tagSize + 0) = 1;
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
      return "Intrepid2_BernsteinBasis_HGRAD_LINE";
    }

    virtual bool requireOrientation() const override {
      return false;
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

      using FunctorType = Bernstein_HGRAD_LINE_Functor<DeviceType, OutputScalar, PointScalar,
                                                       OutputViewType, PointViewType,
                                                       useMinusOneToOneReferenceElement>;

      FunctorType functor(operatorType, outputValues, inputPoints, polyOrder_);

      const int outputVectorSize = getVectorSizeForHierarchicalParallelism<OutputScalar>();
      const int pointVectorSize  = getVectorSizeForHierarchicalParallelism<PointScalar>();
      const int vectorSize = std::max(outputVectorSize, pointVectorSize);
      const int teamSize = 1;

      auto policy = Kokkos::TeamPolicy<ExecutionSpace>(space, numPoints, teamSize, vectorSize);
      Kokkos::parallel_for("Bernstein_HGRAD_LINE_Functor", policy, functor);
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
      using ScratchViewType = Kokkos::View<OutputScalar*, typename ExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      const size_t fadSize = getScalarDimensionForView(inputPoints);
      if (fadSize > 0)
        perThreadSpaceSize = ScratchViewType::shmem_size(this->basisCardinality_, fadSize);
      else
        perThreadSpaceSize = ScratchViewType::shmem_size(this->basisCardinality_);
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
        ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_LINE::getValues) Selecting subsets of basis functions is not supported.");

      using ScratchViewType = Kokkos::View<OutputScalar*, typename ExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      const size_t fadSize = getScalarDimensionForView(outputValues);
      ScratchViewType work = (fadSize > 0)
        ? ScratchViewType(team_member.thread_scratch(threadScratchLevel), this->basisCardinality_, fadSize)
        : ScratchViewType(team_member.thread_scratch(threadScratchLevel), this->basisCardinality_);

      const int numPoints = inputPoints.extent_int(0);
      const ordinal_type polyOrder = this->polyOrder_;
      const ordinal_type numFields = this->basisCardinality_;

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, numPoints), [=](const int pt) {
        const PointScalar x = inputPoints(pt, 0);
        if (operatorType == OPERATOR_VALUE) {
          if constexpr (useMinusOneToOneReferenceElement)
            Polynomials::bernsteinValues(work, polyOrder, x);
          else
            Polynomials::shiftedBernsteinValues(work, polyOrder, x);
        } else {
          const ordinal_type dn = getOperatorOrder(operatorType);
          if constexpr (useMinusOneToOneReferenceElement)
            Polynomials::bernsteinDerivativeValues(work, polyOrder, x, dn);
          else
            Polynomials::shiftedBernsteinDerivativeValues(work, polyOrder, x, dn);
        }
        for (ordinal_type f = 0; f < numFields; ++f) {
          outputValues.access(f, pt, 0) = work(f);
        }
      });
    }

    virtual void getDofCoords( ScalarViewType dofCoords ) const override
    {
#ifdef HAVE_INTREPID2_DEBUG
      INTREPID2_TEST_FOR_EXCEPTION( rank(dofCoords) != 2, std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_LINE::getDofCoords) rank = 2 required for dofCoords array");
      INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoords.extent(0)) != this->getCardinality(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_LINE::getDofCoords) mismatch in number of dof and 0th dimension of dofCoords array");
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.extent(1) != this->getBaseCellTopology().getDimension(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::BernsteinBasis_HGRAD_LINE::getDofCoords) incorrect reference cell (1st) dimension in dofCoords array");
#endif
      Kokkos::deep_copy(dofCoords, this->dofCoords_);
    }

    virtual HostBasisPtr<OutputScalar, PointScalar>
    getHostBasis() const override {
      return Teuchos::rcp( new HostBasis(polyOrder_, pointType_) );
    }
  };
} // end namespace Intrepid2

#endif /* Intrepid2_BernsteinBasis_HGRAD_LINE_h */

