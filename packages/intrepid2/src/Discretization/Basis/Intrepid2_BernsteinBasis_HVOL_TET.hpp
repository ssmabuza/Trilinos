// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_BernsteinBasis_HVOL_TET.hpp
    \brief  H(vol) Bernstein basis on the reference tetrahedron.
 */

#ifndef Intrepid2_BernsteinBasis_HVOL_TET_h
#define Intrepid2_BernsteinBasis_HVOL_TET_h

#include <Kokkos_DynRankView.hpp>

#include <Intrepid2_config.h>

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_TET.hpp"
#include "Intrepid2_Polynomials.hpp"
#include "Intrepid2_Utils.hpp"

namespace Intrepid2
{
  /** \class  Intrepid2::BernsteinBasis_HVOL_TET
      \brief  Basis defining the trivariate Bernstein polynomial basis of degree p >= 0 in H(vol) on the reference tetrahedron.
              All DoFs are associated with the 3D cell interior.
  */
  template<typename DeviceType,
           typename OutputScalar = double,
           typename PointScalar  = double>
  class BernsteinBasis_HVOL_TET
  : public Basis<DeviceType,OutputScalar,PointScalar>
  {
  public:
    using BasisBase = Basis<DeviceType,OutputScalar,PointScalar>;
    using HostBasis = BernsteinBasis_HVOL_TET<typename Kokkos::HostSpace::device_type,OutputScalar,PointScalar>;

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
        \param [in] polyOrder - the polynomial order of the basis (polyOrder >= 0).
        \param [in] pointType - point type for DoF coordinates (POINTTYPE_DEFAULT or POINTTYPE_EQUISPACED).
     */
    BernsteinBasis_HVOL_TET(ordinal_type polyOrder, const EPointType pointType = POINTTYPE_DEFAULT)
    : polyOrder_(polyOrder),
      pointType_(pointType)
    {
      INTREPID2_TEST_FOR_EXCEPTION(polyOrder < 0, std::invalid_argument,
                                   ">>> ERROR (Intrepid2::BernsteinBasis_HVOL_TET): polyOrder must be >= 0.");
      INTREPID2_TEST_FOR_EXCEPTION(pointType != POINTTYPE_DEFAULT && pointType != POINTTYPE_EQUISPACED,
                                   std::invalid_argument,
                                   ">>> ERROR (Intrepid2::BernsteinBasis_HVOL_TET): PointType must be POINTTYPE_DEFAULT or POINTTYPE_EQUISPACED.");

      this->basisCardinality_     = ((polyOrder + 1) * (polyOrder + 2) * (polyOrder + 3)) / 6;
      this->basisDegree_          = polyOrder;
      this->basisCellTopologyKey_ = shards::Tetrahedron<4>::key;
      this->basisType_            = BASIS_FEM_DEFAULT;
      this->basisCoordinates_     = COORDINATES_CARTESIAN;
      this->functionSpace_        = FUNCTION_SPACE_HVOL;

      const ordinal_type card = this->basisCardinality_;

      Kokkos::DynRankView<typename ScalarViewType::value_type, typename DeviceType::execution_space::array_layout, Kokkos::HostSpace>
        dofCoordsHost("BernsteinBasis::HVOL::TET::dofCoords", card, 3);

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

      // All DoFs belong to the 3D cell interior
      {
        const ordinal_type tagSize  = 4;
        const ordinal_type posScDim = 0;
        const ordinal_type posScOrd = 1;
        const ordinal_type posDfOrd = 2;

        OrdinalTypeArray1DHost tagView("tag view", card * tagSize);

        for (ordinal_type i = 0; i < card; ++i) {
          tagView(i * tagSize + 0) = 3;
          tagView(i * tagSize + 1) = 0;
          tagView(i * tagSize + 2) = i;
          tagView(i * tagSize + 3) = card;
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
      return "Intrepid2_BernsteinBasis_HVOL_TET";
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
      Intrepid2::getValues_HVOL_Args(outputValues,
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
      Kokkos::parallel_for("Bernstein_HVOL_TET_Functor", policy, functor);
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
        ">>> ERROR: (Intrepid2::BernsteinBasis_HVOL_TET::getValues) Selecting subsets of basis functions is not supported.");

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
                                    ">>> ERROR: (Intrepid2::BernsteinBasis_HVOL_TET::getDofCoords) rank = 2 required for dofCoords array");
      INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoords.extent(0)) != this->getCardinality(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::BernsteinBasis_HVOL_TET::getDofCoords) mismatch in number of dof and 0th dimension of dofCoords array");
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.extent(1) != this->getBaseCellTopology().getDimension(), std::invalid_argument,
                                    ">>> ERROR: (Intrepid2::BernsteinBasis_HVOL_TET::getDofCoords) incorrect reference cell (1st) dimension in dofCoords array");
#endif
      Kokkos::deep_copy(dofCoords, this->dofCoords_);
    }

    virtual HostBasisPtr<OutputScalar, PointScalar>
    getHostBasis() const override {
      return Teuchos::rcp( new HostBasis(polyOrder_, pointType_) );
    }
  };
} // end namespace Intrepid2

#endif /* Intrepid2_BernsteinBasis_HVOL_TET_h */

