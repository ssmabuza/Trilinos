// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_BernsteinBasis_HGRAD_WEDGE.hpp
    \brief  Tensor-product H(grad) Bernstein basis on the reference wedge (Triangle<3> x Line<2>).
 */

#ifndef Intrepid2_BernsteinBasis_HGRAD_WEDGE_h
#define Intrepid2_BernsteinBasis_HGRAD_WEDGE_h

#include "Intrepid2_DerivedBasis_HGRAD_WEDGE.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_LINE.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_TRI.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_QUAD.hpp"

namespace Intrepid2
{
  /** \class  Intrepid2::BernsteinBasis_HGRAD_WEDGE
      \brief  Tensor-product Bernstein basis of degree (p_xy, p_z) in H(grad) on shards::Wedge<6>.
              Constructed as the tensor product of the simplicial triangle Bernstein basis and the 1D line Bernstein basis.
   */
  template<typename DeviceType,
           typename OutputScalar = double,
           typename PointScalar  = double,
           bool defineVertexFunctions = true>
  class BernsteinBasis_HGRAD_WEDGE
  : public Basis_Derived_HGRAD_WEDGE<
      BernsteinBasis_HGRAD_TRI<DeviceType, OutputScalar, PointScalar, defineVertexFunctions>,
      BernsteinBasis_HGRAD_LINE<DeviceType, OutputScalar, PointScalar, defineVertexFunctions>>
  {
  public:
    using TriBasis    = BernsteinBasis_HGRAD_TRI<DeviceType, OutputScalar, PointScalar, defineVertexFunctions>;
    using LineBasis   = BernsteinBasis_HGRAD_LINE<DeviceType, OutputScalar, PointScalar, defineVertexFunctions>;
    using QuadBasis   = BernsteinBasis_HGRAD_QUAD<DeviceType, OutputScalar, PointScalar, defineVertexFunctions>;
    using Superclass  = Basis_Derived_HGRAD_WEDGE<TriBasis, LineBasis>;
    using BasisBase   = typename Superclass::BasisBase;
    using TensorBasis = typename Superclass::TensorBasis;
    using HostBasis   = BernsteinBasis_HGRAD_WEDGE<typename Kokkos::HostSpace::device_type, OutputScalar, PointScalar, defineVertexFunctions>;

    using typename Superclass::ExecutionSpace;
    using typename Superclass::OutputValueType;
    using typename Superclass::PointValueType;
    using typename Superclass::OutputViewType;
    using typename Superclass::PointViewType;
    using typename Superclass::ScalarViewType;

    /** \brief Anisotropic constructor.
     */
    BernsteinBasis_HGRAD_WEDGE(int polyOrder_xy, int polyOrder_z, const EPointType pointType = POINTTYPE_DEFAULT)
    : Superclass(polyOrder_xy, polyOrder_z, pointType)
    {}

    /** \brief Isotropic constructor.
     */
    BernsteinBasis_HGRAD_WEDGE(int polyOrder, const EPointType pointType = POINTTYPE_DEFAULT)
    : Superclass(polyOrder, polyOrder, pointType)
    {}

    using Superclass::getValues;

    Teuchos::RCP<BasisBase>
    getSubCellRefBasis(const ordinal_type subCellDim, const ordinal_type subCellOrd) const override
    {
      if (subCellDim == 1) {
        if (subCellOrd >= 0 && subCellOrd < 6)
          return Teuchos::rcp(new LineBasis(this->order_xy_, this->pointType_));
        else if (subCellOrd >= 6 && subCellOrd < 9)
          return Teuchos::rcp(new LineBasis(this->order_z_, this->pointType_));
      } else if (subCellDim == 2) {
        switch (subCellOrd) {
          case 0:
          case 1:
            return Teuchos::rcp(new QuadBasis(this->order_xy_, this->order_z_, this->pointType_));
          case 2:
            return Teuchos::rcp(new QuadBasis(this->order_z_, this->order_xy_, this->pointType_));
          case 3:
          case 4:
            return Teuchos::rcp(new TriBasis(this->order_xy_, this->pointType_));
          default:
            break;
        }
      }
      INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Input parameters out of bounds");
    }

    virtual HostBasisPtr<OutputValueType, PointValueType>
    getHostBasis() const override
    {
      return Teuchos::rcp(new HostBasis(this->order_xy_, this->order_z_, this->pointType_));
    }
  };
} // end namespace Intrepid2

#endif /* Intrepid2_BernsteinBasis_HGRAD_WEDGE_h */

