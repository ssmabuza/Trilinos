// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_BernsteinBasis_HGRAD_HEX.hpp
    \brief  Tensor-product H(grad) Bernstein basis on the reference hexahedron [-1,1]^3.
 */

#ifndef Intrepid2_BernsteinBasis_HGRAD_HEX_h
#define Intrepid2_BernsteinBasis_HGRAD_HEX_h

#include "Intrepid2_DerivedBasis_HGRAD_HEX.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_LINE.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_QUAD.hpp"

namespace Intrepid2
{
  /** \class  Intrepid2::BernsteinBasis_HGRAD_HEX
      \brief  Tensor-product Bernstein basis of degree (p_x, p_y, p_z) in H(grad) on shards::Hexahedron<8> ([-1,1]^3).
   */
  template<typename DeviceType,
           typename OutputScalar = double,
           typename PointScalar  = double,
           bool defineVertexFunctions = true>
  class BernsteinBasis_HGRAD_HEX
  : public Basis_Derived_HGRAD_HEX<BernsteinBasis_HGRAD_LINE<DeviceType, OutputScalar, PointScalar, defineVertexFunctions>>
  {
  public:
    using LineBasis   = BernsteinBasis_HGRAD_LINE<DeviceType, OutputScalar, PointScalar, defineVertexFunctions>;
    using QuadBasis   = BernsteinBasis_HGRAD_QUAD<DeviceType, OutputScalar, PointScalar, defineVertexFunctions>;
    using Superclass  = Basis_Derived_HGRAD_HEX<LineBasis>;
    using BasisBase   = typename Superclass::BasisBase;
    using TensorBasis = typename Superclass::TensorBasis;
    using HostBasis   = BernsteinBasis_HGRAD_HEX<typename Kokkos::HostSpace::device_type, OutputScalar, PointScalar, defineVertexFunctions>;

    using typename Superclass::ExecutionSpace;
    using typename Superclass::OutputValueType;
    using typename Superclass::PointValueType;
    using typename Superclass::OutputViewType;
    using typename Superclass::PointViewType;
    using typename Superclass::ScalarViewType;

    /** \brief Anisotropic constructor.
     */
    BernsteinBasis_HGRAD_HEX(int polyOrder_x, int polyOrder_y, int polyOrder_z, const EPointType pointType = POINTTYPE_DEFAULT)
    : Superclass(polyOrder_x, polyOrder_y, polyOrder_z, pointType)
    {}

    /** \brief Isotropic constructor.
     */
    BernsteinBasis_HGRAD_HEX(int polyOrder, const EPointType pointType = POINTTYPE_DEFAULT)
    : Superclass(polyOrder, polyOrder, polyOrder, pointType)
    {}

    using Superclass::getValues;

    Teuchos::RCP<BasisBase>
    getSubCellRefBasis(const ordinal_type subCellDim, const ordinal_type subCellOrd) const override
    {
      if (subCellDim == 1) {
        switch (subCellOrd) {
          case 0:
          case 2:
          case 4:
          case 6:
            return Teuchos::rcp(new LineBasis(this->order_x_, this->pointType_));
          case 1:
          case 3:
          case 5:
          case 7:
            return Teuchos::rcp(new LineBasis(this->order_y_, this->pointType_));
          case 8:
          case 9:
          case 10:
          case 11:
            return Teuchos::rcp(new LineBasis(this->order_z_, this->pointType_));
        }
      } else if (subCellDim == 2) {
        switch (subCellOrd) {
          case 0:
            return Teuchos::rcp(new QuadBasis(this->order_x_, this->order_z_, this->pointType_));
          case 1:
            return Teuchos::rcp(new QuadBasis(this->order_y_, this->order_z_, this->pointType_));
          case 2:
            return Teuchos::rcp(new QuadBasis(this->order_x_, this->order_z_, this->pointType_));
          case 3:
            return Teuchos::rcp(new QuadBasis(this->order_z_, this->order_y_, this->pointType_));
          case 4:
            return Teuchos::rcp(new QuadBasis(this->order_y_, this->order_x_, this->pointType_));
          case 5:
            return Teuchos::rcp(new QuadBasis(this->order_x_, this->order_y_, this->pointType_));
        }
      }
      INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Input parameters out of bounds");
    }

    virtual HostBasisPtr<OutputValueType, PointValueType>
    getHostBasis() const override
    {
      return Teuchos::rcp(new HostBasis(this->order_x_, this->order_y_, this->order_z_, this->pointType_));
    }
  };
} // end namespace Intrepid2

#endif /* Intrepid2_BernsteinBasis_HGRAD_HEX_h */

