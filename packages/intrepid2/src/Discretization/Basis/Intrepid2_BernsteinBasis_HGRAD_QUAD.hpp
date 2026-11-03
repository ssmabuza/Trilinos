// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_BernsteinBasis_HGRAD_QUAD.hpp
    \brief  Tensor-product H(grad) Bernstein basis on the reference quadrilateral [-1,1]^2.
 */

#ifndef Intrepid2_BernsteinBasis_HGRAD_QUAD_h
#define Intrepid2_BernsteinBasis_HGRAD_QUAD_h

#include "Intrepid2_DerivedBasis_HGRAD_QUAD.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_LINE.hpp"

namespace Intrepid2
{
  /** \class  Intrepid2::BernsteinBasis_HGRAD_QUAD
      \brief  Tensor-product Bernstein basis of degree (p_x, p_y) in H(grad) on shards::Quadrilateral<4> ([-1,1]^2).
   */
  template<typename DeviceType,
           typename OutputScalar = double,
           typename PointScalar  = double,
           bool defineVertexFunctions = true>
  class BernsteinBasis_HGRAD_QUAD
  : public Basis_Derived_HGRAD_QUAD<BernsteinBasis_HGRAD_LINE<DeviceType, OutputScalar, PointScalar, defineVertexFunctions>>
  {
  public:
    using LineBasis   = BernsteinBasis_HGRAD_LINE<DeviceType, OutputScalar, PointScalar, defineVertexFunctions>;
    using Superclass  = Basis_Derived_HGRAD_QUAD<LineBasis>;
    using BasisBase   = typename Superclass::BasisBase;
    using TensorBasis = typename Superclass::TensorBasis;
    using HostBasis   = BernsteinBasis_HGRAD_QUAD<typename Kokkos::HostSpace::device_type, OutputScalar, PointScalar, defineVertexFunctions>;

    using typename Superclass::ExecutionSpace;
    using typename Superclass::OutputValueType;
    using typename Superclass::PointValueType;
    using typename Superclass::OutputViewType;
    using typename Superclass::PointViewType;
    using typename Superclass::ScalarViewType;

    /** \brief Anisotropic constructor.
        \param [in] polyOrder_x - polynomial order in x.
        \param [in] polyOrder_y - polynomial order in y.
        \param [in] pointType   - DoF coordinate lattice type (POINTTYPE_DEFAULT or POINTTYPE_EQUISPACED).
     */
    BernsteinBasis_HGRAD_QUAD(int polyOrder_x, int polyOrder_y, const EPointType pointType = POINTTYPE_DEFAULT)
    : Superclass(polyOrder_x, polyOrder_y, pointType)
    {}

    /** \brief Isotropic constructor.
        \param [in] polyOrder - polynomial order in both x and y.
        \param [in] pointType - DoF coordinate lattice type (POINTTYPE_DEFAULT or POINTTYPE_EQUISPACED).
     */
    BernsteinBasis_HGRAD_QUAD(int polyOrder, const EPointType pointType = POINTTYPE_DEFAULT)
    : Superclass(polyOrder, polyOrder, pointType)
    {}

    using Superclass::getValues;

    /** \brief Support 2D scalar OPERATOR_CURL = (+d/dy, -d/dx) in addition to TensorBasis operators.
     */
    virtual void getValues(OutputViewType outputValues,
                           const EOperator operatorType,
                           const PointViewType inputPoints1,
                           const PointViewType inputPoints2,
                           bool tensorPoints) const override
    {
      if (operatorType == OPERATOR_CURL)
      {
        OutputViewType comp0 = Kokkos::subview(outputValues, Kokkos::ALL(), Kokkos::ALL(), 0);
        OutputViewType comp1 = Kokkos::subview(outputValues, Kokkos::ALL(), Kokkos::ALL(), 1);

        // Component 0: +d/dy = VALUE(x) * GRAD(y)
        this->TensorBasis::getValues(comp0, inputPoints1, OPERATOR_VALUE, inputPoints2, OPERATOR_GRAD, tensorPoints,  1.0);
        // Component 1: -d/dx = -GRAD(x) * VALUE(y)
        this->TensorBasis::getValues(comp1, inputPoints1, OPERATOR_GRAD,  inputPoints2, OPERATOR_VALUE, tensorPoints, -1.0);
      }
      else
      {
        this->TensorBasis::getValues(outputValues, operatorType, inputPoints1, inputPoints2, tensorPoints);
      }
    }

    virtual OperatorTensorDecomposition getSimpleOperatorDecomposition(const EOperator &operatorType) const override
    {
      if (operatorType == OPERATOR_CURL)
      {
        std::vector< std::vector<EOperator> > ops{
          {OPERATOR_VALUE, OPERATOR_GRAD},
          {OPERATOR_GRAD,  OPERATOR_VALUE}
        };
        std::vector<double> weights{1.0, -1.0};
        return OperatorTensorDecomposition(ops, weights);
      }
      return this->TensorBasis::getSimpleOperatorDecomposition(operatorType);
    }

    Teuchos::RCP<BasisBase>
    getSubCellRefBasis(const ordinal_type subCellDim, const ordinal_type subCellOrd) const override
    {
      if (subCellDim == 1) {
        switch (subCellOrd) {
          case 0:
          case 2:
            return Teuchos::rcp(new LineBasis(this->order_x_, this->pointType_));
          case 1:
          case 3:
            return Teuchos::rcp(new LineBasis(this->order_y_, this->pointType_));
        }
      }
      INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Input parameters out of bounds");
    }

    virtual HostBasisPtr<OutputValueType, PointValueType>
    getHostBasis() const override
    {
      return Teuchos::rcp(new HostBasis(this->order_x_, this->order_y_, this->pointType_));
    }
  };
} // end namespace Intrepid2

#endif /* Intrepid2_BernsteinBasis_HGRAD_QUAD_h */

