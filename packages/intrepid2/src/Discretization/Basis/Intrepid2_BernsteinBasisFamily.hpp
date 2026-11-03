// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_BernsteinBasisFamily.hpp
    \brief  Stateless factory classes for Bernstein basis families across simplicial
            (Line, Triangle, Tetrahedron) and tensor-product (Quadrilateral, Hexahedron,
            Wedge, and d-dimensional Hypercube) topologies.
 */

#ifndef Intrepid2_BernsteinBasisFamily_h
#define Intrepid2_BernsteinBasisFamily_h

#include "Intrepid2_DerivedBasisFamily.hpp"

#include "Intrepid2_BernsteinBasis_HGRAD_LINE.hpp"
#include "Intrepid2_BernsteinBasis_HVOL_LINE.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_TRI.hpp"
#include "Intrepid2_BernsteinBasis_HVOL_TRI.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_TET.hpp"
#include "Intrepid2_BernsteinBasis_HVOL_TET.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_QUAD.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_HEX.hpp"
#include "Intrepid2_BernsteinBasis_HGRAD_WEDGE.hpp"

namespace Intrepid2
{
  /** \brief Placeholder basis class for simplicial H(curl)/H(div) slots in DerivedBasisFamily
             so factory functions (getTriangleBasis, getTetrahedronBasis, getWedgeBasis, getBasis)
             compile cleanly and throw a clear runtime exception if an unimplemented space is requested.
   */
  template<typename DeviceType, typename OutputScalar, typename PointScalar>
  class UnimplementedBernsteinBasis
  : public Basis<DeviceType, OutputScalar, PointScalar>
  {
  public:
    using BasisBase = Basis<DeviceType, OutputScalar, PointScalar>;
    using HostBasis = UnimplementedBernsteinBasis<typename Kokkos::HostSpace::device_type, OutputScalar, PointScalar>;

    UnimplementedBernsteinBasis(int /*order*/, EPointType /*pointType*/ = POINTTYPE_DEFAULT)
    {
      INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument,
        ">>> ERROR (Intrepid2::BernsteinBasisFamily): Requested simplicial H(curl)/H(div) Bernstein basis is not implemented.");
    }

    UnimplementedBernsteinBasis(int /*order1*/, int /*order2*/, EPointType /*pointType*/ = POINTTYPE_DEFAULT)
    {
      INTREPID2_TEST_FOR_EXCEPTION(true, std::invalid_argument,
        ">>> ERROR (Intrepid2::BernsteinBasisFamily): Requested wedge H(curl)/H(div) Bernstein basis is not implemented.");
    }
  };

  /** \class Intrepid2::BernsteinTriangleBasisFamily
      \brief Simplicial Bernstein basis family on shards::Triangle<3>.
   */
  template<typename DeviceType,
           typename OutputScalar = double,
           typename PointScalar  = double,
           bool defineVertexFunctions = true>
  class BernsteinTriangleBasisFamily
  {
  public:
    using HGRAD = BernsteinBasis_HGRAD_TRI<DeviceType, OutputScalar, PointScalar, defineVertexFunctions>;
    using HCURL = UnimplementedBernsteinBasis<DeviceType, OutputScalar, PointScalar>;
    using HDIV  = UnimplementedBernsteinBasis<DeviceType, OutputScalar, PointScalar>;
    using HVOL  = BernsteinBasis_HVOL_TRI<DeviceType, OutputScalar, PointScalar>;
  };

  /** \class Intrepid2::BernsteinTetrahedronBasisFamily
      \brief Simplicial Bernstein basis family on shards::Tetrahedron<4>.
   */
  template<typename DeviceType,
           typename OutputScalar = double,
           typename PointScalar  = double,
           bool defineVertexFunctions = true>
  class BernsteinTetrahedronBasisFamily
  {
  public:
    using HGRAD = BernsteinBasis_HGRAD_TET<DeviceType, OutputScalar, PointScalar, defineVertexFunctions>;
    using HCURL = UnimplementedBernsteinBasis<DeviceType, OutputScalar, PointScalar>;
    using HDIV  = UnimplementedBernsteinBasis<DeviceType, OutputScalar, PointScalar>;
    using HVOL  = BernsteinBasis_HVOL_TET<DeviceType, OutputScalar, PointScalar>;
  };

  /** \class Intrepid2::BernsteinBasisFamily
      \brief Continuous/conforming Bernstein basis family for simplicial (LINE, TRI, TET) and
             tensor-product (QUAD, HEX, WEDGE, and hypercube) topologies.

             Boundary-supported Bernstein basis functions are associated with their corresponding
             topological subcells (vertices, edges, faces, interior), enabling continuous Galerkin
             assembly and OrientationTools compatibility. Tensor-product H(curl), H(div), and H(vol)
             spaces on QUAD and HEX (and H(vol) on WEDGE) are automatically derived via DerivedBasisFamily.
   */
  template<typename DeviceType,
           typename OutputScalar = double,
           typename PointScalar  = double>
  class BernsteinBasisFamily
  : public DerivedBasisFamily<
      BernsteinBasis_HGRAD_LINE<DeviceType, OutputScalar, PointScalar, true>,
      BernsteinBasis_HVOL_LINE<DeviceType, OutputScalar, PointScalar>,
      BernsteinTriangleBasisFamily<DeviceType, OutputScalar, PointScalar, true>,
      BernsteinTetrahedronBasisFamily<DeviceType, OutputScalar, PointScalar, true>>
  {
    using BaseFamily = DerivedBasisFamily<
      BernsteinBasis_HGRAD_LINE<DeviceType, OutputScalar, PointScalar, true>,
      BernsteinBasis_HVOL_LINE<DeviceType, OutputScalar, PointScalar>,
      BernsteinTriangleBasisFamily<DeviceType, OutputScalar, PointScalar, true>,
      BernsteinTetrahedronBasisFamily<DeviceType, OutputScalar, PointScalar, true>>;
  public:
    using BasisType   = typename BaseFamily::Basis;
    using HGRAD_QUAD  = BernsteinBasis_HGRAD_QUAD<DeviceType, OutputScalar, PointScalar, true>;
    using HGRAD_HEX   = BernsteinBasis_HGRAD_HEX<DeviceType, OutputScalar, PointScalar, true>;
    using HGRAD_WEDGE = BernsteinBasis_HGRAD_WEDGE<DeviceType, OutputScalar, PointScalar, true>;
  };

  /** \class Intrepid2::DGBernsteinBasisFamily
      \brief Discontinuous Galerkin Bernstein basis family in which all basis functions are
             associated with element interiors.
   */
  template<typename DeviceType,
           typename OutputScalar = double,
           typename PointScalar  = double>
  class DGBernsteinBasisFamily
  : public DerivedBasisFamily<
      BernsteinBasis_HGRAD_LINE<DeviceType, OutputScalar, PointScalar, false>,
      BernsteinBasis_HVOL_LINE<DeviceType, OutputScalar, PointScalar>,
      BernsteinTriangleBasisFamily<DeviceType, OutputScalar, PointScalar, false>,
      BernsteinTetrahedronBasisFamily<DeviceType, OutputScalar, PointScalar, false>>
  {
    using BaseFamily = DerivedBasisFamily<
      BernsteinBasis_HGRAD_LINE<DeviceType, OutputScalar, PointScalar, false>,
      BernsteinBasis_HVOL_LINE<DeviceType, OutputScalar, PointScalar>,
      BernsteinTriangleBasisFamily<DeviceType, OutputScalar, PointScalar, false>,
      BernsteinTetrahedronBasisFamily<DeviceType, OutputScalar, PointScalar, false>>;
  public:
    using BasisType   = typename BaseFamily::Basis;
    using HGRAD_QUAD  = BernsteinBasis_HGRAD_QUAD<DeviceType, OutputScalar, PointScalar, false>;
    using HGRAD_HEX   = BernsteinBasis_HGRAD_HEX<DeviceType, OutputScalar, PointScalar, false>;
    using HGRAD_WEDGE = BernsteinBasis_HGRAD_WEDGE<DeviceType, OutputScalar, PointScalar, false>;
  };

  // Standard Intrepid2-style convenience aliases
  template<typename DeviceType, typename OutputScalar = double, typename PointScalar = double>
  using Basis_HGRAD_LINE_Cn_FEM_BERNSTEIN = BernsteinBasis_HGRAD_LINE<DeviceType, OutputScalar, PointScalar, true>;

  template<typename DeviceType, typename OutputScalar = double, typename PointScalar = double>
  using Basis_HGRAD_TRI_Cn_FEM_BERNSTEIN = BernsteinBasis_HGRAD_TRI<DeviceType, OutputScalar, PointScalar, true>;

  template<typename DeviceType, typename OutputScalar = double, typename PointScalar = double>
  using Basis_HGRAD_TET_Cn_FEM_BERNSTEIN = BernsteinBasis_HGRAD_TET<DeviceType, OutputScalar, PointScalar, true>;

  template<typename DeviceType, typename OutputScalar = double, typename PointScalar = double>
  using Basis_HGRAD_QUAD_Cn_FEM_BERNSTEIN = BernsteinBasis_HGRAD_QUAD<DeviceType, OutputScalar, PointScalar, true>;

  template<typename DeviceType, typename OutputScalar = double, typename PointScalar = double>
  using Basis_HGRAD_HEX_Cn_FEM_BERNSTEIN = BernsteinBasis_HGRAD_HEX<DeviceType, OutputScalar, PointScalar, true>;

  template<typename DeviceType, typename OutputScalar = double, typename PointScalar = double>
  using Basis_HGRAD_WEDGE_Cn_FEM_BERNSTEIN = BernsteinBasis_HGRAD_WEDGE<DeviceType, OutputScalar, PointScalar, true>;

  template<typename DeviceType, typename OutputScalar = double, typename PointScalar = double>
  using Basis_HVOL_LINE_Cn_FEM_BERNSTEIN = BernsteinBasis_HVOL_LINE<DeviceType, OutputScalar, PointScalar>;

  template<typename DeviceType, typename OutputScalar = double, typename PointScalar = double>
  using Basis_HVOL_TRI_Cn_FEM_BERNSTEIN = BernsteinBasis_HVOL_TRI<DeviceType, OutputScalar, PointScalar>;

  template<typename DeviceType, typename OutputScalar = double, typename PointScalar = double>
  using Basis_HVOL_TET_Cn_FEM_BERNSTEIN = BernsteinBasis_HVOL_TET<DeviceType, OutputScalar, PointScalar>;

  template<typename DeviceType, typename OutputScalar = double, typename PointScalar = double>
  using Basis_HVOL_QUAD_Cn_FEM_BERNSTEIN = typename BernsteinBasisFamily<DeviceType, OutputScalar, PointScalar>::HVOL_QUAD;

  template<typename DeviceType, typename OutputScalar = double, typename PointScalar = double>
  using Basis_HVOL_HEX_Cn_FEM_BERNSTEIN = typename BernsteinBasisFamily<DeviceType, OutputScalar, PointScalar>::HVOL_HEX;

  template<typename DeviceType, typename OutputScalar = double, typename PointScalar = double>
  using Basis_HVOL_WEDGE_Cn_FEM_BERNSTEIN = typename BernsteinBasisFamily<DeviceType, OutputScalar, PointScalar>::HVOL_WEDGE;

} // end namespace Intrepid2

#endif /* Intrepid2_BernsteinBasisFamily_h */

