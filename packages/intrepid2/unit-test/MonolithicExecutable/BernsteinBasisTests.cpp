// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   BernsteinBasisTests.cpp
    \brief  Comprehensive unit tests for simplicial and tensor-product Bernstein bases in Intrepid2.
 */

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_LAPACK.hpp"

#include "Intrepid2_BernsteinBasisFamily.hpp"
#include "Intrepid2_HierarchicalBasisFamily.hpp"
#include "Intrepid2_NodalBasisFamily.hpp"
#include "Intrepid2_Polynomials.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Orientation.hpp"
#include "Intrepid2_OrientationTools.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Intrepid2_Types.hpp"
#include "Intrepid2_TestUtils.hpp"

#include "BasisEquivalenceHelpers.hpp"

#include "Kokkos_Core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace
{
  using namespace Intrepid2;

  using DeviceType     = DefaultTestDeviceType;
  using ExecutionSpace = typename DeviceType::execution_space;
  using HostSpaceType  = typename Kokkos::HostSpace::device_type;

  // ============================================================================
  // 1. 1D Bernstein Polynomial Helpers: Values, Partition of Unity, Marsden
  // ============================================================================
  TEUCHOS_UNIT_TEST( BernsteinBasis, Polynomials_1D_ValuesAndProperties )
  {
    const double tol = 10.0 * TEST_TOLERANCE_TIGHT;

    // Verify binomialCoefficient symmetry and Pascal's identity
    for (ordinal_type n = 0; n <= 20; ++n) {
      TEST_EQUALITY(Polynomials::binomialCoefficient(n, 0), 1.0);
      TEST_EQUALITY(Polynomials::binomialCoefficient(n, n), 1.0);
      for (ordinal_type k = 1; k < n; ++k) {
        const double lhs = Polynomials::binomialCoefficient(n, k);
        const double rhs = Polynomials::binomialCoefficient(n - 1, k - 1)
                         + Polynomials::binomialCoefficient(n - 1, k);
        TEST_FLOATING_EQUALITY(lhs, rhs, 1e-15);
        TEST_FLOATING_EQUALITY(lhs, Polynomials::binomialCoefficient(n, n - k), 1e-15);
      }
    }

    const std::vector<double> t_pts = {0.0, 0.05, 0.25, 1.0 / 3.0, 0.5, 0.73, 0.95, 1.0};

    for (ordinal_type n = 0; n <= 15; ++n) {
      for (double t : t_pts) {
        const double x = 2.0 * t - 1.0;

        Kokkos::View<double*, DeviceType> vals01("vals01", n + 1);
        Kokkos::View<double*, DeviceType> vals11("vals11", n + 1);

        auto policy = Kokkos::RangePolicy<ExecutionSpace>(0, 1);
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int) {
          Polynomials::shiftedBernsteinValues(vals01, n, t);
          Polynomials::bernsteinValues(vals11, n, x);
        });
        Kokkos::fence();

        auto h01 = getHostCopy(vals01);
        auto h11 = getHostCopy(vals11);

        // Also check symmetry B_{i,n}(-x) == B_{n-i,n}(x)
        Kokkos::View<double*, DeviceType> valsNeg("valsNeg", n + 1);
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int) {
          Polynomials::bernsteinValues(valsNeg, n, -x);
        });
        Kokkos::fence();
        auto hNeg = getHostCopy(valsNeg);

        double sum01     = 0.0;
        double sum11     = 0.0;
        double marsdenT  = 0.0;
        double marsdenX  = 0.0;
        double marsdenT2 = 0.0;

        for (ordinal_type i = 0; i <= n; ++i) {
          const double exact = Polynomials::binomialCoefficient(n, i)
                             * ((i == 0) ? 1.0 : std::pow(t, i))
                             * ((n - i == 0) ? 1.0 : std::pow(1.0 - t, n - i));

          TEST_ASSERT(h01(i) >= -tol);
          TEST_ASSERT(h11(i) >= -tol);
          TEST_ASSERT(std::abs(h01(i) - exact) <= tol);
          TEST_ASSERT(std::abs(h11(i) - exact) <= tol);
          TEST_ASSERT(std::abs(hNeg(i) - h11(n - i)) <= tol);

          sum01 += h01(i);
          sum11 += h11(i);
          if (n >= 1) {
            const double xi_t = static_cast<double>(i) / static_cast<double>(n);
            const double xi_x = -1.0 + 2.0 * xi_t;
            marsdenT += xi_t * h01(i);
            marsdenX += xi_x * h11(i);
          }
          if (n >= 2) {
            const double coeff2 = static_cast<double>(i * (i - 1)) / static_cast<double>(n * (n - 1));
            marsdenT2 += coeff2 * h01(i);
          }
        }

        TEST_FLOATING_EQUALITY(sum01, 1.0, tol);
        TEST_FLOATING_EQUALITY(sum11, 1.0, tol);
        if (n >= 1) {
          TEST_ASSERT(std::abs(marsdenT - t) <= tol);
          TEST_ASSERT(std::abs(marsdenX - x) <= tol);
        }
        if (n >= 2) {
          TEST_ASSERT(std::abs(marsdenT2 - t * t) <= tol);
        }

        // Endpoint Kronecker delta
        if (t == 0.0) {
          for (ordinal_type i = 0; i <= n; ++i) {
            TEST_ASSERT(std::abs(h01(i) - (i == 0 ? 1.0 : 0.0)) <= tol);
            TEST_ASSERT(std::abs(h11(i) - (i == 0 ? 1.0 : 0.0)) <= tol);
          }
        }
        if (t == 1.0) {
          for (ordinal_type i = 0; i <= n; ++i) {
            TEST_ASSERT(std::abs(h01(i) - (i == n ? 1.0 : 0.0)) <= tol);
            TEST_ASSERT(std::abs(h11(i) - (i == n ? 1.0 : 0.0)) <= tol);
          }
        }
      }
    }
  }

  // ============================================================================
  // 2. 1D Bernstein Derivatives vs. Exact Recurrence & Sacado AD
  // ============================================================================
  TEUCHOS_UNIT_TEST( BernsteinBasis, Polynomials_1D_DerivativesAndAD )
  {
    const std::vector<double> t_pts = {0.0, 0.15, 1.0 / 3.0, 0.5, 0.82, 1.0};

    for (ordinal_type n = 0; n <= 10; ++n) {
      for (double t : t_pts) {
        const double x = 2.0 * t - 1.0;

#ifdef HAVE_INTREPID2_SACADO
        {
          using FadType = Sacado::Fad::DFad<double>;
          Kokkos::View<FadType*, Kokkos::HostSpace> fad01("fad01", n + 1, 2);
          Kokkos::View<FadType*, Kokkos::HostSpace> fad11("fad11", n + 1, 2);
          FadType tFad(1, 0, t);
          FadType xFad(1, 0, x);
          Polynomials::shiftedBernsteinValues(fad01, n, tFad);
          Polynomials::bernsteinValues(fad11, n, xFad);

          Kokkos::View<double*, DeviceType> d01("d01", n + 1);
          Kokkos::View<double*, DeviceType> d11("d11", n + 1);
          auto policy = Kokkos::RangePolicy<ExecutionSpace>(0, 1);
          Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int) {
            Polynomials::shiftedBernsteinDerivativeValues(d01, n, t, 1);
            Polynomials::bernsteinDerivativeValues(d11, n, x, 1);
          });
          Kokkos::fence();
          auto hd01 = getHostCopy(d01);
          auto hd11 = getHostCopy(d11);

          const double adTol = 1e-12 * std::max(1.0, static_cast<double>(n * n));
          for (ordinal_type i = 0; i <= n; ++i) {
            TEST_ASSERT(std::abs(hd01(i) - fad01(i).dx(0)) <= adTol);
            TEST_ASSERT(std::abs(hd11(i) - fad11(i).dx(0)) <= adTol);
          }
        }
#endif

        // Verify higher derivatives k = 1..8 against exact degree-(n-1) difference relation
        // D^k B_{i,n}(t) = n * (D^{k-1} B_{i-1,n-1}(t) - D^{k-1} B_{i,n-1}(t))
        for (ordinal_type k = 1; k <= 8; ++k) {
          Kokkos::View<double*, DeviceType> dk01("dk01", n + 1);
          Kokkos::View<double*, DeviceType> dk11("dk11", n + 1);
          auto policy = Kokkos::RangePolicy<ExecutionSpace>(0, 1);
          Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int) {
            Polynomials::shiftedBernsteinDerivativeValues(dk01, n, t, k);
            Polynomials::bernsteinDerivativeValues(dk11, n, x, k);
          });
          Kokkos::fence();
          auto hdk01 = getHostCopy(dk01);
          auto hdk11 = getHostCopy(dk11);

          const double scale = std::ldexp(1.0, -k); // (1/2)^k
          const double dkTol = 1e-10 * std::pow(std::max(1.0, static_cast<double>(n)), k);
          double sumDk01 = 0.0;
          double sumDk11 = 0.0;

          if (n == 0 || k > n) {
            for (ordinal_type i = 0; i <= n; ++i) {
              TEST_EQUALITY(hdk01(i), 0.0);
              TEST_EQUALITY(hdk11(i), 0.0);
            }
          } else {
            Kokkos::View<double*, Kokkos::HostSpace> prev01("prev01", n);
            Polynomials::shiftedBernsteinDerivativeValues(prev01, n - 1, t, k - 1);
            for (ordinal_type i = 0; i <= n; ++i) {
              const double left  = (i > 0) ? prev01(i - 1) : 0.0;
              const double right = (i < n) ? prev01(i)     : 0.0;
              const double expected01 = static_cast<double>(n) * (left - right);
              const double expected11 = scale * expected01;
              TEST_ASSERT(std::abs(hdk01(i) - expected01) <= dkTol);
              TEST_ASSERT(std::abs(hdk11(i) - expected11) <= dkTol);
              sumDk01 += hdk01(i);
              sumDk11 += hdk11(i);
            }
            TEST_ASSERT(std::abs(sumDk01) <= dkTol);
            TEST_ASSERT(std::abs(sumDk11) <= dkTol);
          }
        }
      }
    }
  }

  // ============================================================================
  // Helper: Verify Partition of Unity, Non-Negativity, Marsden Identity,
  //         Vertex Kronecker Delta, and Subcell Trace Locality
  // ============================================================================
  template<typename BasisType>
  void verifyBernsteinGeometricProperties(const BasisType& basis,
                                          const Kokkos::DynRankView<double, DeviceType>& testPts,
                                          const bool checkSubcellLocality,
                                          Teuchos::FancyOStream& out,
                                          bool& success)
  {
    const ordinal_type numFields = basis.getCardinality();
    const ordinal_type numPts    = testPts.extent_int(0);
    const ordinal_type dim       = basis.getBaseCellTopology().getDimension();
    const double tol             = 1e-12;

    Kokkos::DynRankView<double, DeviceType> vals("vals", numFields, numPts);
    basis.getValues(vals, testPts, OPERATOR_VALUE);

    auto hVals = getHostCopy(vals);
    auto hPts  = getHostCopy(testPts);

    // Check non-negativity and partition of unity
    for (ordinal_type p = 0; p < numPts; ++p) {
      double sum = 0.0;
      for (ordinal_type f = 0; f < numFields; ++f) {
        if (hVals(f, p) < -tol) {
          success = false;
          out << "FAILURE (" << basis.getName() << " deg " << basis.getDegree()
              << "): negative value " << hVals(f, p) << " at field " << f << ", pt " << p << "\n";
        }
        sum += hVals(f, p);
      }
      if (std::abs(sum - 1.0) > tol) {
        success = false;
        out << "FAILURE (" << basis.getName() << " deg " << basis.getDegree()
            << "): partition of unity sum = " << sum << " != 1 at pt " << p << "\n";
      }
    }

    // Check gradients sum to zero (for degree >= 0)
    if (basis.getFunctionSpace() == FUNCTION_SPACE_HGRAD) {
      Kokkos::DynRankView<double, DeviceType> grads("grads", numFields, numPts, dim);
      basis.getValues(grads, testPts, OPERATOR_GRAD);
      auto hGrads = getHostCopy(grads);

      for (ordinal_type p = 0; p < numPts; ++p) {
        for (ordinal_type d = 0; d < dim; ++d) {
          double gsum = 0.0;
          for (ordinal_type f = 0; f < numFields; ++f) {
            gsum += hGrads(f, p, d);
          }
          if (std::abs(gsum) > 10.0 * tol) {
            success = false;
            out << "FAILURE (" << basis.getName() << "): grad sum(" << d << ") = " << gsum << " != 0\n";
          }
        }
      }
    }

    // Check Marsden linear coordinate reproduction using getDofCoords:
    //   sum_f dofCoords(f, d) * B_f(x) == x_d   (when degree >= 1 in each coordinate)
    if (basis.getDegree() >= 1) {
      Kokkos::DynRankView<double, DeviceType> dofCoords("dofCoords", numFields, dim);
      basis.getDofCoords(dofCoords);
      auto hCoords = getHostCopy(dofCoords);

      for (ordinal_type p = 0; p < numPts; ++p) {
        for (ordinal_type d = 0; d < dim; ++d) {
          double xReprod = 0.0;
          for (ordinal_type f = 0; f < numFields; ++f) {
            xReprod += hCoords(f, d) * hVals(f, p);
          }
          if (std::abs(xReprod - hPts(p, d)) > 10.0 * tol) {
            success = false;
            out << "FAILURE (" << basis.getName() << " deg " << basis.getDegree()
                << "): Marsden reproduction dim " << d << ": got " << xReprod
                << ", expected " << hPts(p, d) << "\n";
          }
        }
      }
    }

    // Check vertex Kronecker property and subcell trace vanishing for continuous HGRAD bases
    if (checkSubcellLocality) {
      const shards::CellTopology topo = basis.getBaseCellTopology();
      const ordinal_type numVerts = static_cast<ordinal_type>(topo.getVertexCount());
      Kokkos::DynRankView<double, HostSpaceType> refVerts("refVerts", numVerts, dim);
      for (ordinal_type v = 0; v < numVerts; ++v) {
        CellTools<HostSpaceType>::getReferenceVertex(Kokkos::subview(refVerts, v, Kokkos::ALL()), topo, v);
      }
      Kokkos::DynRankView<double, DeviceType> dVerts("dVerts", numVerts, dim);
      Kokkos::deep_copy(dVerts, refVerts);

      Kokkos::DynRankView<double, DeviceType> vertVals("vertVals", numFields, numVerts);
      basis.getValues(vertVals, dVerts, OPERATOR_VALUE);
      auto hVertVals = getHostCopy(vertVals);

      for (ordinal_type v = 0; v < numVerts; ++v) {
        const ordinal_type vDof = basis.getDofOrdinal(0, v, 0);
        for (ordinal_type f = 0; f < numFields; ++f) {
          const double expected = (f == vDof) ? 1.0 : 0.0;
          if (std::abs(hVertVals(f, v) - expected) > tol) {
            success = false;
            out << "FAILURE (" << basis.getName() << "): vertex Kronecker failed at vertex "
                << v << ", field " << f << ": got " << hVertVals(f, v) << ", expected " << expected << "\n";
          }
        }
      }

      // Check edge midpoint locality: at midpoint of edge e = (v0, v1), only DOFs on v0, v1, or edge e can be nonzero
      if (dim >= 2) {
        const ordinal_type numEdges = static_cast<ordinal_type>(topo.getEdgeCount());
        Kokkos::DynRankView<double, HostSpaceType> edgeMids("edgeMids", numEdges, dim);
        for (ordinal_type e = 0; e < numEdges; ++e) {
          const ordinal_type v0 = static_cast<ordinal_type>(topo.getNodeMap(1, e, 0));
          const ordinal_type v1 = static_cast<ordinal_type>(topo.getNodeMap(1, e, 1));
          for (ordinal_type d = 0; d < dim; ++d) {
            edgeMids(e, d) = 0.5 * (refVerts(v0, d) + refVerts(v1, d));
          }
        }
        Kokkos::DynRankView<double, DeviceType> dEdgeMids("dEdgeMids", numEdges, dim);
        Kokkos::deep_copy(dEdgeMids, edgeMids);
        Kokkos::DynRankView<double, DeviceType> edgeVals("edgeVals", numFields, numEdges);
        basis.getValues(edgeVals, dEdgeMids, OPERATOR_VALUE);
        auto hEdgeVals = getHostCopy(edgeVals);

        auto tags = basis.getAllDofTags();
        for (ordinal_type e = 0; e < numEdges; ++e) {
          const ordinal_type v0 = static_cast<ordinal_type>(topo.getNodeMap(1, e, 0));
          const ordinal_type v1 = static_cast<ordinal_type>(topo.getNodeMap(1, e, 1));
          for (ordinal_type f = 0; f < numFields; ++f) {
            const ordinal_type scDim = tags(f, 0);
            const ordinal_type scOrd = tags(f, 1);
            const bool onClosure = (scDim == 0 && (scOrd == v0 || scOrd == v1))
                                || (scDim == 1 && scOrd == e);
            if (!onClosure && std::abs(hEdgeVals(f, e)) > tol) {
              success = false;
              out << "FAILURE (" << basis.getName() << "): field " << f << " (scDim=" << scDim
                  << ", scOrd=" << scOrd << ") is nonzero (" << hEdgeVals(f, e)
                  << ") on edge " << e << "\n";
            }
            if (onClosure && hEdgeVals(f, e) <= tol) {
              success = false;
              out << "FAILURE (" << basis.getName() << "): closure field " << f << " (scDim=" << scDim
                  << ", scOrd=" << scOrd << ") is not strictly positive (" << hEdgeVals(f, e)
                  << ") at interior of edge " << e << "\n";
            }
          }
        }
      }

      // Check face centroid locality in 3D: at centroid of face s, only DOFs on vertices/edges/interior of face s are nonzero
      if (dim == 3) {
        const ordinal_type numFaces = static_cast<ordinal_type>(topo.getFaceCount());
        Kokkos::DynRankView<double, HostSpaceType> faceCents("faceCents", numFaces, dim);
        for (ordinal_type s = 0; s < numFaces; ++s) {
          const ordinal_type nfv = static_cast<ordinal_type>(topo.getVertexCount(2, s));
          for (ordinal_type d = 0; d < dim; ++d) faceCents(s, d) = 0.0;
          for (ordinal_type iv = 0; iv < nfv; ++iv) {
            const ordinal_type v = static_cast<ordinal_type>(topo.getNodeMap(2, s, iv));
            for (ordinal_type d = 0; d < dim; ++d) {
              faceCents(s, d) += refVerts(v, d) / static_cast<double>(nfv);
            }
          }
        }
        Kokkos::DynRankView<double, DeviceType> dFaceCents("dFaceCents", numFaces, dim);
        Kokkos::deep_copy(dFaceCents, faceCents);
        Kokkos::DynRankView<double, DeviceType> faceVals("faceVals", numFields, numFaces);
        basis.getValues(faceVals, dFaceCents, OPERATOR_VALUE);
        auto hFaceVals = getHostCopy(faceVals);

        auto tags = basis.getAllDofTags();
        for (ordinal_type s = 0; s < numFaces; ++s) {
          const ordinal_type nfv = static_cast<ordinal_type>(topo.getVertexCount(2, s));
          std::vector<ordinal_type> faceVerts(nfv);
          for (ordinal_type iv = 0; iv < nfv; ++iv) {
            faceVerts[iv] = static_cast<ordinal_type>(topo.getNodeMap(2, s, iv));
          }
          for (ordinal_type f = 0; f < numFields; ++f) {
            const ordinal_type scDim = tags(f, 0);
            const ordinal_type scOrd = tags(f, 1);
            bool onFaceClosure = false;
            if (scDim == 0) {
              onFaceClosure = (std::find(faceVerts.begin(), faceVerts.end(), scOrd) != faceVerts.end());
            } else if (scDim == 1) {
              const ordinal_type ev0 = static_cast<ordinal_type>(topo.getNodeMap(1, scOrd, 0));
              const ordinal_type ev1 = static_cast<ordinal_type>(topo.getNodeMap(1, scOrd, 1));
              onFaceClosure = (std::find(faceVerts.begin(), faceVerts.end(), ev0) != faceVerts.end())
                           && (std::find(faceVerts.begin(), faceVerts.end(), ev1) != faceVerts.end());
            } else if (scDim == 2) {
              onFaceClosure = (scOrd == s);
            }
            if (!onFaceClosure && std::abs(hFaceVals(f, s)) > tol) {
              success = false;
              out << "FAILURE (" << basis.getName() << "): field " << f << " (scDim=" << scDim
                  << ", scOrd=" << scOrd << ") is nonzero (" << hFaceVals(f, s)
                  << ") on face " << s << "\n";
            }
            if (onFaceClosure && hFaceVals(f, s) <= tol) {
              success = false;
              out << "FAILURE (" << basis.getName() << "): face-closure field " << f << " (scDim=" << scDim
                  << ", scOrd=" << scOrd << ") is not strictly positive (" << hFaceVals(f, s)
                  << ") at centroid of face " << s << "\n";
            }
          }
        }
      }
    }
  }

  // Helper to generate reference cell test points (cubature + lattice including boundary)
  Kokkos::DynRankView<double, DeviceType>
  makeRefCellTestPoints(const shards::CellTopology& topo, const int latticeOrder = 4)
  {
    const ordinal_type dim  = static_cast<ordinal_type>(topo.getDimension());
    const ordinal_type nLat = PointTools::getLatticeSize(topo, latticeOrder, 0);
    Kokkos::DynRankView<double, HostSpaceType> hPts("hPts", nLat, dim);
    PointTools::getLattice(hPts, topo, latticeOrder, 0, POINTTYPE_EQUISPACED);
    Kokkos::DynRankView<double, DeviceType> dPts("dPts", nLat, dim);
    Kokkos::deep_copy(dPts, hPts);
    return dPts;
  }

  // ============================================================================
  // 3. Geometric Properties Across All Simplicial & Tensor Topologies
  // ============================================================================
  TEUCHOS_UNIT_TEST( BernsteinBasis, PartitionOfUnity_NonNegativity_Marsden_AllTopologies )
  {
    // --- LINE [-1, 1] and [0, 1] ---
    {
      const shards::CellTopology lineTopo(shards::getCellTopologyData<shards::Line<2>>());
      auto pts11 = makeRefCellTestPoints(lineTopo, 7);

      Kokkos::DynRankView<double, HostSpaceType> hPts01("hPts01", 8, 1);
      for (int i = 0; i < 8; ++i) hPts01(i, 0) = static_cast<double>(i) / 7.0;
      Kokkos::DynRankView<double, DeviceType> pts01("pts01", 8, 1);
      Kokkos::deep_copy(pts01, hPts01);

      for (int p = 1; p <= 6; ++p) {
        BernsteinBasis_HGRAD_LINE<DeviceType, double, double, true, true>  lineCG(p);
        BernsteinBasis_HGRAD_LINE<DeviceType, double, double, false, true> lineDG(p);
        BernsteinBasis_HGRAD_LINE<DeviceType, double, double, true, false> line01(p);
        verifyBernsteinGeometricProperties(lineCG, pts11, true,  out, success);
        verifyBernsteinGeometricProperties(lineDG, pts11, false, out, success);
        verifyBernsteinGeometricProperties(line01, pts01, false, out, success);
      }
      for (int p = 0; p <= 5; ++p) {
        BernsteinBasis_HVOL_LINE<DeviceType, double, double, true>  lineHvol(p);
        BernsteinBasis_HVOL_LINE<DeviceType, double, double, false> lineHvol01(p);
        verifyBernsteinGeometricProperties(lineHvol,   pts11, false, out, success);
        verifyBernsteinGeometricProperties(lineHvol01, pts01, false, out, success);
      }
    }

    // --- TRIANGLE ---
    {
      const shards::CellTopology triTopo(shards::getCellTopologyData<shards::Triangle<3>>());
      auto pts = makeRefCellTestPoints(triTopo, 5);
      for (int p = 1; p <= 6; ++p) {
        BernsteinBasis_HGRAD_TRI<DeviceType, double, double, true>  triCG(p);
        BernsteinBasis_HGRAD_TRI<DeviceType, double, double, false> triDG(p);
        verifyBernsteinGeometricProperties(triCG, pts, true,  out, success);
        verifyBernsteinGeometricProperties(triDG, pts, false, out, success);
      }
      for (int p = 0; p <= 5; ++p) {
        BernsteinBasis_HVOL_TRI<DeviceType> triHvol(p);
        verifyBernsteinGeometricProperties(triHvol, pts, false, out, success);
      }
    }

    // --- QUADRILATERAL (isotropic & anisotropic) ---
    {
      const shards::CellTopology quadTopo(shards::getCellTopologyData<shards::Quadrilateral<4>>());
      auto pts = makeRefCellTestPoints(quadTopo, 5);
      for (int p = 1; p <= 5; ++p) {
        BernsteinBasis_HGRAD_QUAD<DeviceType, double, double, true>  quadCG(p);
        BernsteinBasis_HGRAD_QUAD<DeviceType, double, double, false> quadDG(p);
        verifyBernsteinGeometricProperties(quadCG, pts, true,  out, success);
        verifyBernsteinGeometricProperties(quadDG, pts, false, out, success);
      }
      BernsteinBasis_HGRAD_QUAD<DeviceType> quadAniso1(2, 4);
      BernsteinBasis_HGRAD_QUAD<DeviceType> quadAniso2(3, 1);
      verifyBernsteinGeometricProperties(quadAniso1, pts, true, out, success);
      verifyBernsteinGeometricProperties(quadAniso2, pts, true, out, success);
    }

    // --- TETRAHEDRON ---
    {
      const shards::CellTopology tetTopo(shards::getCellTopologyData<shards::Tetrahedron<4>>());
      auto pts = makeRefCellTestPoints(tetTopo, 4);
      for (int p = 1; p <= 5; ++p) {
        BernsteinBasis_HGRAD_TET<DeviceType, double, double, true>  tetCG(p);
        BernsteinBasis_HGRAD_TET<DeviceType, double, double, false> tetDG(p);
        verifyBernsteinGeometricProperties(tetCG, pts, true,  out, success);
        verifyBernsteinGeometricProperties(tetDG, pts, false, out, success);
      }
      for (int p = 0; p <= 4; ++p) {
        BernsteinBasis_HVOL_TET<DeviceType> tetHvol(p);
        verifyBernsteinGeometricProperties(tetHvol, pts, false, out, success);
      }
    }

    // --- HEXAHEDRON (isotropic & anisotropic) ---
    {
      const shards::CellTopology hexTopo(shards::getCellTopologyData<shards::Hexahedron<8>>());
      auto pts = makeRefCellTestPoints(hexTopo, 3);
      for (int p = 1; p <= 4; ++p) {
        BernsteinBasis_HGRAD_HEX<DeviceType, double, double, true>  hexCG(p);
        BernsteinBasis_HGRAD_HEX<DeviceType, double, double, false> hexDG(p);
        verifyBernsteinGeometricProperties(hexCG, pts, true,  out, success);
        verifyBernsteinGeometricProperties(hexDG, pts, false, out, success);
      }
      BernsteinBasis_HGRAD_HEX<DeviceType> hexAniso(1, 2, 3);
      verifyBernsteinGeometricProperties(hexAniso, pts, true, out, success);
    }

    // --- WEDGE (isotropic & anisotropic) ---
    {
      const shards::CellTopology wedgeTopo(shards::getCellTopologyData<shards::Wedge<6>>());
      // Construct test points on Wedge = Tri x Line
      const shards::CellTopology triTopo(shards::getCellTopologyData<shards::Triangle<3>>());
      const ordinal_type nTri = PointTools::getLatticeSize(triTopo, 3, 0);
      Kokkos::DynRankView<double, HostSpaceType> hTri("hTri", nTri, 2);
      PointTools::getLattice(hTri, triTopo, 3, 0, POINTTYPE_EQUISPACED);
      const std::vector<double> z_pts = {-1.0, -0.4, 0.25, 1.0};
      const ordinal_type nWedge = nTri * static_cast<ordinal_type>(z_pts.size());
      Kokkos::DynRankView<double, HostSpaceType> hWedge("hWedge", nWedge, 3);
      ordinal_type idx = 0;
      for (double z : z_pts) {
        for (ordinal_type i = 0; i < nTri; ++i, ++idx) {
          hWedge(idx, 0) = hTri(i, 0);
          hWedge(idx, 1) = hTri(i, 1);
          hWedge(idx, 2) = z;
        }
      }
      Kokkos::DynRankView<double, DeviceType> dWedge("dWedge", nWedge, 3);
      Kokkos::deep_copy(dWedge, hWedge);

      for (int p = 1; p <= 4; ++p) {
        BernsteinBasis_HGRAD_WEDGE<DeviceType, double, double, true>  wedgeCG(p);
        BernsteinBasis_HGRAD_WEDGE<DeviceType, double, double, false> wedgeDG(p);
        verifyBernsteinGeometricProperties(wedgeCG, dWedge, true,  out, success);
        verifyBernsteinGeometricProperties(wedgeDG, dWedge, false, out, success);
      }
      BernsteinBasis_HGRAD_WEDGE<DeviceType> wedgeAniso(3, 2);
      verifyBernsteinGeometricProperties(wedgeAniso, dWedge, true, out, success);
    }
  }

#ifdef HAVE_INTREPID2_SACADO
  // ============================================================================
  // Helper: Verify Analytical GRAD, CURL, and D1..D4 against Sacado DFad AD
  // ============================================================================
  template<template<typename, typename, typename, bool...> class BasisTemplate, bool... Flags>
  void verifyDerivativesAgainstAD(const int polyOrder,
                                  const Kokkos::DynRankView<double, DeviceType>& pts,
                                  const int maxDkOrder,
                                  const bool test2DCurl,
                                  Teuchos::FancyOStream& out,
                                  bool& success)
  {
    using FadType   = Sacado::Fad::DFad<double>;
    using DblBasis  = BasisTemplate<DeviceType, double, double, Flags...>;
    using FadBasis  = BasisTemplate<DeviceType, FadType, FadType, Flags...>;

    DblBasis dblBasis(polyOrder);
    FadBasis fadBasis(polyOrder);

    const ordinal_type numFields = dblBasis.getCardinality();
    const ordinal_type numPts    = pts.extent_int(0);
    const ordinal_type dim       = pts.extent_int(1);
    const double tol             = 1e-11 * std::max(1.0, std::pow(static_cast<double>(polyOrder), maxDkOrder));

    auto hPts = getHostCopy(pts);

    // Seed Fad input points on host and copy to device
    Kokkos::DynRankView<FadType, HostSpaceType> hFadPts("hFadPts", numPts, dim, dim + 1);
    for (ordinal_type p = 0; p < numPts; ++p) {
      for (ordinal_type d = 0; d < dim; ++d) {
        hFadPts(p, d) = FadType(dim, d, hPts(p, d));
      }
    }
    Kokkos::DynRankView<FadType, DeviceType> dFadPts("dFadPts", numPts, dim, dim + 1);
    Kokkos::deep_copy(dFadPts, hFadPts);

    // 1. Compare OPERATOR_VALUE (val) and OPERATOR_GRAD / OPERATOR_D1 / OPERATOR_CURL against AD of OPERATOR_VALUE
    Kokkos::DynRankView<FadType, DeviceType> fadVals("fadVals", numFields, numPts, dim + 1);
    fadBasis.getValues(fadVals, dFadPts, OPERATOR_VALUE);
    auto hFadVals = getHostCopy(fadVals);

    Kokkos::DynRankView<double, DeviceType> dblVals("dblVals", numFields, numPts);
    Kokkos::DynRankView<double, DeviceType> dblGrads("dblGrads", numFields, numPts, dim);
    Kokkos::DynRankView<double, DeviceType> dblD1("dblD1", numFields, numPts, dim);
    dblBasis.getValues(dblVals,  pts, OPERATOR_VALUE);
    dblBasis.getValues(dblGrads, pts, OPERATOR_GRAD);
    dblBasis.getValues(dblD1,    pts, OPERATOR_D1);

    auto hVals  = getHostCopy(dblVals);
    auto hGrads = getHostCopy(dblGrads);
    auto hD1    = getHostCopy(dblD1);

    for (ordinal_type f = 0; f < numFields; ++f) {
      for (ordinal_type p = 0; p < numPts; ++p) {
        if (std::abs(hVals(f, p) - hFadVals(f, p).val()) > tol) {
          success = false;
          out << "FAILURE (" << dblBasis.getName() << "): VALUE mismatch vs Fad.val()\n";
        }
        for (ordinal_type d = 0; d < dim; ++d) {
          const double adGrad = hFadVals(f, p).dx(d);
          if (std::abs(hGrads(f, p, d) - adGrad) > tol) {
            success = false;
            out << "FAILURE (" << dblBasis.getName() << " p=" << polyOrder
                << "): GRAD mismatch at f=" << f << ", pt=" << p << ", d=" << d
                << ": got " << hGrads(f, p, d) << ", AD=" << adGrad << "\n";
          }
          if (std::abs(hD1(f, p, d) - adGrad) > tol) {
            success = false;
            out << "FAILURE (" << dblBasis.getName() << " p=" << polyOrder
                << "): D1 mismatch at f=" << f << ", pt=" << p << ", d=" << d << "\n";
          }
        }
      }
    }

    if (test2DCurl && dim == 2) {
      Kokkos::DynRankView<double, DeviceType> dblCurl("dblCurl", numFields, numPts, 2);
      dblBasis.getValues(dblCurl, pts, OPERATOR_CURL);
      auto hCurl = getHostCopy(dblCurl);
      for (ordinal_type f = 0; f < numFields; ++f) {
        for (ordinal_type p = 0; p < numPts; ++p) {
          const double dfdx = hFadVals(f, p).dx(0);
          const double dfdy = hFadVals(f, p).dx(1);
          if (std::abs(hCurl(f, p, 0) - dfdy) > tol || std::abs(hCurl(f, p, 1) - (-dfdx)) > tol) {
            success = false;
            out << "FAILURE (" << dblBasis.getName() << " p=" << polyOrder
                << "): 2D CURL mismatch vs (+df/dy, -df/dx)\n";
          }
        }
      }
    }

    // 2. Inductively verify OPERATOR_D{k} (k = 2..maxDkOrder) against AD of OPERATOR_D{k-1}!
    for (int k = 2; k <= maxDkOrder; ++k) {
      const EOperator opPrev = static_cast<EOperator>(OPERATOR_D1 + (k - 2));
      const EOperator opCurr = static_cast<EOperator>(OPERATOR_D1 + (k - 1));
      const ordinal_type cardPrev = getDkCardinality(opPrev, dim);
      const ordinal_type cardCurr = getDkCardinality(opCurr, dim);

      Kokkos::DynRankView<FadType, DeviceType> fadDkPrev("fadDkPrev", numFields, numPts, cardPrev, dim + 1);
      fadBasis.getValues(fadDkPrev, dFadPts, opPrev);
      auto hFadPrev = getHostCopy(fadDkPrev);

      Kokkos::DynRankView<double, DeviceType> dblDkCurr("dblDkCurr", numFields, numPts, cardCurr);
      dblBasis.getValues(dblDkCurr, pts, opCurr);
      auto hCurr = getHostCopy(dblDkCurr);

      if (dim == 1) {
        for (ordinal_type f = 0; f < numFields; ++f) {
          for (ordinal_type p = 0; p < numPts; ++p) {
            const double adVal = hFadPrev(f, p, 0).dx(0);
            if (std::abs(hCurr(f, p, 0) - adVal) > tol) {
              success = false;
              out << "FAILURE (" << dblBasis.getName() << " p=" << polyOrder
                  << "): D" << k << " mismatch: got " << hCurr(f, p, 0) << ", AD=" << adVal << "\n";
            }
          }
        }
      } else if (dim == 2) {
        for (ordinal_type bx = 0; bx <= k - 1; ++bx) {
          const ordinal_type by = (k - 1) - bx;
          const ordinal_type prevIdx = getDkEnumeration<2>(bx, by);
          const ordinal_type idxX    = getDkEnumeration<2>(bx + 1, by);
          const ordinal_type idxY    = getDkEnumeration<2>(bx, by + 1);
          for (ordinal_type f = 0; f < numFields; ++f) {
            for (ordinal_type p = 0; p < numPts; ++p) {
              const double adDx = hFadPrev(f, p, prevIdx).dx(0);
              const double adDy = hFadPrev(f, p, prevIdx).dx(1);
              if (std::abs(hCurr(f, p, idxX) - adDx) > tol || std::abs(hCurr(f, p, idxY) - adDy) > tol) {
                success = false;
                out << "FAILURE (" << dblBasis.getName() << " p=" << polyOrder
                    << "): 2D D" << k << " mixed partial mismatch at (" << bx << "," << by << ")\n";
              }
            }
          }
        }
      } else if (dim == 3) {
        for (ordinal_type bx = 0; bx <= k - 1; ++bx) {
          for (ordinal_type by = 0; by <= (k - 1) - bx; ++by) {
            const ordinal_type bz = (k - 1) - bx - by;
            const ordinal_type prevIdx = getDkEnumeration<3>(bx, by, bz);
            const ordinal_type idxX    = getDkEnumeration<3>(bx + 1, by, bz);
            const ordinal_type idxY    = getDkEnumeration<3>(bx, by + 1, bz);
            const ordinal_type idxZ    = getDkEnumeration<3>(bx, by, bz + 1);
            for (ordinal_type f = 0; f < numFields; ++f) {
              for (ordinal_type p = 0; p < numPts; ++p) {
                const double adDx = hFadPrev(f, p, prevIdx).dx(0);
                const double adDy = hFadPrev(f, p, prevIdx).dx(1);
                const double adDz = hFadPrev(f, p, prevIdx).dx(2);
                if (std::abs(hCurr(f, p, idxX) - adDx) > tol ||
                    std::abs(hCurr(f, p, idxY) - adDy) > tol ||
                    std::abs(hCurr(f, p, idxZ) - adDz) > tol) {
                  success = false;
                  out << "FAILURE (" << dblBasis.getName() << " p=" << polyOrder
                      << "): 3D D" << k << " mixed partial mismatch at ("
                      << bx << "," << by << "," << bz << ")\n";
                }
              }
            }
          }
        }
      }
    }
  }

  // Wrapper aliases with uniform template signature <Device, Out, Pt>
  template<typename DT, typename OS, typename PS>
  using LineBasis11 = BernsteinBasis_HGRAD_LINE<DT, OS, PS, true, true>;

  template<typename DT, typename OS, typename PS>
  using LineBasis01 = BernsteinBasis_HGRAD_LINE<DT, OS, PS, true, false>;

  template<typename DT, typename OS, typename PS>
  using TriBasisCG = BernsteinBasis_HGRAD_TRI<DT, OS, PS, true>;

  template<typename DT, typename OS, typename PS>
  using QuadBasisCG = BernsteinBasis_HGRAD_QUAD<DT, OS, PS, true>;

  template<typename DT, typename OS, typename PS>
  using TetBasisCG = BernsteinBasis_HGRAD_TET<DT, OS, PS, true>;

  template<typename DT, typename OS, typename PS>
  using HexBasisCG = BernsteinBasis_HGRAD_HEX<DT, OS, PS, true>;

  template<typename DT, typename OS, typename PS>
  using WedgeBasisCG = BernsteinBasis_HGRAD_WEDGE<DT, OS, PS, true>;

  // ============================================================================
  // 4. Analytical Derivatives vs. Sacado AD Across All Topologies
  // ============================================================================
  TEUCHOS_UNIT_TEST( BernsteinBasis, AnalyticalDerivativesMatchSacadoAD )
  {
    const shards::CellTopology lineTopo(shards::getCellTopologyData<shards::Line<2>>());
    const shards::CellTopology triTopo(shards::getCellTopologyData<shards::Triangle<3>>());
    const shards::CellTopology quadTopo(shards::getCellTopologyData<shards::Quadrilateral<4>>());
    const shards::CellTopology tetTopo(shards::getCellTopologyData<shards::Tetrahedron<4>>());
    const shards::CellTopology hexTopo(shards::getCellTopologyData<shards::Hexahedron<8>>());

    auto linePts11 = makeRefCellTestPoints(lineTopo, 5);
    Kokkos::DynRankView<double, HostSpaceType> hPts01("hPts01", 6, 1);
    for (int i = 0; i < 6; ++i) hPts01(i, 0) = static_cast<double>(i) / 5.0;
    Kokkos::DynRankView<double, DeviceType> linePts01("linePts01", 6, 1);
    Kokkos::deep_copy(linePts01, hPts01);

    auto triPts  = makeRefCellTestPoints(triTopo, 4);
    auto quadPts = makeRefCellTestPoints(quadTopo, 3);
    auto tetPts  = makeRefCellTestPoints(tetTopo, 3);
    auto hexPts  = makeRefCellTestPoints(hexTopo, 2);

    // Wedge points (boundary + interior)
    Kokkos::DynRankView<double, HostSpaceType> hWedgePts("hWedgePts", 6, 3);
    const double wp[6][3] = {
      {0.0, 0.0, -1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 0.0},
      {0.2, 0.3, -0.5}, {0.5, 0.5, 0.4}, {1.0/3.0, 1.0/3.0, 0.1}
    };
    for (int i = 0; i < 6; ++i)
      for (int d = 0; d < 3; ++d)
        hWedgePts(i, d) = wp[i][d];
    Kokkos::DynRankView<double, DeviceType> wedgePts("wedgePts", 6, 3);
    Kokkos::deep_copy(wedgePts, hWedgePts);

    for (int p = 1; p <= 5; ++p) {
      verifyDerivativesAgainstAD<LineBasis11>(p, linePts11, 5, false, out, success);
      verifyDerivativesAgainstAD<LineBasis01>(p, linePts01, 5, false, out, success);
      verifyDerivativesAgainstAD<TriBasisCG>(p,  triPts,    4, true,  out, success);
    }
    for (int p = 1; p <= 4; ++p) {
      verifyDerivativesAgainstAD<QuadBasisCG>(p,  quadPts,  3, true,  out, success);
      verifyDerivativesAgainstAD<TetBasisCG>(p,   tetPts,   4, false, out, success);
    }
    for (int p = 1; p <= 3; ++p) {
      verifyDerivativesAgainstAD<HexBasisCG>(p,   hexPts,   3, false, out, success);
      verifyDerivativesAgainstAD<WedgeBasisCG>(p, wedgePts, 3, false, out, success);
    }
  }
#endif // HAVE_INTREPID2_SACADO

  // ============================================================================
  // Helper: Verify Team-Level Workset getValues Matches Standard getValues
  // ============================================================================
  template<typename BasisType>
  void verifyTeamLevelGetValues(const BasisType& basis,
                                const Kokkos::DynRankView<double, DeviceType>& inputPts,
                                const std::vector<EOperator>& ops,
                                Teuchos::FancyOStream& out,
                                bool& success)
  {
    const int ncells    = 4;
    const int numFields = basis.getCardinality();
    const int npts      = inputPts.extent_int(0);
    const int ndim      = inputPts.extent_int(1);
    const double tol    = 10.0 * TEST_TOLERANCE_TIGHT;

    auto basisPtrDevice    = copy_virtual_class_to_device<DeviceType, BasisType>(basis);
    auto basisRawPtrDevice = basisPtrDevice.get();

    const int scratchLevel = 1;
    const int vectorSize   = getVectorSizeForHierarchicalParallelism<double>();
    Kokkos::TeamPolicy<ExecutionSpace> basePolicy(ncells, Kokkos::AUTO, vectorSize);

    for (EOperator op : ops) {
      Kokkos::DynRankView<double, DeviceType> teamOut, stdOut;
      if (op == OPERATOR_VALUE) {
        teamOut = Kokkos::DynRankView<double, DeviceType>("teamOut", ncells, numFields, npts);
        stdOut  = Kokkos::DynRankView<double, DeviceType>("stdOut",  numFields, npts);
      } else {
        teamOut = Kokkos::DynRankView<double, DeviceType>("teamOut", ncells, numFields, npts, ndim);
        stdOut  = Kokkos::DynRankView<double, DeviceType>("stdOut",  numFields, npts, ndim);
      }

      auto functor = KOKKOS_LAMBDA(typename Kokkos::TeamPolicy<ExecutionSpace>::member_type teamMember) {
        auto cellView = Kokkos::subview(teamOut, teamMember.league_rank(), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        basisRawPtrDevice->getValues(cellView, inputPts, op, teamMember, scratchLevel);
      };

      int teamSize = std::min(npts, basePolicy.team_size_recommended(functor, Kokkos::ParallelForTag()));
      Kokkos::TeamPolicy<ExecutionSpace> policy(ncells, teamSize, vectorSize);
      int perThreadSpaceSize = 0;
      basis.getScratchSpaceSize(perThreadSpaceSize, inputPts, op);
      policy.set_scratch_size(scratchLevel, Kokkos::PerThread(perThreadSpaceSize));

      Kokkos::parallel_for(policy, functor);
      Kokkos::fence();

      basis.getValues(stdOut, inputPts, op);

      auto hTeam = getHostCopy(teamOut);
      auto hStd  = getHostCopy(stdOut);

      for (int c = 0; c < ncells; ++c) {
        for (int f = 0; f < numFields; ++f) {
          for (int p = 0; p < npts; ++p) {
            if (op == OPERATOR_VALUE) {
              if (std::abs(hTeam(c, f, p) - hStd(f, p)) > tol) {
                success = false;
                out << "FAILURE (" << basis.getName() << "): team VALUE mismatch at cell " << c << "\n";
              }
            } else {
              for (int d = 0; d < ndim; ++d) {
                if (std::abs(hTeam(c, f, p, d) - hStd(f, p, d)) > tol) {
                  success = false;
                  out << "FAILURE (" << basis.getName() << "): team " << EOperatorToString(op)
                      << " mismatch at cell " << c << "\n";
                }
              }
            }
          }
        }
      }
    }
  }

  // ============================================================================
  // 5. Team-Level Workset getValues Across Simplicial Bases
  // ============================================================================
  TEUCHOS_UNIT_TEST( BernsteinBasis, TeamLevelWorksetGetValues )
  {
    const shards::CellTopology lineTopo(shards::getCellTopologyData<shards::Line<2>>());
    const shards::CellTopology triTopo(shards::getCellTopologyData<shards::Triangle<3>>());
    const shards::CellTopology tetTopo(shards::getCellTopologyData<shards::Tetrahedron<4>>());

    auto linePts = makeRefCellTestPoints(lineTopo, 6);
    auto triPts  = makeRefCellTestPoints(triTopo,  4);
    auto tetPts  = makeRefCellTestPoints(tetTopo,  3);

    for (int p = 1; p <= 5; ++p) {
      BernsteinBasis_HGRAD_LINE<DeviceType> lineCG(p);
      BernsteinBasis_HGRAD_TRI<DeviceType>  triCG(p);
      BernsteinBasis_HGRAD_TET<DeviceType>  tetCG(p);

      verifyTeamLevelGetValues(lineCG, linePts, {OPERATOR_VALUE, OPERATOR_GRAD}, out, success);
      verifyTeamLevelGetValues(triCG,  triPts,  {OPERATOR_VALUE, OPERATOR_GRAD, OPERATOR_CURL}, out, success);
      verifyTeamLevelGetValues(tetCG,  tetPts,  {OPERATOR_VALUE, OPERATOR_GRAD}, out, success);
    }

    for (int p = 0; p <= 4; ++p) {
      BernsteinBasis_HVOL_LINE<DeviceType> lineHvol(p);
      BernsteinBasis_HVOL_TRI<DeviceType>  triHvol(p);
      BernsteinBasis_HVOL_TET<DeviceType>  tetHvol(p);

      verifyTeamLevelGetValues(lineHvol, linePts, {OPERATOR_VALUE}, out, success);
      verifyTeamLevelGetValues(triHvol,  triPts,  {OPERATOR_VALUE}, out, success);
      verifyTeamLevelGetValues(tetHvol,  tetPts,  {OPERATOR_VALUE}, out, success);
    }
  }

  // ============================================================================
  // Helper: Verify Subcell Orientation Coefficient Matrices Are Exact Permutations
  // ============================================================================
  template<typename BasisType>
  void verifyOrientationPermutationMatrices(const BasisType& basis,
                                            const bool checkIdentityAtOrt0,
                                            Teuchos::FancyOStream& out,
                                            bool& success)
  {
    using ots = OrientationTools<DeviceType>;

    auto hostBasis = basis.getHostBasis();
    TEST_ASSERT(hostBasis.get() != nullptr);
    TEST_EQUALITY(hostBasis->getCardinality(), basis.getCardinality());

    if (!basis.requireOrientation()) {
      return;
    }

    const auto matData     = ots::createCoeffMatrix(&basis);
    const auto matDataHost = getHostCopy(matData);
    const auto topo        = basis.getBaseCellTopology();
    const ordinal_type numEdges = static_cast<ordinal_type>(topo.getSubcellCount(1));
    const ordinal_type numFaces = static_cast<ordinal_type>(topo.getSubcellCount(2));

    bool hasEdgeDofs = false;
    for (ordinal_type e = 0; e < numEdges; ++e) {
      const ordinal_type ndof = basis.getDofCount(1, e);
      if (ndof > 0) hasEdgeDofs = true;
      for (ordinal_type ort = 0; ort < 2 && ndof > 0; ++ort) {
        for (ordinal_type i = 0; i < ndof; ++i) {
          double rowSum = 0.0;
          double colSum = 0.0;
          for (ordinal_type j = 0; j < ndof; ++j) {
            const double valIJ = matDataHost(e, ort, i, j);
            const double valJI = matDataHost(e, ort, j, i);
            // Every entry of a Bernstein orientation matrix must be in {0, 1}
            if (std::abs(valIJ) > 1e-12 && std::abs(valIJ - 1.0) > 1e-12) {
              success = false;
              out << "FAILURE (" << basis.getName() << "): non-binary edge coeff " << valIJ << "\n";
            }
            rowSum += valIJ;
            colSum += valJI;
          }
          if (std::abs(rowSum - 1.0) > 1e-12 || std::abs(colSum - 1.0) > 1e-12) {
            success = false;
            out << "FAILURE (" << basis.getName() << " edge " << e << " ort " << ort
                << "): rowSum=" << rowSum << ", colSum=" << colSum << "\n";
          }
        }
        // On simplicial bases (TRI, TET), local subcell DOF numbering matches Shards ort == 0
        if (checkIdentityAtOrt0 && ort == 0) {
          for (ordinal_type i = 0; i < ndof; ++i) {
            if (std::abs(matDataHost(e, 0, i, i) - 1.0) > 1e-12) {
              success = false;
              out << "FAILURE (" << basis.getName() << " edge " << e
                  << "): ort 0 is not identity at i=" << i << "\n";
            }
          }
        }
      }
    }

    const ordinal_type faceOffset = hasEdgeDofs ? numEdges : 0;
    for (ordinal_type f = 0; f < numFaces; ++f) {
      const ordinal_type ndof = basis.getDofCount(2, f);
      if (ndof == 0) continue;
      const ordinal_type numFaceOrts = 2 * static_cast<ordinal_type>(topo.getSideCount(2, f));
      for (ordinal_type ort = 0; ort < numFaceOrts; ++ort) {
        for (ordinal_type i = 0; i < ndof; ++i) {
          double rowSum = 0.0;
          double colSum = 0.0;
          for (ordinal_type j = 0; j < ndof; ++j) {
            const double valIJ = matDataHost(faceOffset + f, ort, i, j);
            const double valJI = matDataHost(faceOffset + f, ort, j, i);
            if (std::abs(valIJ) > 1e-12 && std::abs(valIJ - 1.0) > 1e-12) {
              success = false;
              out << "FAILURE (" << basis.getName() << " face " << f << " ort " << ort
                  << "): non-binary face coeff " << valIJ << "\n";
            }
            rowSum += valIJ;
            colSum += valJI;
          }
          if (std::abs(rowSum - 1.0) > 1e-12 || std::abs(colSum - 1.0) > 1e-12) {
            success = false;
            out << "FAILURE (" << basis.getName() << " face " << f << " ort " << ort
                << "): rowSum=" << rowSum << ", colSum=" << colSum << "\n";
          }
        }
        if (checkIdentityAtOrt0 && ort == 0) {
          for (ordinal_type i = 0; i < ndof; ++i) {
            if (std::abs(matDataHost(faceOffset + f, 0, i, i) - 1.0) > 1e-12) {
              success = false;
              out << "FAILURE (" << basis.getName() << " face " << f
                  << "): ort 0 is not identity at i=" << i << "\n";
            }
          }
        }
      }
    }

    ots::clearCoeffMatrix();
  }

  // ============================================================================
  // 6. Subcell Orientation Permutation Matrices (TRI, QUAD, TET, HEX, WEDGE)
  // ============================================================================
  TEUCHOS_UNIT_TEST( BernsteinBasis, OrientationPermutationMatrices )
  {
    for (int p = 1; p <= 5; ++p) {
      BernsteinBasis_HGRAD_TRI<DeviceType>  tri(p);
      BernsteinBasis_HGRAD_QUAD<DeviceType> quad(p);
      BernsteinBasis_HGRAD_TET<DeviceType>  tet(p);

      TEST_EQUALITY(tri.requireOrientation(), (p > 2));
      TEST_EQUALITY(tet.requireOrientation(), (p > 2));

      verifyOrientationPermutationMatrices(tri,  true,  out, success);
      verifyOrientationPermutationMatrices(quad, false, out, success);
      verifyOrientationPermutationMatrices(tet,  true,  out, success);
    }

    for (int p = 1; p <= 4; ++p) {
      BernsteinBasis_HGRAD_HEX<DeviceType>   hex(p);
      BernsteinBasis_HGRAD_WEDGE<DeviceType> wedge(p);

      verifyOrientationPermutationMatrices(hex,   false, out, success);
      verifyOrientationPermutationMatrices(wedge, false, out, success);
    }
  }

  // ============================================================================
  // 7. Two-Element Oriented Patch C^0 Continuity on TET and HEX (All Face Orts)
  // ============================================================================
  TEUCHOS_UNIT_TEST( BernsteinBasis, TwoElementOrientedPatchContinuity_TET_And_HEX )
  {
    using ct  = CellTools<HostSpaceType>;
    using ots = OrientationTools<DeviceType>;

    const double tol = 1e-11;
    Teuchos::LAPACK<int, double> lapack;

    // ------------------------------------------------------------------------
    // Part A: Two Tetrahedra sharing a face across all 6 face vertex permutations
    // ------------------------------------------------------------------------
    {
      const shards::CellTopology tetTopo(shards::getCellTopologyData<shards::Tetrahedron<4>>());
      const double physVerts[5][3] = {
        { 0.0,  0.0, -1.0}, // 0 (apex of Tet 0)
        { 1.0,  0.1,  0.0}, // 1 (shared face)
        {-0.2,  1.1,  0.1}, // 2 (shared face)
        { 0.1, -0.3,  0.9}, // 3 (shared face)
        { 0.8,  0.7,  1.5}  // 4 (apex of Tet 1)
      };

      // Test all 6 permutations of shared face {1, 2, 3} on Tet 1
      // (while adjusting apex/ordering so Jacobian > 0 for valid right-handed tetrahedra,
      //  or testing all 3 cyclic rotations of Tet 1's base face (1,2,3) which preserve positive volume!)
      // Even better: for ANY permutation of {1,2,3}, if det(J) < 0 on Tet 1, swapping nodes (0,1) vs keeping
      // Tet 0 permuted gives all 6 relative orientations between Tet 0 and Tet 1!
      const int perms[6][4] = {
        // Permute Tet 0's vertices so the shared face {1,2,3} appears with positive Jacobian in all 6 ways:
        // Wait: in shards::Tetrahedron<4>, vertices (v0, v1, v2, v3) have positive volume when
        // det([v1-v0, v2-v0, v3-v0]) > 0.
        // For apex 0 = (0,0,-1) and shared face {1,2,3}:
        //   v1-v0 = (1, 0.1, 1), v2-v0 = (-0.2, 1.1, 1.1), v3-v0 = (0.1, -0.3, 1.9) -> det > 0!
        // Cyclic rotations of (1,2,3) keep det > 0; odd permutations of (1,2,3) with apex at slot 0 vs slot 3:
        // Notice swapping two vertices AND moving apex from slot 3 to slot 2 flips sign twice -> det > 0!
        {1, 2, 3, 0}, // check sign dynamically below and swap apex slot if needed!
        {2, 3, 1, 0},
        {3, 1, 2, 0},
        {1, 3, 2, 0},
        {3, 2, 1, 0},
        {2, 1, 3, 0}
      };

      for (int polyOrder : {3, 4}) {
        BernsteinBasis_HGRAD_TET<DeviceType> basis(polyOrder);
        const int numFields = basis.getCardinality();

        const int nPts = PointTools::getLatticeSize(tetTopo, polyOrder + 1, 0);
        Kokkos::DynRankView<double, HostSpaceType> refPts("refPts", nPts, 3);
        PointTools::getLattice(refPts, tetTopo, polyOrder + 1, 0, POINTTYPE_EQUISPACED);
        Kokkos::DynRankView<double, DeviceType> dRefPts("dRefPts", nPts, 3);
        Kokkos::deep_copy(dRefPts, refPts);

        Kokkos::DynRankView<double, DeviceType> refBasisVals("refBasisVals", numFields, nPts);
        basis.getValues(refBasisVals, dRefPts, OPERATOR_VALUE);

        for (int pIdx = 0; pIdx < 6; ++pIdx) {
          // Construct positive-Jacobian Tet 0 and Tet 1 sharing global vertices {1, 2, 3}
          int tet0Nodes[4] = {perms[pIdx][0], perms[pIdx][1], perms[pIdx][2], perms[pIdx][3]};
          // Check signed volume of tet0Nodes; if negative, swap slot 2 and slot 3 (putting apex at slot 2)
          auto signedVol = [&](const int n[4]) {
            const double ax = physVerts[n[1]][0] - physVerts[n[0]][0];
            const double ay = physVerts[n[1]][1] - physVerts[n[0]][1];
            const double az = physVerts[n[1]][2] - physVerts[n[0]][2];
            const double bx = physVerts[n[2]][0] - physVerts[n[0]][0];
            const double by = physVerts[n[2]][1] - physVerts[n[0]][1];
            const double bz = physVerts[n[2]][2] - physVerts[n[0]][2];
            const double cx = physVerts[n[3]][0] - physVerts[n[0]][0];
            const double cy = physVerts[n[3]][1] - physVerts[n[0]][1];
            const double cz = physVerts[n[3]][2] - physVerts[n[0]][2];
            return ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
          };
          if (signedVol(tet0Nodes) < 0.0) {
            std::swap(tet0Nodes[2], tet0Nodes[3]);
          }
          int tet1Nodes[4] = {1, 3, 2, 4};
          if (signedVol(tet1Nodes) < 0.0) {
            std::swap(tet1Nodes[1], tet1Nodes[2]);
          }

          Kokkos::DynRankView<ordinal_type, HostSpaceType> elemNodes("elemNodes", 2, 4);
          Kokkos::DynRankView<double, HostSpaceType> cellWorkset("cellWorkset", 2, 4, 3);
          for (int v = 0; v < 4; ++v) {
            elemNodes(0, v) = tet0Nodes[v];
            elemNodes(1, v) = tet1Nodes[v];
            for (int d = 0; d < 3; ++d) {
              cellWorkset(0, v, d) = physVerts[tet0Nodes[v]][d];
              cellWorkset(1, v, d) = physVerts[tet1Nodes[v]][d];
            }
          }

          Kokkos::DynRankView<Orientation, HostSpaceType> hOrts("hOrts", 2);
          ots::getOrientation(hOrts, elemNodes, tetTopo);
          Kokkos::DynRankView<Orientation, DeviceType> dOrts("dOrts", 2);
          Kokkos::deep_copy(dOrts, hOrts);

          Kokkos::DynRankView<double, DeviceType> cellBasisVals("cellBasisVals", 2, numFields, nPts);
          auto hRefBasisVals = getHostCopy(refBasisVals);
          auto hCellBasisIn  = Kokkos::create_mirror_view(cellBasisVals);
          for (int c = 0; c < 2; ++c)
            for (int f = 0; f < numFields; ++f)
              for (int q = 0; q < nPts; ++q)
                hCellBasisIn(c, f, q) = hRefBasisVals(f, q);
          Kokkos::deep_copy(cellBasisVals, hCellBasisIn);

          Kokkos::DynRankView<double, DeviceType> orientedVals("orientedVals", 2, numFields, nPts);
          ots::modifyBasisByOrientation(orientedVals, cellBasisVals, dOrts, &basis);
          auto hOriented = getHostCopy(orientedVals);

          // Map refPts to physical space on each cell and evaluate exact degree-polyOrder polynomial
          Kokkos::DynRankView<double, HostSpaceType> physPts("physPts", 2, nPts, 3);
          ct::mapToPhysicalFrame(physPts, refPts, cellWorkset, tetTopo);

          auto polyFun = [&](double x, double y, double z) {
            double val = (1.0 + 0.7 * x - 0.4 * y + 0.5 * z) * (0.3 - x + 1.2 * y + 0.8 * z)
                       * (-0.6 + 0.5 * x + 0.3 * y - 0.9 * z);
            if (polyOrder >= 4) val *= (0.9 + 0.2 * x - 0.7 * y + 0.4 * z);
            return val;
          };

          double coeffs[2][64] = {};
          for (int c = 0; c < 2; ++c) {
            std::vector<double> A(nPts * numFields);
            std::vector<double> b(nPts);
            for (int q = 0; q < nPts; ++q) {
              b[q] = polyFun(physPts(c, q, 0), physPts(c, q, 1), physPts(c, q, 2));
              for (int f = 0; f < numFields; ++f) {
                A[q + f * nPts] = hOriented(c, f, q);
              }
            }
            int info = 0;
            int lwork = 4 * (nPts + numFields);
            std::vector<double> work(lwork);
            lapack.GELS('N', nPts, numFields, 1, A.data(), nPts, b.data(), nPts, work.data(), lwork, &info);
            TEST_EQUALITY(info, 0);
            for (int f = 0; f < numFields; ++f) coeffs[c][f] = b[f];

            // Verify exact polynomial reproduction on Cell c
            for (int q = 0; q < nPts; ++q) {
              double uApprox = 0.0;
              for (int f = 0; f < numFields; ++f) uApprox += coeffs[c][f] * hOriented(c, f, q);
              const double uExact = polyFun(physPts(c, q, 0), physPts(c, q, 1), physPts(c, q, 2));
              TEST_ASSERT(std::abs(uApprox - uExact) <= tol);
            }
          }

          // Check shared vertex DOFs ({1, 2, 3}), shared edge DOFs, and shared face DOFs!
          for (int gv : {1, 2, 3}) {
            int lv0 = -1, lv1 = -1;
            for (int v = 0; v < 4; ++v) {
              if (tet0Nodes[v] == gv) lv0 = v;
              if (tet1Nodes[v] == gv) lv1 = v;
            }
            TEST_ASSERT(std::abs(coeffs[0][basis.getDofOrdinal(0, lv0, 0)] -
                                 coeffs[1][basis.getDofOrdinal(0, lv1, 0)]) <= tol);
          }

          const int sharedEdgePairs[3][2] = {{1, 2}, {2, 3}, {1, 3}};
          const int nEdgeDofs = basis.getDofCount(1, 0);
          for (int ie = 0; ie < 3; ++ie) {
            int e0 = -1, e1 = -1;
            for (int e = 0; e < 6; ++e) {
              int a0 = tet0Nodes[tetTopo.getNodeMap(1, e, 0)], b0 = tet0Nodes[tetTopo.getNodeMap(1, e, 1)];
              if ((a0 == sharedEdgePairs[ie][0] && b0 == sharedEdgePairs[ie][1]) ||
                  (a0 == sharedEdgePairs[ie][1] && b0 == sharedEdgePairs[ie][0])) e0 = e;
              int a1 = tet1Nodes[tetTopo.getNodeMap(1, e, 0)], b1 = tet1Nodes[tetTopo.getNodeMap(1, e, 1)];
              if ((a1 == sharedEdgePairs[ie][0] && b1 == sharedEdgePairs[ie][1]) ||
                  (a1 == sharedEdgePairs[ie][1] && b1 == sharedEdgePairs[ie][0])) e1 = e;
            }
            TEST_ASSERT(e0 >= 0 && e1 >= 0);
            for (int j = 0; j < nEdgeDofs; ++j) {
              TEST_ASSERT(std::abs(coeffs[0][basis.getDofOrdinal(1, e0, j)] -
                                   coeffs[1][basis.getDofOrdinal(1, e1, j)]) <= tol);
            }
          }

          // Shared face: find the face on Tet 0 and Tet 1 whose vertices are {1, 2, 3}
          int f0 = -1, f1 = -1;
          for (int s = 0; s < 4; ++s) {
            std::array<int, 3> fv0 = {
              tet0Nodes[tetTopo.getNodeMap(2, s, 0)],
              tet0Nodes[tetTopo.getNodeMap(2, s, 1)],
              tet0Nodes[tetTopo.getNodeMap(2, s, 2)]
            };
            std::sort(fv0.begin(), fv0.end());
            if (fv0 == std::array<int, 3>{1, 2, 3}) f0 = s;

            std::array<int, 3> fv1 = {
              tet1Nodes[tetTopo.getNodeMap(2, s, 0)],
              tet1Nodes[tetTopo.getNodeMap(2, s, 1)],
              tet1Nodes[tetTopo.getNodeMap(2, s, 2)]
            };
            std::sort(fv1.begin(), fv1.end());
            if (fv1 == std::array<int, 3>{1, 2, 3}) f1 = s;
          }
          TEST_ASSERT(f0 >= 0 && f1 >= 0);
          const int nFaceDofs = basis.getDofCount(2, f0);
          for (int j = 0; j < nFaceDofs; ++j) {
            if (std::abs(coeffs[0][basis.getDofOrdinal(2, f0, j)] -
                         coeffs[1][basis.getDofOrdinal(2, f1, j)]) > tol) {
              success = false;
              out << "FAILURE (TET p=" << polyOrder << " perm=" << pIdx
                  << "): shared face DOF " << j << " mismatch: "
                  << coeffs[0][basis.getDofOrdinal(2, f0, j)] << " vs "
                  << coeffs[1][basis.getDofOrdinal(2, f1, j)] << "\n";
            }
          }
        }
        ots::clearCoeffMatrix();
      }
    }

    // ------------------------------------------------------------------------
    // Part B: Two Hexahedra sharing a quad face under rotations & reflections
    // ------------------------------------------------------------------------
    {
      const shards::CellTopology hexTopo(shards::getCellTopologyData<shards::Hexahedron<8>>());
      const double physVerts[12][3] = {
        {-1,-1,-1}, {1,-1,-1}, {1,1,-1}, {-1,1,-1},
        {-1,-1, 0}, {1,-1, 0}, {1,1, 0}, {-1,1, 0},
        {-1,-1, 1}, {1,-1, 1}, {1,1, 1}, {-1,1, 1}
      };
      // 4 rotations of Hex 0 around z-axis + 4 inverted-z / reflected configurations of Hex 0
      // (all right-handed positive-volume hexes sharing face {4, 5, 6, 7}!)
      const int hex0Configs[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7}, // rot 0 (shared top face {4,5,6,7})
        {1, 2, 3, 0, 5, 6, 7, 4}, // rot 90
        {2, 3, 0, 1, 6, 7, 4, 5}, // rot 180
        {3, 0, 1, 2, 7, 4, 5, 6}, // rot 270
        {4, 7, 6, 5, 0, 3, 2, 1}, // shared bottom face (reflected!)
        {7, 6, 5, 4, 3, 2, 1, 0}, // shared bottom face rot 90
        {6, 5, 4, 7, 2, 1, 0, 3}, // shared bottom face rot 180
        {5, 4, 7, 6, 1, 0, 3, 2}  // shared bottom face rot 270
      };
      const int hex1Nodes[8] = {4, 5, 6, 7, 8, 9, 10, 11};

      const int polyOrder = 3;
      BernsteinBasis_HGRAD_HEX<DeviceType> basis(polyOrder);
      const int numFields = basis.getCardinality();

      const int nPts = PointTools::getLatticeSize(hexTopo, polyOrder + 1, 0);
      Kokkos::DynRankView<double, HostSpaceType> refPts("refPts", nPts, 3);
      PointTools::getLattice(refPts, hexTopo, polyOrder + 1, 0, POINTTYPE_EQUISPACED);
      Kokkos::DynRankView<double, DeviceType> dRefPts("dRefPts", nPts, 3);
      Kokkos::deep_copy(dRefPts, refPts);

      Kokkos::DynRankView<double, DeviceType> refBasisVals("refBasisVals", numFields, nPts);
      basis.getValues(refBasisVals, dRefPts, OPERATOR_VALUE);
      auto hRefBasisVals = getHostCopy(refBasisVals);

      for (int cfg = 0; cfg < 8; ++cfg) {
        Kokkos::DynRankView<ordinal_type, HostSpaceType> elemNodes("elemNodes", 2, 8);
        Kokkos::DynRankView<double, HostSpaceType> cellWorkset("cellWorkset", 2, 8, 3);
        for (int v = 0; v < 8; ++v) {
          elemNodes(0, v) = hex0Configs[cfg][v];
          elemNodes(1, v) = hex1Nodes[v];
          for (int d = 0; d < 3; ++d) {
            cellWorkset(0, v, d) = physVerts[hex0Configs[cfg][v]][d];
            cellWorkset(1, v, d) = physVerts[hex1Nodes[v]][d];
          }
        }

        Kokkos::DynRankView<Orientation, HostSpaceType> hOrts("hOrts", 2);
        ots::getOrientation(hOrts, elemNodes, hexTopo);
        Kokkos::DynRankView<Orientation, DeviceType> dOrts("dOrts", 2);
        Kokkos::deep_copy(dOrts, hOrts);

        Kokkos::DynRankView<double, DeviceType> cellBasisVals("cellBasisVals", 2, numFields, nPts);
        auto hCellBasisIn = Kokkos::create_mirror_view(cellBasisVals);
        for (int c = 0; c < 2; ++c)
          for (int f = 0; f < numFields; ++f)
            for (int q = 0; q < nPts; ++q)
              hCellBasisIn(c, f, q) = hRefBasisVals(f, q);
        Kokkos::deep_copy(cellBasisVals, hCellBasisIn);

        Kokkos::DynRankView<double, DeviceType> orientedVals("orientedVals", 2, numFields, nPts);
        ots::modifyBasisByOrientation(orientedVals, cellBasisVals, dOrts, &basis);
        auto hOriented = getHostCopy(orientedVals);

        Kokkos::DynRankView<double, HostSpaceType> physPts("physPts", 2, nPts, 3);
        ct::mapToPhysicalFrame(physPts, refPts, cellWorkset, hexTopo);

        auto polyFun = [](double x, double y, double z) {
          return (1.0 + 0.6 * x - 0.3 * x * x + 0.4 * x * x * x)
               * (0.8 - 0.5 * y + 0.7 * y * y - 0.2 * y * y * y)
               * (0.5 + 0.9 * z - 0.4 * z * z + 0.3 * z * z * z);
        };

        std::vector<double> coeffs[2];
        for (int c = 0; c < 2; ++c) {
          coeffs[c].resize(numFields);
          std::vector<double> A(nPts * numFields);
          std::vector<double> b(nPts);
          for (int q = 0; q < nPts; ++q) {
            b[q] = polyFun(physPts(c, q, 0), physPts(c, q, 1), physPts(c, q, 2));
            for (int f = 0; f < numFields; ++f) {
              A[q + f * nPts] = hOriented(c, f, q);
            }
          }
          int info = 0;
          int lwork = 4 * (nPts + numFields);
          std::vector<double> work(lwork);
          lapack.GELS('N', nPts, numFields, 1, A.data(), nPts, b.data(), nPts, work.data(), lwork, &info);
          TEST_EQUALITY(info, 0);
          for (int f = 0; f < numFields; ++f) coeffs[c][f] = b[f];
        }

        // Compare shared face DOFs ({4, 5, 6, 7})
        const int f0 = (cfg < 4) ? 5 : 4; // face 5 is top (4,5,6,7), face 4 is bottom (0,3,2,1) in Shards Hex
        const int f1 = 4;                 // face 4 of Hex 1 is {4, 7, 6, 5} -> {4, 5, 6, 7}
        const int nFaceDofs = basis.getDofCount(2, f0);
        for (int j = 0; j < nFaceDofs; ++j) {
          if (std::abs(coeffs[0][basis.getDofOrdinal(2, f0, j)] -
                       coeffs[1][basis.getDofOrdinal(2, f1, j)]) > tol) {
            success = false;
            out << "FAILURE (HEX cfg=" << cfg << "): shared face DOF " << j << " mismatch\n";
          }
        }
      }
      ots::clearCoeffMatrix();
    }
  }

  // ============================================================================
  // 8. BasisFamily Factories, Derived H(curl)/H(div)/H(vol), and Hypercubes
  // ============================================================================
  TEUCHOS_UNIT_TEST( BernsteinBasis, FamilyFactories_And_DerivedDeRham )
  {
    using BFamily   = BernsteinBasisFamily<DeviceType>;
    using DGBFamily = DGBernsteinBasisFamily<DeviceType>;
    using HFamily   = HierarchicalBasisFamily<DeviceType>;

    // Factory dispatch checks via getBasis
    {
      const shards::CellTopology lineTopo(shards::getCellTopologyData<shards::Line<2>>());
      const shards::CellTopology triTopo(shards::getCellTopologyData<shards::Triangle<3>>());
      const shards::CellTopology quadTopo(shards::getCellTopologyData<shards::Quadrilateral<4>>());
      const shards::CellTopology tetTopo(shards::getCellTopologyData<shards::Tetrahedron<4>>());
      const shards::CellTopology hexTopo(shards::getCellTopologyData<shards::Hexahedron<8>>());
      auto bLine      = getBasis<BFamily>(lineTopo,  FUNCTION_SPACE_HGRAD, 3);
      auto bTri       = getBasis<BFamily>(triTopo,   FUNCTION_SPACE_HGRAD, 3);
      auto bQuad      = getBasis<BFamily>(quadTopo,  FUNCTION_SPACE_HGRAD, 3);
      auto bTet       = getBasis<BFamily>(tetTopo,   FUNCTION_SPACE_HGRAD, 3);
      auto bHex       = getBasis<BFamily>(hexTopo,   FUNCTION_SPACE_HGRAD, 3);
      auto bWedge     = getWedgeBasis<BFamily>(FUNCTION_SPACE_HGRAD, 3);
      auto bWedgeHvol = getWedgeBasis<BFamily>(FUNCTION_SPACE_HVOL,  3);

      TEST_EQUALITY(bLine->getCardinality(),      4);
      TEST_EQUALITY(bTri->getCardinality(),       10);
      TEST_EQUALITY(bQuad->getCardinality(),      16);
      TEST_EQUALITY(bTet->getCardinality(),       20);
      TEST_EQUALITY(bHex->getCardinality(),       64);
      TEST_EQUALITY(bWedge->getCardinality(),     40);
      TEST_EQUALITY(bWedgeHvol->getCardinality(), 40);

      auto dgTri  = getBasis<DGBFamily>(triTopo,  FUNCTION_SPACE_HGRAD, 3);
      auto dgTet  = getBasis<DGBFamily>(tetTopo,  FUNCTION_SPACE_HGRAD, 3);
      auto dgQuad = getBasis<DGBFamily>(quadTopo, FUNCTION_SPACE_HGRAD, 3);
      TEST_EQUALITY(dgTri->getDofCount(2, 0),  10);
      TEST_EQUALITY(dgTet->getDofCount(3, 0),  20);
      TEST_EQUALITY(dgQuad->getDofCount(2, 0), 16);
    }

    // Derived H(curl), H(div), H(vol) on QUAD and HEX vs HierarchicalBasisFamily
    {
      const double relTol = 1e-11;
      const double absTol = 1e-11;

      for (int p = 1; p <= 3; ++p) {
        typename BFamily::HCURL_QUAD bHcurlQuad(p);
        typename HFamily::HCURL_QUAD hHcurlQuad(p);
        BasisEquivalenceHelpers::testBasisEquivalence<DeviceType>(
          bHcurlQuad, hHcurlQuad, {OPERATOR_VALUE, OPERATOR_CURL}, relTol, absTol, out, success);

        typename BFamily::HDIV_QUAD bHdivQuad(p);
        typename HFamily::HDIV_QUAD hHdivQuad(p);
        BasisEquivalenceHelpers::testBasisEquivalence<DeviceType>(
          bHdivQuad, hHdivQuad, {OPERATOR_VALUE, OPERATOR_DIV}, relTol, absTol, out, success);
      }

      for (int p = 0; p <= 3; ++p) {
        typename BFamily::HVOL_QUAD bHvolQuad(p);
        typename HFamily::HVOL_QUAD hHvolQuad(p);
        BasisEquivalenceHelpers::testBasisEquivalence<DeviceType>(
          bHvolQuad, hHvolQuad, {OPERATOR_VALUE}, relTol, absTol, out, success);
      }

      for (int p = 1; p <= 2; ++p) {
        typename BFamily::HCURL_HEX bHcurlHex(p);
        typename HFamily::HCURL_HEX hHcurlHex(p);
        BasisEquivalenceHelpers::testBasisEquivalence<DeviceType>(
          bHcurlHex, hHcurlHex, {OPERATOR_VALUE, OPERATOR_CURL}, relTol, absTol, out, success);

        typename BFamily::HDIV_HEX bHdivHex(p);
        typename HFamily::HDIV_HEX hHdivHex(p);
        BasisEquivalenceHelpers::testBasisEquivalence<DeviceType>(
          bHdivHex, hHdivHex, {OPERATOR_VALUE, OPERATOR_DIV}, relTol, absTol, out, success);
      }

      for (int p = 0; p <= 2; ++p) {
        typename BFamily::HVOL_HEX bHvolHex(p);
        typename HFamily::HVOL_HEX hHvolHex(p);
        BasisEquivalenceHelpers::testBasisEquivalence<DeviceType>(
          bHvolHex, hHvolHex, {OPERATOR_VALUE}, relTol, absTol, out, success);
      }
    }
  }

} // anonymous namespace
