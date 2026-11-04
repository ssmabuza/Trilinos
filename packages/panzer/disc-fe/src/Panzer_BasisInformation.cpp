// @HEADER
// *****************************************************************************
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//
// Copyright 2011 NTESS and the Panzer contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include "Panzer_BasisInformation.hpp"

#include "Teuchos_Assert.hpp"

#include <sstream>

namespace panzer {

BasisInformation::
BasisInformation(const std::string & in_basis_type,
	         const int in_basis_order,
	         const shards::CellTopology & cell_topo) :
  topology_(cell_topo)
{
  // Support for deprecated basis descriptions
  std::string basis_type = in_basis_type;
  int basis_order = in_basis_order;

  if (basis_type=="Q1" || basis_type=="T1") {
    basis_type = "HGrad";
    basis_order = 1;
  }
  else if (basis_type == "Q2" || basis_type=="T2") {
    basis_type = "HGrad";
    basis_order = 2;
  }
  else if (basis_type == "TEdge1" || basis_type=="QEdge1") {
    basis_type = "HCurl";
    basis_order = 1;
  }
  else if(basis_type == "Const") {
    basis_type = "Const";
    basis_order = 0;
  }
  // End deprecated basis support

  basis_type_ = basis_type;
  basis_order_ = basis_order;

  if(  basis_type_ == "HGrad" || basis_type_ == "HGrad_Bernstein" ||
       basis_type_ == "Bernstein" || basis_type_ == "HGrad:Bernstein")
    element_space_ = HGRAD;
  else if(basis_type_=="HCurl" || basis_type_=="HCurl_Bernstein" ||
          basis_type_=="HCurl:Bernstein")
    element_space_ = HCURL;
  else if(basis_type_=="HDiv" || basis_type_=="HDiv_Bernstein" ||
          basis_type_=="HDiv:Bernstein")
    element_space_ = HDIV;
  else if(basis_type_=="Const")
    element_space_ = CONST;
  else if(basis_type_=="HVol" || basis_type_=="HVol_Bernstein" ||
          basis_type_=="HVol:Bernstein" || basis_type_=="DG_Bernstein")
    element_space_ = HVOL;
  else { TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,
				    "BasisInformation::initializeIntrospection - Invalid basis name \"" 
				    << basis_type_ << "\""); }
}

}
