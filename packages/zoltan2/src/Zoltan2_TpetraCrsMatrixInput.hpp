// @HEADER
// ***********************************************************************
//
//                Copyright message goes here.   TODO
//
// ***********************************************************************
// @HEADER

/*! \file Zoltan2_TpetraCrsMatrixInput.hpp

    \brief An input adapter for a Tpetra::CrsMatrix.
*/

#ifndef _ZOLTAN2_TPETRACRSMATRIXINPUT_HPP_
#define _ZOLTAN2_TPETRACRSMATRIXINPUT_HPP_

#include <Tpetra_CrsMatrix.hpp>
#include <Zoltan2_XpetraCrsMatrixInput.hpp>

namespace Zoltan2 {

/*! Zoltan2::TpetraCrsMatrixInput
    \brief Provides access for Zoltan2 to Tpetra::CrsMatrix data. 
*/

CONSISTENT_CLASS_TEMPLATE_LINE
class TpetraCrsMatrixInput : 
  public XpetraCrsMatrixInput<CONSISTENT_TEMPLATE_PARAMS>{
private:

  typedef Tpetra::CrsMatrix<Scalar, LNO, GNO, Node> crsMatrix;
  RCP<const crsMatrix > _inmatrix;

public:
  std::string inputAdapterName()const {return std::string("TpetraCrsMatrix");}

  ~TpetraCrsMatrixInput() { }

  /*! Constructor 
   */
  TpetraCrsMatrixInput(const RCP<const crsMatrix> matrix):
    XpetraCrsMatrixInput<CONSISTENT_TEMPLATE_PARAMS>(
      Teuchos::rcp(new Xpetra::TpetraCrsMatrix<Scalar, LNO, GNO, Node>(
        Teuchos::rcp_const_cast<crsMatrix>(matrix))))
    
  {
    _inmatrix = matrix;
  }

  /*! Access to matrix that instantiated adapter
   */

  RCP<const crsMatrix> getMatrix() const
  { 
    return _inmatrix;
  }
};
} // namespace

#endif
