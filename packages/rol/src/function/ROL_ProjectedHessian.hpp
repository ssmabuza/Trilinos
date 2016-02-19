// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER

#ifndef ROL_PROJECTEDHESSIAN_H
#define ROL_PROJECTEDHESSIAN_H

#include "ROL_LinearOperator.hpp"
#include "ROL_Objective.hpp"
#include "ROL_BoundConstraint.hpp"
#include "ROL_Types.hpp"

namespace ROL {

template <class Real>
class ProjectedHessian : public LinearOperator<Real> {
private:

  const Teuchos::RCP<Objective<Real> > obj_;
  const Teuchos::RCP<BoundConstraint<Real> > bnd_;
  const Teuchos::RCP<Vector<Real> > x_;
  const Teuchos::RCP<Vector<Real> > g_;
  Teuchos::RCP<Vector<Real> > v_;
  Real eps_;

public:

  ProjectedHessian(const Teuchos::RCP<Objective<Real> > &obj,
                   const Teuchos::RCP<BoundConstraint<Real> > &bnd, 
                   const Teuchos::RCP<Vector<Real> > &x,
                   const Teuchos::RCP<Vector<Real> > &g,
                   Real eps = 0.0 ) 
    : obj_(obj), bnd_(bnd), x_(x), g_(g), eps_(eps) {
    v_ = x_->clone();
  }

  /** \brief Apply Hessian.

      This function applies the Hessian to a vector.
      @param[out]         Hv  is the output vector.
      @param[in]          v   is the input vector.
      @param[in]          tol is a tolerance for inexact Hessian application.
  */
  void apply( Vector<Real> &Hv, const Vector<Real> &v, Real &tol ) const {
    v_->set(v);
    bnd_->pruneActive(*v_,*g_,*x_,eps_);
    obj_->hessVec(Hv,*v_,*x_,tol);
    bnd_->pruneActive(Hv,*g_,*x_,eps_);
    v_->set(v);
    bnd_->pruneInactive(*v_,*g_,*x_,eps_);
    Hv.plus(v_->dual());
  }

  /** \brief Apply inverse Hessian.

      This function applies the inverse of the Hessian to a vector.
      @param[out]         Hv  is the output vector.
      @param[in]          v   is the input vector.
      @param[in]          tol is a tolerance for inexact Hessian application.
  */
  void applyInverse( Vector<Real> &Hv, const Vector<Real> &v, Real &tol ) const {
    v_->set(v);
    bnd_->pruneActive(*v_,*g_,*x_,eps_);
    obj_->invHessVec(Hv,*v_,*x_,tol);
    bnd_->pruneActive(Hv,*g_,*x_,eps_);
    v_->set(v);
    bnd_->pruneInactive(*v_,*g_,*x_,eps_);
    Hv.plus(v_->dual());
  }

}; // class Hessian

} // namespace ROL

#endif
