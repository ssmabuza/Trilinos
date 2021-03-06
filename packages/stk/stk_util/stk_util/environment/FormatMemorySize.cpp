// Copyright 2002 - 2008, 2010, 2011 National Technology Engineering
// Solutions of Sandia, LLC (NTESS). Under the terms of Contract
// DE-NA0003525 with NTESS, the U.S. Government retains certain rights
// in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
// 
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
// 
//     * Neither the name of NTESS nor the names of its contributors
//       may be used to endorse or promote products derived from this
//       software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 

#include "stk_util/environment/FormatMemorySize.hpp"
#include <iomanip>  // for operator<<, setprecision, setw
#include <sstream>  // for basic_ostream, operator<<, basic_ostream<>::__ostream_type, stringstream
#include <string>   // for string

namespace stk {

std::string format_memory_size(double size)
{
  std::string result;

  const double kb = 1024.0;
  const double mb = kb * kb;
  if (size < 0.0) {
    return "bad value";
  } else {
    double numMB = size / mb;
    std::stringstream memOut;

    if(numMB == 0.0) {
      //
      //  Short format for exact zero
      //
      memOut << std::setw(6) << std::fixed << std::setprecision(1) << numMB <<" MB";
    } else if(numMB < 1.0) {
      //
      //  Go to small number scientific notation
      //
      memOut << std::setw(6) << std::scientific << std::setprecision(5) << numMB <<" MB";
    } else if (numMB <= 99999.9) {
      //
      //  For most any common numbers output value in MB fixed format
      //
      memOut << std::setw(6) << std::fixed << std::setprecision(1) << numMB <<" MB";
    } else {
      //
      //  For huge values go back to scientific notation
      //
      memOut << std::setw(6) << std::scientific << std::setprecision(5) << numMB <<" MB";
    }
    return memOut.str();
  }
}

std::string format_memory_size(MemorySize size) {
  return(format_memory_size((double)size));
}

} // namespace stk

