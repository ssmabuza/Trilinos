TRILINOS_DRIVER_SETUP()

TRILINOS_DRIVER_ADD_DASHBOARD(
  DEBUG_CUDA_NOUVM_NODEPRECATED_DEV
  ctest_linux_nightly_mpi_debug_nouvm_ascicgpu031_crs.cmake
  CTEST_INSTALLER_TYPE debug
  RUN_SERIAL
  TIMEOUT_MINUTES 420
  )

TRILINOS_DRIVER_ADD_DASHBOARD(
  DEBUG_CUDA_UVM_NODEPRECATED_DEV
  ctest_linux_nightly_mpi_debug_uvm_nodeprecated_ascicgpu031_crs.cmake
  CTEST_INSTALLER_TYPE debug
  RUN_SERIAL
  TIMEOUT_MINUTES 420
  )

TRILINOS_DRIVER_ADD_DASHBOARD(
  DEBUG_CUDA_UVM_DEPRECATED_DEV
  ctest_linux_nightly_mpi_debug_uvm_deprecated_ascicgpu031_crs.cmake
  CTEST_INSTALLER_TYPE debug
  RUN_SERIAL
  TIMEOUT_MINUTES 420
  )

TRILINOS_ADD_REQUIRED_CMAKE_INSTALLS()
