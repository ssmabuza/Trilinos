# Assert this script is sourced, not run!
called=$_
if [ "$called" == "$0" ] ; then
  echo "This script '$0' is being called.  Instead, it must be sourced!"
  exit 1
fi

# Assert that ATDM_CONFIG_BUILD_NAME is set!
if [ -z "$ATDM_CONFIG_BUILD_NAME" ] ; then
  echo "Error, must set ATDM_CONFIG_BUILD_NAME in env!"
  return
fi

echo "Setting compiler and build options for buld name '${ATDM_CONFIG_BUILD_NAME}'"

# Set the defaults
export ATDM_CONFIG_COMPILER=DEFAULT
export ATDM_CONFIG_KOKKOS_ARCH=DEFAULT
export ATDM_CONFIG_BUILD_TYPE=DEBUG
export ATDM_CONFIG_USE_OPENMP=OFF
export ATDM_CONFIG_USE_CUDA=OFF
export ATDM_CONFIG_USE_PTHREADS=OFF

# Process system custom build logic
export ATDM_CONFIG_CUSTOM_COMPILER_SET=0
if [ -e $ATDM_CONFIG_SCRIPT_DIR/$ATDM_CONFIG_KNOWN_SYSTEM_NAME/custom_builds.sh ]; then
  source $ATDM_CONFIG_SCRIPT_DIR/$ATDM_CONFIG_KNOWN_SYSTEM_NAME/custom_builds.sh
fi

# NOTE: Currently only the specialization of ATDM_CONFIG_COMPILER from
# custom_builds.sh is supported below.  ToDo: Add support for customizing
# other things as needed.

# Set the compiler
if [[ "${ATDM_CONFIG_COMPILER}" != "DEFAULT" ]] ; then
  # Custom compile already set
  export ATDM_CONFIG_CUSTOM_COMPILER_SET=1
elif [[ $ATDM_CONFIG_BUILD_NAME == *"default" ]]; then
  export ATDM_CONFIG_COMPILER=DEFAULT
elif [[ $ATDM_CONFIG_BUILD_NAME == *"gnu-4.9.3"* ]]; then
  export ATDM_CONFIG_COMPILER=GNU-4.9.3
elif [[ $ATDM_CONFIG_BUILD_NAME == *"gnu-7.2.0"* ]]; then
  export ATDM_CONFIG_COMPILER=GNU-7.2.0
elif [[ $ATDM_CONFIG_BUILD_NAME == *"gnu"* ]]; then
  export ATDM_CONFIG_COMPILER=GNU
elif [[ $ATDM_CONFIG_BUILD_NAME == *"intel"* ]]; then
 export ATDM_CONFIG_COMPILER=INTEL
elif [[ $ATDM_CONFIG_BUILD_NAME == *"cuda-8.0"* ]]; then
  export ATDM_CONFIG_COMPILER=CUDA-8.0
elif [[ $ATDM_CONFIG_BUILD_NAME == *"cuda-9.0"* ]]; then
  export ATDM_CONFIG_COMPILER=CUDA-9.0
elif [[ $ATDM_CONFIG_BUILD_NAME == *"cuda-9.2"* ]]; then
  export ATDM_CONFIG_COMPILER=CUDA-9.2
elif [[ $ATDM_CONFIG_BUILD_NAME == *"cuda"* ]]; then
  export ATDM_CONFIG_COMPILER=CUDA
elif [[ $ATDM_CONFIG_BUILD_NAME == *"clang"* ]]; then
  export ATDM_CONFIG_COMPILER=CLANG
else
  echo
  echo "***"
  echo "*** ERROR: A compiler was not specified in '$ATDM_CONFIG_BUILD_NAME'!"
  echo "***"
fi

# Set the KOKKOS_ARCH
if [[ $ATDM_CONFIG_BUILD_NAME == *"-AMDAVX"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=AMDAVX
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-ARMv8-ThunderX"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=ARMv8-ThunderX
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-ARMv80"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=ARMv80
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-ARMv81"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=ARMv81
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-BDW"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=BDW
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-BGQ"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=BGQ
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-HSW"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=HSW
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Kepler30"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Kepler30
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Kepler32"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Kepler32
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Kepler35"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Kepler35
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Kepler37"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Kepler37
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-KNC"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=KNC
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-KNL"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=KNL
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Maxwell50"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Maxwell50
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Maxwell52"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Maxwell52
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Maxwell53"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Maxwell53
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Pascal60"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Pascal60
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Pascal61"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Pascal61
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Power7"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Power7
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Power8"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Power8
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Power9"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Power9
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-SKX"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=SKX
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-SNB"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=SNB
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Volta70"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Volta70
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-Volta72"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=Volta72
elif [[ $ATDM_CONFIG_BUILD_NAME == *"-WSM"* ]]; then
  ATDM_CONFIG_KOKKOS_ARCH=WSM
else
  ATDM_CONFIG_KOKKOS_ARCH=DEFAULT
  if [[ $ATDM_CONFIG_VERBOSE == "1" ]] ; then
    echo "No KOKKOS_ARCH specified so using system default"
  fi
  # NOTE: <system_name>/environment.sh may set ATDM_CONFIG_KOKKOS_ARCH="" as
  # an allowed value!
fi

# Set the optimization level
# Defaults to debug
if [[ $ATDM_CONFIG_BUILD_NAME == *"release-debug"* ]]; then
  export ATDM_CONFIG_BUILD_TYPE=RELEASE_DEBUG;
elif [[ $ATDM_CONFIG_BUILD_NAME == *"release"* ]]; then
  export ATDM_CONFIG_BUILD_TYPE=RELEASE;
elif [[ $ATDM_CONFIG_BUILD_NAME == *"debug"* ]]; then
  export ATDM_CONFIG_BUILD_TYPE=DEBUG;
elif [[ $ATDM_CONFIG_BUILD_NAME == *"opt"* ]]; then
  export ATDM_CONFIG_BUILD_TYPE=RELEASE;
fi

# Set the node types default to serial
ATDM_CONFIG_NODE_TYPE=SERIAL
if [[ $ATDM_CONFIG_BUILD_NAME == *"cuda"* ]]; then
  export ATDM_CONFIG_USE_CUDA=ON
  export ATDM_CONFIG_NODE_TYPE=CUDA
elif [[ $ATDM_CONFIG_BUILD_NAME == *"serial"* ]]; then
  export ATDM_CONFIG_NODE_TYPE=SERIAL
elif [[ $ATDM_CONFIG_BUILD_NAME == *"pthread"* ]]; then
  export ATDM_CONFIG_USE_PTHREADS=ON
  export ATDM_CONFIG_NODE_TYPE=THREAD
elif [[ $ATDM_CONFIG_BUILD_NAME == *"openmp"* ]]; then
  export ATDM_CONFIG_USE_OPENMP=ON
  export ATDM_CONFIG_NODE_TYPE=OPENMP
fi
# NOTE: Above we move 'openmp' last to avoid clashing with 'openmpi' in the
# build name!  So as long as the build name explicitly contains 'serial' or
# 'cuda' or 'pthread', then that will be selected for the Kokkos backend.
# Otherwise, if one fo these are not selected and 'openmpi' is present in the
# build name, then the default Kokkos backend will become 'openmp'!
