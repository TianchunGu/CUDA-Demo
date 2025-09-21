# ===========================================================================
# CMakeLists.txt for the 'util' library
#
# This file is responsible for defining and building the 'util' static library,
# which provides common utility functions and classes for the project.
# ===========================================================================

# --- 1. Source File Discovery ---
file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cu include/*.cuh)


# --- 2. Create Static Library ---
add_library(util STATIC ${srcs})


# --- 3. Specify Include Directories ---
target_include_directories(util PUBLIC include)