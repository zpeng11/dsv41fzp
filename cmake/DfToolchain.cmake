# Supported-toolchain gate and reusable INTERFACE targets (D0005).
#
# df_cxx_baseline: C++20 without compiler extensions, common warnings.
# df_strict_fp:    numeric baseline; no fast-math, no FP contraction.
# Neither is applied globally; targets opt in via target_link_libraries.

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "Unsupported C++ compiler '${CMAKE_CXX_COMPILER_ID}'; only GCC is supported (D0005).")
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
  message(FATAL_ERROR "Unsupported platform '${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}'; only Linux x86_64 is supported (D0005).")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

# User-supplied flags reach every target and would silently break df_strict_fp.
set(_df_flag_vars CMAKE_CXX_FLAGS)
foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})
  string(TOUPPER "${_cfg}" _cfg)
  list(APPEND _df_flag_vars "CMAKE_CXX_FLAGS_${_cfg}")
endforeach()
list(REMOVE_DUPLICATES _df_flag_vars)
foreach(_var IN LISTS _df_flag_vars)
  foreach(_bad IN ITEMS -ffast-math -Ofast -funsafe-math-optimizations -ffp-contract=fast -ffinite-math-only)
    string(FIND " ${${_var}} " " ${_bad} " _pos)
    if(NOT _pos EQUAL -1)
      message(FATAL_ERROR "${_var} contains '${_bad}', which contradicts the strict numeric baseline (D0005).")
    endif()
  endforeach()
endforeach()

add_library(df_cxx_baseline INTERFACE)
target_compile_features(df_cxx_baseline INTERFACE cxx_std_20)
target_compile_options(df_cxx_baseline INTERFACE -Wall -Wextra -Wpedantic)

add_library(df_strict_fp INTERFACE)
target_compile_options(df_strict_fp INTERFACE -fno-fast-math -ffp-contract=off)
