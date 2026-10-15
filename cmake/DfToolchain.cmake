# Supported-toolchain gate and reusable INTERFACE targets (D0005).
#
# df_cxx_baseline: C++20 without compiler extensions, common warnings.
# df_strict_fp:    numeric baseline; no fast-math, no FP contraction.
# df_sanitize:     ASan/UBSan when DF_SANITIZE is ON (cpu-asan preset, D0007); empty otherwise.
# None is applied globally; targets opt in via target_link_libraries.

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

option(DF_SANITIZE "Instrument opted-in targets with ASan and UBSan" OFF)
add_library(df_sanitize INTERFACE)
if(DF_SANITIZE)
  set(_df_sanitize_flags -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  target_compile_options(df_sanitize INTERFACE ${_df_sanitize_flags})
  target_link_options(df_sanitize INTERFACE ${_df_sanitize_flags})
endif()

set(DF_TEST_SCOPE "all" CACHE STRING "Test subtrees to build: all, quant only, or core only")
set_property(CACHE DF_TEST_SCOPE PROPERTY STRINGS all quant core)
if(NOT DF_TEST_SCOPE MATCHES "^(all|quant|core)$")
  message(FATAL_ERROR "DF_TEST_SCOPE must be 'all', 'quant' or 'core', got '${DF_TEST_SCOPE}'.")
endif()
