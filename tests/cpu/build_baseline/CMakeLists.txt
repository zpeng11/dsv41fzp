# Build-baseline smoke artifacts (P0009). Test-only; not engine modules.

add_library(bb_smoke STATIC smoke.cpp)
target_include_directories(bb_smoke PUBLIC include)
target_link_libraries(bb_smoke PRIVATE df_cxx_baseline)

add_executable(bb_smoke_test smoke_test.cpp)
target_link_libraries(bb_smoke_test PRIVATE bb_smoke df_cxx_baseline)
add_test(NAME bb.smoke COMMAND bb_smoke_test)

add_library(bb_version STATIC version.cpp)
target_include_directories(bb_version PUBLIC include)
target_link_libraries(bb_version PRIVATE df_cxx_baseline)
df_use_version_info(bb_version)

add_executable(bb_version_print version_print.cpp)
target_link_libraries(bb_version_print PRIVATE bb_version df_cxx_baseline)
df_add_python_test(bb.version "${CMAKE_CURRENT_SOURCE_DIR}/check_version.py"
  --exe "$<TARGET_FILE:bb_version_print>"
  --source-dir "${PROJECT_SOURCE_DIR}"
  --git "${GIT_EXECUTABLE}")

# fp_probe.cpp is a separate TU so the compiler cannot fold or reorder the
# probed arithmetic across fesetround/MXCSR changes in fp_env.cpp.
add_executable(bb_fp_env fp_env.cpp fp_probe.cpp)
target_link_libraries(bb_fp_env PRIVATE df_cxx_baseline df_strict_fp)
add_test(NAME bb.fp_env COMMAND bb_fp_env)

# Contraction probe: -O2 -mfma regardless of build type, because GCC does not
# contract at -O0. The fast twin proves the probe can observe contraction.
foreach(_variant IN ITEMS strict fast)
  add_executable(bb_contract_${_variant} contract_main.cpp contract_probe.cpp)
  target_link_libraries(bb_contract_${_variant} PRIVATE df_cxx_baseline)
  target_compile_options(bb_contract_${_variant} PRIVATE -O2 -mfma)
endforeach()
target_link_libraries(bb_contract_strict PRIVATE df_strict_fp)
target_compile_options(bb_contract_fast PRIVATE -ffp-contract=fast)
add_test(NAME bb.contract.strict COMMAND bb_contract_strict --expect separate)
add_test(NAME bb.contract.fast_control COMMAND bb_contract_fast --expect fused)

df_add_python_test(bb.compile_flags "${CMAKE_CURRENT_SOURCE_DIR}/check_compile_flags.py"
  --compile-commands "${PROJECT_BINARY_DIR}/compile_commands.json")
