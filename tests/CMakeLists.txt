# Unified CTest entry (P0009–P0012): C++ build baseline, quant parity and Python suites.
# DF_TEST_SCOPE=quant (cpu-asan) registers only the quant subtree and its flag check.

function(df_add_python_test name script)
  add_test(NAME ${name} COMMAND "${Python3_EXECUTABLE}" "${script}" ${ARGN})
  set_tests_properties(${name} PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endfunction()

set(DF_COMPILE_FLAGS_CHECK "${CMAKE_CURRENT_SOURCE_DIR}/cpu/build_baseline/check_compile_flags.py")

add_subdirectory(quant)

if(DF_TEST_SCOPE STREQUAL "quant")
  df_add_python_test(quant.compile_flags "${DF_COMPILE_FLAGS_CHECK}"
    --compile-commands "${PROJECT_BINARY_DIR}/compile_commands.json" --profile quant-sanitize)
  return()
endif()

add_subdirectory(cpu/build_baseline)

df_add_python_test(quant.formats "${PROJECT_SOURCE_DIR}/tests/quant/test_formats.py")
df_add_python_test(quant.compute "${PROJECT_SOURCE_DIR}/tests/quant/test_compute.py")
df_add_python_test(tools.check_repo "${PROJECT_SOURCE_DIR}/tests/tools/test_check_repo.py")
