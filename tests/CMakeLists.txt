# Unified CTest entry (P0009): C++ build baseline plus existing Python suites.

function(df_add_python_test name script)
  add_test(NAME ${name} COMMAND "${Python3_EXECUTABLE}" "${script}" ${ARGN})
  set_tests_properties(${name} PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endfunction()

add_subdirectory(cpu/build_baseline)
add_subdirectory(quant)

df_add_python_test(quant.formats "${PROJECT_SOURCE_DIR}/tests/quant/test_formats.py")
df_add_python_test(tools.check_repo "${PROJECT_SOURCE_DIR}/tests/tools/test_check_repo.py")
