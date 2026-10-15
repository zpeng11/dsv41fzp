# Unified CTest entry (P0009–P0025): build baseline, quant and core suites.
# Sanitizer scopes register only their selected subtree and compile-flag check.

function(df_add_python_test name script)
  add_test(NAME ${name} COMMAND "${Python3_EXECUTABLE}" "${script}" ${ARGN})
  set_tests_properties(${name} PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endfunction()

set(DF_COMPILE_FLAGS_CHECK "${CMAKE_CURRENT_SOURCE_DIR}/cpu/build_baseline/check_compile_flags.py")

if(NOT DF_TEST_SCOPE STREQUAL "core")
  add_subdirectory(quant)
endif()
if(NOT DF_TEST_SCOPE STREQUAL "quant")
  add_subdirectory(core)
endif()

if(DF_TEST_SCOPE STREQUAL "quant")
  df_add_python_test(quant.compile_flags "${DF_COMPILE_FLAGS_CHECK}"
    --compile-commands "${PROJECT_BINARY_DIR}/compile_commands.json" --profile quant-sanitize)
  return()
elseif(DF_TEST_SCOPE STREQUAL "core")
  df_add_python_test(core.compile_flags "${DF_COMPILE_FLAGS_CHECK}"
    --compile-commands "${PROJECT_BINARY_DIR}/compile_commands.json" --profile core-sanitize)
  return()
endif()

add_subdirectory(cpu/build_baseline)

df_add_python_test(quant.formats "${PROJECT_SOURCE_DIR}/tests/quant/test_formats.py")
df_add_python_test(quant.compute "${PROJECT_SOURCE_DIR}/tests/quant/test_compute.py")
df_add_python_test(quant.act_quant "${PROJECT_SOURCE_DIR}/tests/quant/test_act_quant.py")
df_add_python_test(quant.expert "${PROJECT_SOURCE_DIR}/tests/quant/test_expert.py")
df_add_python_test(core.router "${PROJECT_SOURCE_DIR}/tests/core/test_router.py")
df_add_python_test(core.moe "${PROJECT_SOURCE_DIR}/tests/core/test_moe.py")
df_add_python_test(core.router.expert "${PROJECT_SOURCE_DIR}/tests/core/test_router_expert.py"
  --exe $<TARGET_FILE:df_quant_exec>)
df_add_python_test(tools.check_repo "${PROJECT_SOURCE_DIR}/tests/tools/test_check_repo.py")
df_add_python_test(tools.ckpt_inventory "${PROJECT_SOURCE_DIR}/tests/tools/test_ckpt_inventory.py")
df_add_python_test(tools.ckpt_expert "${PROJECT_SOURCE_DIR}/tests/tools/test_ckpt_expert.py"
  --exe $<TARGET_FILE:df_quant_exec>)
df_add_python_test(tools.ckpt_expert_mlp "${PROJECT_SOURCE_DIR}/tests/tools/test_ckpt_expert_mlp.py"
  --exe $<TARGET_FILE:df_quant_exec>)
df_add_python_test(tools.ckpt_linear "${PROJECT_SOURCE_DIR}/tests/tools/test_ckpt_linear.py"
  --exe $<TARGET_FILE:df_quant_exec>)
