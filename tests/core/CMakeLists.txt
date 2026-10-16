# P0024/P0026 production core executor and public consumer.

add_executable(df_core_exec core_exec.cpp)
target_include_directories(df_core_exec PRIVATE "${PROJECT_SOURCE_DIR}/src/core")
target_link_libraries(df_core_exec PRIVATE df_core df_cxx_baseline df_sanitize)

add_executable(df_router_consumer router_consumer.cpp)
target_link_libraries(df_router_consumer PRIVATE df_core df_cxx_baseline df_sanitize)
add_test(NAME core.router.consumer COMMAND df_router_consumer)
add_test(NAME core.moe.consumer COMMAND df_router_consumer --moe-only)
df_add_python_test(core.moe.parity "${CMAKE_CURRENT_SOURCE_DIR}/test_moe_scalar.py"
  --exe "$<TARGET_FILE:df_core_exec>" --suite production)
df_add_python_test(core.moe.harness "${CMAKE_CURRENT_SOURCE_DIR}/test_moe_scalar.py"
  --exe "$<TARGET_FILE:df_core_exec>" --suite harness)

if(NOT DF_TEST_SCOPE STREQUAL "core")
  df_add_python_test(core.compile_flags "${DF_COMPILE_FLAGS_CHECK}"
    --compile-commands "${PROJECT_BINARY_DIR}/compile_commands.json" --profile default)
endif()

foreach(_suite IN ITEMS parity contract harness)
  df_add_python_test(core.router.${_suite} "${CMAKE_CURRENT_SOURCE_DIR}/test_router_parity.py"
    --exe "$<TARGET_FILE:df_core_exec>" --suite ${_suite})
endforeach()
if(NOT DF_TEST_SCOPE STREQUAL "core")
  df_add_python_test(core.router.combination "${CMAKE_CURRENT_SOURCE_DIR}/test_router_parity.py"
    --exe "$<TARGET_FILE:df_core_exec>" --expert-exe "$<TARGET_FILE:df_quant_exec>"
    --suite combination)
endif()
