# Production quant consumers (P0010, P0011, P0013, P0017, P0019): the executor links df_quant only,
# never its sources.

add_executable(df_quant_exec quant_exec.cpp)
target_link_libraries(df_quant_exec PRIVATE df_quant df_cxx_baseline df_sanitize)
target_include_directories(df_quant_exec PRIVATE "${PROJECT_SOURCE_DIR}/src/quant")

foreach(_suite IN ITEMS parity contract harness)
  df_add_python_test(quant.codec.${_suite} "${CMAKE_CURRENT_SOURCE_DIR}/test_codec_parity.py"
    --exe "$<TARGET_FILE:df_quant_exec>" --suite ${_suite})
endforeach()

foreach(_suite IN ITEMS parity contract multiply harness)
  df_add_python_test(quant.dequant.${_suite} "${CMAKE_CURRENT_SOURCE_DIR}/test_dequant_parity.py"
    --exe "$<TARGET_FILE:df_quant_exec>" --suite ${_suite})
endforeach()

foreach(_suite IN ITEMS parity contract combination harness)
  df_add_python_test(quant.act_quant.${_suite} "${CMAKE_CURRENT_SOURCE_DIR}/test_act_quant_parity.py"
    --exe "$<TARGET_FILE:df_quant_exec>" --suite ${_suite})
endforeach()

add_executable(df_linear_consumer linear_consumer.cpp)
target_link_libraries(df_linear_consumer PRIVATE df_quant df_cxx_baseline df_sanitize)
add_test(NAME quant.linear.consumer COMMAND df_linear_consumer)

foreach(_suite IN ITEMS parity contract harness)
  df_add_python_test(quant.linear.${_suite} "${CMAKE_CURRENT_SOURCE_DIR}/test_linear_parity.py"
    --exe "$<TARGET_FILE:df_quant_exec>" --suite ${_suite})
endforeach()

foreach(_suite IN ITEMS parity contract add harness)
  df_add_python_test(quant.compute.${_suite} "${CMAKE_CURRENT_SOURCE_DIR}/test_compute_parity.py"
    --exe "$<TARGET_FILE:df_quant_exec>" --suite ${_suite})
endforeach()
