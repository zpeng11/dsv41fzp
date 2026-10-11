# Production quant consumers (P0010): the executor links df_quant only, never its sources.

add_executable(df_quant_exec quant_exec.cpp)
target_link_libraries(df_quant_exec PRIVATE df_quant df_cxx_baseline)

foreach(_suite IN ITEMS parity)
  df_add_python_test(quant.codec.${_suite} "${CMAKE_CURRENT_SOURCE_DIR}/test_codec_parity.py"
    --exe "$<TARGET_FILE:df_quant_exec>" --suite ${_suite})
endforeach()
