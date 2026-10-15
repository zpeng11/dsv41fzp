# Production quant library (P0010): the single source of truth for format codecs.

set(DF_SILU_TABLE_HEADER "${PROJECT_BINARY_DIR}/generated/df/quant/silu_table.h")
add_custom_command(
  OUTPUT "${DF_SILU_TABLE_HEADER}"
  COMMAND "${CMAKE_COMMAND}" -E env PYTHONDONTWRITEBYTECODE=1
          "${Python3_EXECUTABLE}" -B "${PROJECT_SOURCE_DIR}/tools/gen_silu_table.py"
          "${DF_SILU_TABLE_HEADER}"
  DEPENDS
    "${PROJECT_SOURCE_DIR}/tools/gen_silu_table.py"
    "${PROJECT_SOURCE_DIR}/ref/expert.py"
    "${PROJECT_SOURCE_DIR}/ref/formats.py"
  COMMENT "Generating the verified BF16 SiLU table"
  VERBATIM)

add_library(df_quant STATIC codec.cpp dequant.cpp compute.cpp act_quant.cpp linear.cpp swiglu.cpp
            contract.h detail.h "${DF_SILU_TABLE_HEADER}")
target_include_directories(df_quant PUBLIC "${PROJECT_SOURCE_DIR}/include"
                            PRIVATE "${PROJECT_BINARY_DIR}/generated")
target_compile_features(df_quant PUBLIC cxx_std_20)
target_link_libraries(df_quant PRIVATE df_cxx_baseline df_strict_fp df_sanitize)
