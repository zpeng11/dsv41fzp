# Production quant library (P0010): the single source of truth for format codecs.

add_library(df_quant STATIC codec.cpp contract.h)
target_include_directories(df_quant PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(df_quant PUBLIC cxx_std_20)
target_link_libraries(df_quant PRIVATE df_cxx_baseline df_strict_fp)
