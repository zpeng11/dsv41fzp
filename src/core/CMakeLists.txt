# Production scalar router (P0024).

add_library(df_core STATIC router.cpp router_math.cpp router_detail.h router_fixed.h)
target_include_directories(df_core PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_compile_features(df_core PUBLIC cxx_std_20)
target_link_libraries(df_core PRIVATE df_quant df_cxx_baseline df_strict_fp df_sanitize)
