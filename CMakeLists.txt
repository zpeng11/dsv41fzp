cmake_minimum_required(VERSION 3.21)
project(dsv41fzp LANGUAGES CXX)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed; use a preset (binary dir build/<preset>).")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(DfToolchain)
include(DfVersion)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_subdirectory(src/quant)
if(NOT DF_TEST_SCOPE STREQUAL "quant")
  add_subdirectory(src/core)
endif()

enable_testing()
add_subdirectory(tests)
