cmake_minimum_required(VERSION 4.0.2)

project(nand-vm LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL ON)

set(SANITIZER "" CACHE STRING "Sanitizer to use (none, address, thread, memory)")

if (SANITIZER STREQUAL "address")
  add_compile_options(-fsanitize=undefined,address,leak -fno-omit-frame-pointer)
  link_libraries(-fsanitize=undefined,address,leak)
elseif (SANITIZER STREQUAL "thread")
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  link_libraries(-fsanitize=thread)
elseif (SANITIZER STREQUAL "memory")
  add_compile_options(-fsanitize=memory -fno-omit-frame-pointer)
  link_libraries(-fsanitize=memory)
endif ()

add_compile_options(-Wall -Wextra -pedantic)
add_compile_options(-march=native)
#add_compile_options(-ffast-math)

add_executable(main src/main.cpp)

target_sources(main PRIVATE FILE_SET CXX_MODULES FILES
  src/lib/lib.non_copyable.cppm
  src/mind.cppm
)

find_package(OpenMP REQUIRED)
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)


include(cmake/CPM.cmake)

CPMAddPackage(
  NAME xtl
  VERSION 0.8.0
  URL https://github.com/xtensor-stack/xtl/archive/refs/tags/0.8.0.tar.gz
  URL_HASH SHA256=ee38153b7dd0ec84cee3361f5488a4e7e6ddd26392612ac8821cbc76e740273a
  SYSTEM ON
  EXCLUDE_FROM_ALL ON
  OPTIONS
  "XTENSOR_USE_XSIMD ON"
  "XTENSOR_USE_OPENMP OFF"
  "XTENSOR_USE_TBB OFF"
)

CPMAddPackage(
  NAME xsimd
  VERSION 13.2.0
  URL https://github.com/xtensor-stack/xsimd/archive/refs/tags/13.2.0.tar.gz
  URL_HASH SHA256=edd8cd3d548c185adc70321c53c36df41abe64c1fe2c67bc6d93c3ecda82447a
  SYSTEM ON
  EXCLUDE_FROM_ALL ON
  OPTIONS
  "XSIMD_REFACTORING ON" # what is it?
  "ENABLE_XTL_COMPLEX OFF"
  "BUILD_TESTS OFF"
  "BUILD_BENCHMARK OFF"
  "BUILD_EXAMPLES OFF"
  "XSIMD_SKIP_INSTALL ON"
)

CPMAddPackage(
  NAME xtensor
  VERSION 0.26.0
  URL https://github.com/xtensor-stack/xtensor/archive/refs/tags/0.26.0.tar.gz
  URL_HASH SHA256=f5f42267d850f781d71097b50567a480a82cd6875a5ec3e6238555e0ef987dc6
  SYSTEM ON
  EXCLUDE_FROM_ALL ON
  OPTIONS
  "XTENSOR_USE_XSIMD ON"
  "XTENSOR_USE_OPENMP OFF"
  "XTENSOR_USE_TBB OFF"
)
target_link_libraries(main PRIVATE xtensor)