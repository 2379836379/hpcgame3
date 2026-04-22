cmake_minimum_required(VERSION 3.14)

# Prefer mpicxx wrapper if available to avoid FindMPI issues in restricted envs.
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(MPICXX mpicxx)
    if(MPICXX)
        set(CMAKE_CXX_COMPILER "${MPICXX}" CACHE FILEPATH "MPI CXX compiler" FORCE)
    endif()
endif()

project(mr_spgemm_topk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MPI QUIET)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/spgemm_topk.cpp)
    set(SPGEMM_IMPL src/spgemm_topk.cpp)
    message(STATUS "Using spgemm_topk.cpp")
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/spgemm.cpp)
    set(SPGEMM_IMPL src/spgemm.cpp)
    message(STATUS "Using spgemm.cpp")
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/spgemm_naive.cpp)
    set(SPGEMM_IMPL src/spgemm_naive.cpp)
    message(STATUS "Using spgemm_naive.cpp")
else()
    message(FATAL_ERROR "No SpGEMM implementation file found!")
endif()

add_executable(mr_spgemm_topk
    src/main.cpp
    ${SPGEMM_IMPL}
)

if(MPI_CXX_FOUND)
    target_link_libraries(mr_spgemm_topk PRIVATE MPI::MPI_CXX)
endif()

target_compile_options(mr_spgemm_topk PRIVATE
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
)
