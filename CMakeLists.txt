cmake_minimum_required(VERSION 3.22)
project(gpu-nvme-direct LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA architecture: RTX 3090 = sm_86
set(CMAKE_CUDA_ARCHITECTURES 86)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror=implicit-function-declaration")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Debug / Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0 -DDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Find CUDA toolkit (needed for cudaMallocHost etc.)
find_package(CUDAToolkit REQUIRED)

# Option to build with simulator (for WSL/dev without hardware)
option(GPUNVME_USE_SIM "Build with NVMe simulator instead of real hardware" ON)

# ---------- NVMe Simulator Library ----------
if(GPUNVME_USE_SIM)
    add_library(nvme_sim STATIC
        src/sim/nvme_sim.c
    )
    target_link_libraries(nvme_sim PRIVATE
        CUDA::cudart
        pthread
    )
    target_include_directories(nvme_sim PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/sim
    )
    add_compile_definitions(GPUNVME_USE_SIM=1)
endif()

# ---------- GPU Device Library ----------
add_library(gpunvme_device STATIC
    src/device/block_io.cu
)
target_include_directories(gpunvme_device PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(gpunvme_device PUBLIC
    CUDA::cudart
)
if(GPUNVME_USE_SIM)
    target_link_libraries(gpunvme_device PUBLIC nvme_sim)
endif()

# ---------- Host Library (Phase 1+) ----------
if(NOT GPUNVME_USE_SIM)
    add_library(gpunvme_host STATIC
        src/host/controller.c
        src/host/admin.c
        src/host/io_queue.c
        src/host/bar_map.c
        src/host/dma_alloc.c
        src/host/pci_util.c
        src/host/monitor.c
    )
    target_include_directories(gpunvme_host PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(gpunvme_host PUBLIC
        CUDA::cudart
        pthread
    )
endif()

# ---------- Tests ----------
enable_testing()

# Struct size/alignment test (pure C++, no CUDA needed)
add_executable(test_nvme_structs tests/test_nvme_structs.cpp)
target_include_directories(test_nvme_structs PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_nvme_structs COMMAND test_nvme_structs)

# Simulator-based GPU test
if(GPUNVME_USE_SIM)
    add_executable(test_sim_basic tests/test_sim_basic.cu)
    target_include_directories(test_sim_basic PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(test_sim_basic PRIVATE
        gpunvme_device
        nvme_sim
        CUDA::cudart
        pthread
    )
    add_test(NAME test_sim_basic COMMAND test_sim_basic)
endif()

# ---------- Benchmarks (Phase 5) ----------
add_subdirectory(bench)

# ---------- Hardware Tests (Phase 1+) ----------
if(NOT GPUNVME_USE_SIM)
    add_executable(test_single_block tests/test_single_block.cu)
    target_include_directories(test_single_block PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(test_single_block PRIVATE
        gpunvme_host
        gpunvme_device
        CUDA::cudart
        stdc++
        m
    )
    add_test(NAME test_single_block COMMAND test_single_block)

    add_executable(test_multi_block tests/test_multi_block.cu)
    target_include_directories(test_multi_block PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
    )
    target_link_libraries(test_multi_block PRIVATE
        gpunvme_host
        gpunvme_device
        CUDA::cudart
        stdc++
        m
    )
    add_test(NAME test_multi_block COMMAND test_multi_block)
endif()

# ---------- Tools (Phase 1+) ----------
# Tools are built separately when needed on bare metal
