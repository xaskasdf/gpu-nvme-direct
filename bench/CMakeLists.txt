# gpu-nvme-direct: Benchmark Build Configuration
#
# Each .cu file becomes its own executable.
# Python scripts are copied to the build directory.
#
# SPDX-License-Identifier: BSD-2-Clause

# ---------- Benchmark Executables ----------

# Collect all .cu benchmark source files
file(GLOB BENCH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

foreach(bench_src ${BENCH_SOURCES})
    # Extract filename without extension to use as target name
    get_filename_component(bench_name ${bench_src} NAME_WE)

    add_executable(${bench_name} ${bench_src})

    target_include_directories(${bench_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(${bench_name} PRIVATE
        gpunvme_device
        CUDA::cudart
        pthread
    )

    # cuFile (GPUDirect Storage) is optional
    if(bench_name STREQUAL "bench_cufile")
        find_library(CUFILE_LIB cufile HINTS
            ${CUDAToolkit_LIBRARY_DIR}
            /usr/local/cuda/lib64
            /usr/lib/x86_64-linux-gnu
        )
        if(CUFILE_LIB)
            target_link_libraries(${bench_name} PRIVATE ${CUFILE_LIB})
            target_compile_definitions(${bench_name} PRIVATE HAVE_CUFILE=1)
            message(STATUS "bench_cufile: linking with cuFile (${CUFILE_LIB})")
        else()
            message(STATUS "bench_cufile: cuFile not found, will use POSIX fallback")
        endif()
    endif()

    # Link against the simulator when building in sim mode
    if(GPUNVME_USE_SIM AND TARGET nvme_sim)
        target_link_libraries(${bench_name} PRIVATE nvme_sim)
    endif()

    set_target_properties(${bench_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bench"
    )
endforeach()

# ---------- Python Script Targets ----------

file(GLOB BENCH_PYTHON_SCRIPTS "${CMAKE_CURRENT_SOURCE_DIR}/*.py")

foreach(py_script ${BENCH_PYTHON_SCRIPTS})
    get_filename_component(py_name ${py_script} NAME)
    get_filename_component(py_name_we ${py_script} NAME_WE)

    add_custom_target(${py_name_we} ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${py_script}
            "${CMAKE_BINARY_DIR}/bench/${py_name}"
        COMMENT "Copying ${py_name} to build directory"
    )
endforeach()
