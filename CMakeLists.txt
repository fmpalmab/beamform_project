cmake_minimum_required(VERSION 3.24)

project(charts_voltage_beamformer_poc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BEAMFORMER_ENABLE_CUDA "Enable CUDA targets when a CUDA compiler is available" ON)

set(BEAMFORMER_HAS_CUDA OFF)
if(BEAMFORMER_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(BEAMFORMER_HAS_CUDA ON)
        message(STATUS "CUDA support enabled: ${CMAKE_CUDA_COMPILER}")
    else()
        message(WARNING "CUDA compiler not found; building the CPU-only foundation")
    endif()
endif()

add_library(beamformer_core
    src/cpu_beamformer.cpp
    src/geometry.cpp
    src/io.cpp
    src/quantization.cpp
    src/synthetic_data.cpp
    src/temporal_integration.cpp
    src/weights.cpp
)
target_include_directories(beamformer_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(beamformer_core PUBLIC BEAMFORMER_HAS_CUDA=$<BOOL:${BEAMFORMER_HAS_CUDA}>)
target_compile_options(beamformer_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

add_executable(generate_fake_data tools/generate_fake_data.cpp)
target_link_libraries(generate_fake_data PRIVATE beamformer_core)

add_executable(generate_weights tools/generate_weights.cpp)
target_link_libraries(generate_weights PRIVATE beamformer_core)

add_executable(beamformer_cpu tools/beamformer_cpu.cpp)
target_link_libraries(beamformer_cpu PRIVATE beamformer_core)

if(BEAMFORMER_HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(beamformer_cuda_core
        src/cuda_beamformer.cu
        src/cuda_offline_runner.cu
        src/cuda_quantize_int8.cu
    )
    target_include_directories(beamformer_cuda_core
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(beamformer_cuda_core PUBLIC beamformer_core CUDA::cudart)
    set_target_properties(beamformer_cuda_core PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    add_executable(beamformer_cuda tools/beamformer_cuda.cpp)
    target_link_libraries(beamformer_cuda PRIVATE beamformer_cuda_core)

    add_executable(benchmark_cpu_cuda tools/benchmark_cpu_cuda.cpp)
    target_link_libraries(benchmark_cpu_cuda PRIVATE beamformer_cuda_core)

    add_executable(benchmark_cuda_quantized tools/benchmark_cuda_quantized.cpp)
    target_link_libraries(benchmark_cuda_quantized PRIVATE beamformer_cuda_core)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test_contract tests/test_contract.cpp)
    target_link_libraries(test_contract PRIVATE beamformer_core)
    target_compile_options(test_contract PRIVATE -UNDEBUG)
    add_test(NAME contract COMMAND test_contract)

    add_executable(test_geometry tests/test_geometry.cpp)
    target_link_libraries(test_geometry PRIVATE beamformer_core)
    target_compile_options(test_geometry PRIVATE -UNDEBUG)
    add_test(NAME geometry COMMAND test_geometry)

    add_executable(test_synthetic_data tests/test_synthetic_data.cpp)
    target_link_libraries(test_synthetic_data PRIVATE beamformer_core)
    target_compile_options(test_synthetic_data PRIVATE -UNDEBUG)
    add_test(NAME synthetic_data COMMAND test_synthetic_data)

    add_executable(test_cpu_beamformer tests/test_cpu_beamformer.cpp)
    target_link_libraries(test_cpu_beamformer PRIVATE beamformer_core)
    target_compile_options(test_cpu_beamformer PRIVATE -UNDEBUG)
    add_test(NAME cpu_beamformer COMMAND test_cpu_beamformer)

    add_executable(test_temporal_integration tests/test_temporal_integration.cpp)
    target_link_libraries(test_temporal_integration PRIVATE beamformer_core)
    target_compile_options(test_temporal_integration PRIVATE -UNDEBUG)
    add_test(NAME temporal_integration COMMAND test_temporal_integration)

    add_executable(test_quantization tests/test_quantization.cpp)
    target_link_libraries(test_quantization PRIVATE beamformer_core)
    target_compile_options(test_quantization PRIVATE -UNDEBUG)
    add_test(NAME quantization COMMAND test_quantization)

    add_executable(test_cuda_frame_contract tests/test_cuda_frame_contract.cpp)
    target_link_libraries(test_cuda_frame_contract PRIVATE beamformer_core)
    target_compile_options(test_cuda_frame_contract PRIVATE -UNDEBUG)
    add_test(NAME cuda_frame_contract COMMAND test_cuda_frame_contract)
    set_tests_properties(cuda_frame_contract PROPERTIES LABELS "contract")

    if(BEAMFORMER_HAS_CUDA)
        add_executable(test_cuda_point_source tests/test_cuda_point_source.cpp)
        target_link_libraries(test_cuda_point_source PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_point_source PRIVATE -UNDEBUG)
        add_test(NAME cuda_point_source COMMAND test_cuda_point_source)
        set_tests_properties(cuda_point_source PROPERTIES
            LABELS "cuda;physics"
            TIMEOUT 60
        )

        add_executable(test_cuda_packed tests/test_cuda_packed.cu)
        target_link_libraries(test_cuda_packed PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_packed PRIVATE -UNDEBUG)
        add_test(NAME cuda_packed COMMAND test_cuda_packed)
        set_tests_properties(cuda_packed PROPERTIES
            LABELS "cuda;packed;validation"
            TIMEOUT 60
        )

        add_executable(test_cuda_temporal_integration
                       tests/test_cuda_temporal_integration.cpp)
        target_link_libraries(test_cuda_temporal_integration PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_temporal_integration PRIVATE -UNDEBUG)
        add_test(NAME cuda_temporal_integration COMMAND test_cuda_temporal_integration)
        set_tests_properties(cuda_temporal_integration PROPERTIES
            LABELS "cuda;temporal-integration;validation"
            TIMEOUT 60
        )

        add_executable(test_cuda_quantization tests/test_cuda_quantization.cpp)
        target_link_libraries(test_cuda_quantization PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_quantization PRIVATE -UNDEBUG)
        add_test(NAME cuda_quantization COMMAND test_cuda_quantization)
        set_tests_properties(cuda_quantization PROPERTIES
            LABELS "cuda;quantization;validation"
            TIMEOUT 60
        )

        add_executable(test_cuda_frame_pipeline
                       tests/test_cuda_frame_pipeline.cpp)
        set_source_files_properties(
            tests/test_cuda_frame_pipeline.cpp PROPERTIES LANGUAGE CUDA)
        target_link_libraries(test_cuda_frame_pipeline PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_frame_pipeline PRIVATE -UNDEBUG)
        add_test(NAME cuda_frame_pipeline COMMAND test_cuda_frame_pipeline)
        set_tests_properties(cuda_frame_pipeline PROPERTIES
            LABELS "cuda;contract;validation"
            TIMEOUT 60
        )

        add_executable(test_cuda_offline_runner
                       tests/test_cuda_offline_runner.cpp)
        target_link_libraries(test_cuda_offline_runner PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_offline_runner PRIVATE -UNDEBUG)
        add_test(NAME cuda_offline_runner COMMAND test_cuda_offline_runner)
        set_tests_properties(cuda_offline_runner PROPERTIES
            LABELS "cuda;contract;validation"
            TIMEOUT 60
        )

        add_test(
            NAME benchmark_config
            COMMAND benchmark_cpu_cuda --dry-run
                    --n-ant 32,64 --beams 16,32,64,128
                    --times 15360,24576,30720 --validation-time 16
        )
        set_tests_properties(benchmark_config PROPERTIES
            LABELS "cuda;benchmark"
            TIMEOUT 10
        )
    endif()
endif()
