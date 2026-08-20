cmake_minimum_required(VERSION 3.24)

project(charts_voltage_beamformer_poc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable OpenMP
find_package(OpenMP REQUIRED)

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
    src/beam_tracker.cpp
    src/beam_tracker_opt.cpp
    src/beam_tracker_opt_v2.cpp
    src/cpu_opt_beam_tracker.cpp
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
target_link_libraries(beamformer_core PUBLIC OpenMP::OpenMP_CXX)

# Optional: instrument the optimized tracker TU inside beamformer_core with
# -DBEAMFORMER_TRACKER_DEBUG so run_into captures the full coarse-to-fine
# search state (covariance, snapshots, per-window spectra) that
# CpuOptBeamTracker::debug_search_dump() can write to disk. Default OFF to
# keep the capture overhead out of normal builds; enable explicitly when
# debugging a failing DOA-recovery assertion:
#      cmake -DBEAMFORMER_TRACKER_DEBUG=ON ...
# This only affects beamformer_core (and the tests that link it); the
# perf-enabled benchmark core (beamformer_core_perf) is left untouched.
option(BEAMFORMER_TRACKER_DEBUG
       "Capture optimized-tracker search state in run_into for debug dumps" OFF)
if(BEAMFORMER_TRACKER_DEBUG)
    target_compile_definitions(beamformer_core PRIVATE BEAMFORMER_TRACKER_DEBUG)
endif()

# Performance build of the whole CPU core: the optimized tracker TU is compiled
# with BEAMFORMER_TRACKER_PERF defined (which populates the per-window latency
# vector inside run_into) and -O3 -march=native, *and* every other CPU TU is compiled at -O3
# too so the headline per-frame numbers are measured at release optimisation.
add_library(beamformer_core_perf STATIC
    src/cpu_beamformer.cpp
    src/beam_tracker.cpp
    src/beam_tracker_opt.cpp
    src/beam_tracker_opt_v2.cpp
    src/cpu_opt_beam_tracker.cpp
    src/geometry.cpp
    src/io.cpp
    src/quantization.cpp
    src/synthetic_data.cpp
    src/temporal_integration.cpp
    src/weights.cpp
)
target_include_directories(beamformer_core_perf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(beamformer_core_perf PUBLIC
    BEAMFORMER_HAS_CUDA=$<BOOL:${BEAMFORMER_HAS_CUDA}>
)
target_compile_definitions(beamformer_core_perf PRIVATE
    BEAMFORMER_TRACKER_PERF
)
target_compile_options(beamformer_core_perf PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
)
target_link_libraries(beamformer_core_perf PUBLIC OpenMP::OpenMP_CXX)

add_executable(generate_fake_data tools/generate_fake_data.cpp)
target_link_libraries(generate_fake_data PRIVATE beamformer_core)

add_executable(generate_weights tools/generate_weights.cpp)
target_link_libraries(generate_weights PRIVATE beamformer_core)

add_executable(beamformer_cpu tools/beamformer_cpu.cpp)
target_link_libraries(beamformer_cpu PRIVATE beamformer_core)

add_executable(beam_tracker_cpu tools/beam_tracker_cpu.cpp)
target_link_libraries(beam_tracker_cpu PRIVATE beamformer_core)

# Benchmark for the optimized CPU tracker.
add_executable(benchmark_cpu_opt_beam_tracker
    benchmarks/benchmark_cpu_opt_beam_tracker.cpp)
target_link_libraries(benchmark_cpu_opt_beam_tracker PRIVATE beamformer_core_perf OpenMP::OpenMP_CXX)
target_compile_options(benchmark_cpu_opt_beam_tracker PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
)

# Benchmark for the optimized NAIVE tracker (src/beam_tracker_opt.cpp): runs
# both the naive tracker (src/beam_tracker.cpp) and the optimized variant on
# identical input, reports actual end-to-end latency for each plus a speedup
# ratio, and re-asserts the naive==opt byte-equality contract in-process.
# Built against the perf core so both the opt TU and the naive TU it is
# compared against are at -O3 -march=native for a fair headline comparison.
add_executable(benchmark_beam_tracker_opt
    benchmarks/benchmark_beam_tracker_opt.cpp)
target_link_libraries(benchmark_beam_tracker_opt PRIVATE beamformer_core_perf OpenMP::OpenMP_CXX)
target_compile_options(benchmark_beam_tracker_opt PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
)

# Benchmark for the v2 optimized naive tracker (src/beam_tracker_opt_v2.cpp):
# runs the naive tracker, v1, and v2 on identical input at the same config and
# reports end-to-end latency for each, plus v1-vs-naive, v2-vs-naive and
# v2-vs-v1 speedup ratios. Re-asserts the three-way byte-equality contract
# (naive==v1==v2) in-process before timing, so a fast-but-wrong kernel is
# caught here too. Built against the perf core so v1, v2, and the naive TU they
# are compared against are all at -O3 -march=native for a fair comparison.
add_executable(benchmark_beam_tracker_opt_v2
    benchmarks/benchmark_beam_tracker_opt_v2.cpp)
target_link_libraries(benchmark_beam_tracker_opt_v2 PRIVATE beamformer_core_perf OpenMP::OpenMP_CXX)
target_compile_options(benchmark_beam_tracker_opt_v2 PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
)

add_executable(beam_tracker_naive_cpu_test_suite
    tests/cpu/beam_tracker_naive_cpu_test_suite.cpp)
target_link_libraries(beam_tracker_naive_cpu_test_suite PRIVATE beamformer_core)
target_compile_options(beam_tracker_naive_cpu_test_suite PRIVATE -UNDEBUG -O2)

add_executable(run_tracker_stream tools/run_tracker_stream.cpp)
if(BEAMFORMER_HAS_CUDA)
    target_link_libraries(run_tracker_stream PRIVATE beamformer_cuda_core OpenMP::OpenMP_CXX)
else()
    target_link_libraries(run_tracker_stream PRIVATE beamformer_core OpenMP::OpenMP_CXX)
endif()

if(BEAMFORMER_HAS_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(beamformer_cuda_core
        src/cuda_beamformer.cu
        src/cuda_offline_runner.cu
        src/cuda_quantize_int8.cu
        src/cuda_beam_tracker_v2.cu
        src/cuda_beam_tracker_fused_warp_shuffle.cu
        src/cuda_beam_tracker_v3.cu
        src/cuda_beam_tracker_v4.cu
        src/cuda_beam_tracker_v5.cu
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

    add_executable(benchmark_cpu_cuda benchmarks/benchmark_cpu_cuda.cpp)
    target_link_libraries(benchmark_cpu_cuda PRIVATE beamformer_cuda_core)

    add_executable(benchmark_cuda_quantized benchmarks/benchmark_cuda_quantized.cpp)
    target_link_libraries(benchmark_cuda_quantized PRIVATE beamformer_cuda_core)

    # Benchmark for CUDA Tracker V2 vs CPU optimized trackers
    add_executable(benchmark_cuda_tracker_v2 benchmarks/benchmark_cuda_tracker_v2.cpp)
    target_link_libraries(benchmark_cuda_tracker_v2 
        PRIVATE 
        beamformer_cuda_core 
        OpenMP::OpenMP_CXX
    )
    target_compile_options(benchmark_cuda_tracker_v2 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
    )

    # Benchmark for CUDA Tracker V3 (ILP, PTX bfe, Tiled, Streaming, Graph)
    add_executable(benchmark_cuda_tracker_v3 benchmarks/benchmark_cuda_tracker_v3.cpp)
    target_link_libraries(benchmark_cuda_tracker_v3 
        PRIVATE 
        beamformer_cuda_core 
        OpenMP::OpenMP_CXX
    )
    target_compile_options(benchmark_cuda_tracker_v3 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
    )

    # Benchmark for CUDA Tracker V4 (Tensor Core WMMA, Half2 SIMD, Deep ILP, Graph)
    add_executable(benchmark_cuda_tracker_v4 benchmarks/benchmark_cuda_tracker_v4.cpp)
    target_link_libraries(benchmark_cuda_tracker_v4 
        PRIVATE 
        beamformer_cuda_core 
        OpenMP::OpenMP_CXX
    )
    target_compile_options(benchmark_cuda_tracker_v4 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
    )

    # Benchmark for CUDA Tracker V5 (Unified Warp Reduction Single Engine)
    add_executable(benchmark_cuda_tracker_v5 benchmarks/benchmark_cuda_tracker_v5.cpp)
    target_link_libraries(benchmark_cuda_tracker_v5 
        PRIVATE 
        beamformer_cuda_core 
        OpenMP::OpenMP_CXX
    )
    target_compile_options(benchmark_cuda_tracker_v5 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native;-Wall;-Wextra;-Wpedantic>
    )
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test_contract tests/cpu/test_contract.cpp)
    target_link_libraries(test_contract PRIVATE beamformer_core)
    target_compile_options(test_contract PRIVATE -UNDEBUG)
    add_test(NAME contract COMMAND test_contract)

    add_executable(test_geometry tests/cpu/test_geometry.cpp)
    target_link_libraries(test_geometry PRIVATE beamformer_core)
    target_compile_options(test_geometry PRIVATE -UNDEBUG)
    add_test(NAME geometry COMMAND test_geometry)

    add_executable(test_synthetic_data tests/cpu/test_synthetic_data.cpp)
    target_link_libraries(test_synthetic_data PRIVATE beamformer_core)
    target_compile_options(test_synthetic_data PRIVATE -UNDEBUG)
    add_test(NAME synthetic_data COMMAND test_synthetic_data)

    add_executable(test_cpu_beamformer tests/cpu/test_cpu_beamformer.cpp)
    target_link_libraries(test_cpu_beamformer PRIVATE beamformer_core)
    target_compile_options(test_cpu_beamformer PRIVATE -UNDEBUG)
    add_test(NAME cpu_beamformer COMMAND test_cpu_beamformer)

    add_executable(test_beam_tracker tests/cpu/test_beam_tracker.cpp)
    target_link_libraries(test_beam_tracker PRIVATE beamformer_core)
    target_compile_options(test_beam_tracker PRIVATE -UNDEBUG)
    add_test(NAME beam_tracker COMMAND test_beam_tracker)

    # Extensive tests for the optimized CPU tracker (O1–O6 + back-compat).
    add_executable(test_cpu_opt_beam_tracker
                   tests/cpu/test_cpu_opt_beam_tracker.cpp)
    target_link_libraries(test_cpu_opt_beam_tracker PRIVATE beamformer_core)
    target_compile_options(test_cpu_opt_beam_tracker PRIVATE -UNDEBUG)
    if(BEAMFORMER_TRACKER_DEBUG)
        target_compile_definitions(test_cpu_opt_beam_tracker PRIVATE
            BEAMFORMER_TRACKER_DEBUG)
    endif()
    add_test(NAME cpu_opt_beam_tracker COMMAND test_cpu_opt_beam_tracker)
    set_tests_properties(cpu_opt_beam_tracker PROPERTIES
        LABELS "tracker;optimization;validation"
        TIMEOUT 120
    )

    # Regression tests for the optimized NAIVE tracker (src/beam_tracker_opt.cpp)
    # against the reference naive tracker (src/beam_tracker.cpp). Asserts
    # byte-for-bit equality `assert(naive == opt)` across stationary, drifting,
    # noise, multi-window, n_ant 32/64, integration_spectra=1, and validation
    # parity scenarios. The opt TU is intentionally built against the plain
    # beamformer_core (not the perf core) so the regression runs the same build
    # profile as the rest of the suite.
    add_executable(test_beam_tracker_opt tests/cpu/test_beam_tracker_opt.cpp)
    target_link_libraries(test_beam_tracker_opt PRIVATE beamformer_core)
    target_compile_options(test_beam_tracker_opt PRIVATE -UNDEBUG)
    add_test(NAME beam_tracker_opt COMMAND test_beam_tracker_opt)
    set_tests_properties(beam_tracker_opt PROPERTIES
        LABELS "tracker;optimization;validation"
        TIMEOUT 120
    )

    # Regression tests for the v2 optimized NAIVE tracker
    # (src/beam_tracker_opt_v2.cpp) against BOTH the reference naive tracker
    # (src/beam_tracker.cpp) and v1 (src/beam_tracker_opt.cpp). Asserts
    # byte-for-bit equality `assert(naive == v2)` AND `assert(v1 == v2)` across
    # the same scenario matrix as test_beam_tracker_opt (stationary, drifting,
    # noise, multi-window, n_ant 32/64, integration_spectra=1, validation
    # parity across all three paths, output-size contract). Built against the
    # plain beamformer_core for the same reason as the v1 test.
    add_executable(test_beam_tracker_opt_v2 tests/cpu/test_beam_tracker_opt_v2.cpp)
    target_link_libraries(test_beam_tracker_opt_v2 PRIVATE beamformer_core)
    target_compile_options(test_beam_tracker_opt_v2 PRIVATE -UNDEBUG)
    add_test(NAME beam_tracker_opt_v2 COMMAND test_beam_tracker_opt_v2)
    set_tests_properties(beam_tracker_opt_v2 PROPERTIES
        LABELS "tracker;optimization;validation"
        TIMEOUT 120
    )

    add_executable(test_temporal_integration tests/cpu/test_temporal_integration.cpp)
    target_link_libraries(test_temporal_integration PRIVATE beamformer_core)
    target_compile_options(test_temporal_integration PRIVATE -UNDEBUG)
    add_test(NAME temporal_integration COMMAND test_temporal_integration)

    add_executable(test_quantization tests/cpu/test_quantization.cpp)
    target_link_libraries(test_quantization PRIVATE beamformer_core)
    target_compile_options(test_quantization PRIVATE -UNDEBUG)
    add_test(NAME quantization COMMAND test_quantization)

    add_executable(test_cuda_frame_contract tests/cpu/test_cuda_frame_contract.cpp)
    target_link_libraries(test_cuda_frame_contract PRIVATE beamformer_core)
    target_compile_options(test_cuda_frame_contract PRIVATE -UNDEBUG)
    add_test(NAME cuda_frame_contract COMMAND test_cuda_frame_contract)
    set_tests_properties(cuda_frame_contract PROPERTIES LABELS "contract")

    if(BEAMFORMER_HAS_CUDA)
        add_executable(test_cuda_point_source tests/cuda/test_cuda_point_source.cpp)
        target_link_libraries(test_cuda_point_source PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_point_source PRIVATE -UNDEBUG)
        add_test(NAME cuda_point_source COMMAND test_cuda_point_source)
        set_tests_properties(cuda_point_source PROPERTIES
            LABELS "cuda;physics"
            TIMEOUT 60
        )

        add_executable(test_cuda_packed tests/cuda/test_cuda_packed.cu)
        target_link_libraries(test_cuda_packed PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_packed PRIVATE -UNDEBUG)
        add_test(NAME cuda_packed COMMAND test_cuda_packed)
        set_tests_properties(cuda_packed PROPERTIES
            LABELS "cuda;packed;validation"
            TIMEOUT 60
        )

        add_executable(test_cuda_temporal_integration
                       tests/cuda/test_cuda_temporal_integration.cpp)
        target_link_libraries(test_cuda_temporal_integration PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_temporal_integration PRIVATE -UNDEBUG)
        add_test(NAME cuda_temporal_integration COMMAND test_cuda_temporal_integration)
        set_tests_properties(cuda_temporal_integration PROPERTIES
            LABELS "cuda;temporal-integration;validation"
            TIMEOUT 60
        )

        add_executable(test_cuda_quantization tests/cuda/test_cuda_quantization.cpp)
        target_link_libraries(test_cuda_quantization PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_quantization PRIVATE -UNDEBUG)
        add_test(NAME cuda_quantization COMMAND test_cuda_quantization)
        set_tests_properties(cuda_quantization PROPERTIES
            LABELS "cuda;quantization;validation"
            TIMEOUT 60
        )

        add_executable(test_cuda_frame_pipeline
                       tests/cuda/test_cuda_frame_pipeline.cpp)
        set_source_files_properties(
            tests/cuda/test_cuda_frame_pipeline.cpp PROPERTIES LANGUAGE CUDA)
        target_link_libraries(test_cuda_frame_pipeline PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_frame_pipeline PRIVATE -UNDEBUG)
        add_test(NAME cuda_frame_pipeline COMMAND test_cuda_frame_pipeline)
        set_tests_properties(cuda_frame_pipeline PROPERTIES
            LABELS "cuda;contract;validation"
            TIMEOUT 60
        )

        add_executable(test_cuda_offline_runner
                       tests/cuda/test_cuda_offline_runner.cpp)
        target_link_libraries(test_cuda_offline_runner PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_offline_runner PRIVATE -UNDEBUG)
        add_test(NAME cuda_offline_runner COMMAND test_cuda_offline_runner)
        set_tests_properties(cuda_offline_runner PROPERTIES
            LABELS "cuda;contract;validation"
            TIMEOUT 60
        )
        add_executable(test_cuda_tracker_v2
                       tests/cuda/test_cuda_tracker_v2.cpp)
        target_link_libraries(test_cuda_tracker_v2 PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_tracker_v2 PRIVATE -UNDEBUG)
        add_test(NAME cuda_tracker_v2 COMMAND test_cuda_tracker_v2)
        set_tests_properties(cuda_tracker_v2 PROPERTIES
            LABELS "cuda;tracker;validation"
            TIMEOUT 120
        )

        add_executable(test_cuda_beam_tracker_fused_warp_shuffle
                       tests/cuda/test_cuda_beam_tracker_fused_warp_shuffle.cpp)
        target_link_libraries(test_cuda_beam_tracker_fused_warp_shuffle PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_beam_tracker_fused_warp_shuffle PRIVATE -UNDEBUG)
        add_test(NAME cuda_beam_tracker_fused_warp_shuffle COMMAND test_cuda_beam_tracker_fused_warp_shuffle)
        set_tests_properties(cuda_beam_tracker_fused_warp_shuffle PROPERTIES
            LABELS "cuda;tracker;validation"
            TIMEOUT 120
        )

        add_executable(test_cuda_beam_tracker_v3
                       tests/cuda/test_cuda_beam_tracker_v3.cpp)
        target_link_libraries(test_cuda_beam_tracker_v3 PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_beam_tracker_v3 PRIVATE -UNDEBUG)
        add_test(NAME cuda_beam_tracker_v3 COMMAND test_cuda_beam_tracker_v3)
        set_tests_properties(cuda_beam_tracker_v3 PROPERTIES
            LABELS "cuda;tracker;validation"
            TIMEOUT 120
        )

        add_executable(test_cuda_beam_tracker_v4
                       tests/cuda/test_cuda_beam_tracker_v4.cpp)
        target_link_libraries(test_cuda_beam_tracker_v4 PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_beam_tracker_v4 PRIVATE -UNDEBUG)
        add_test(NAME cuda_beam_tracker_v4 COMMAND test_cuda_beam_tracker_v4)
        set_tests_properties(cuda_beam_tracker_v4 PROPERTIES
            LABELS "cuda;tracker;validation"
            TIMEOUT 120
        )

        add_executable(test_cuda_beam_tracker_v5
                       tests/cuda/test_cuda_beam_tracker_v5.cpp)
        target_link_libraries(test_cuda_beam_tracker_v5 PRIVATE beamformer_cuda_core)
        target_compile_options(test_cuda_beam_tracker_v5 PRIVATE -UNDEBUG)
        add_test(NAME cuda_beam_tracker_v5 COMMAND test_cuda_beam_tracker_v5)
        set_tests_properties(cuda_beam_tracker_v5 PROPERTIES
            LABELS "cuda;tracker;validation"
            TIMEOUT 120
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