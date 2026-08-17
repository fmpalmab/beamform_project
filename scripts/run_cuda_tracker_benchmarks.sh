#!/bin/bash
# scripts/run_cuda_tracker_benchmarks.sh
#
# Automates the build and benchmarking of the CPU vs CUDA beam trackers.

set -e

echo "=== Building Project (Release Mode) ==="
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
# Compile using available CPU cores
make test_cuda_tracker_v2 benchmark_cuda_tracker_v2 -j$(nproc)

echo -e "\n=== Running Correctness Tests ==="
./test_cuda_tracker_v2

# Create a timestamped results directory (matching your existing structure)
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
OUTDIR="../results_cuda_benchmarks/${TIMESTAMP}"
mkdir -p "${OUTDIR}"

echo -e "\n=== Running Benchmarks ==="
echo "Saving results to ${OUTDIR}..."

# Sweep 1: 64 antennas, scaling CPU OpenMP threads for the baseline comparisons
for THREADS in 1 4 8 16; do
    echo "-> Running with 64 antennas, ${THREADS} threads..."
    ./benchmark_cuda_tracker_v2 --n-ant 64 --threads "${THREADS}" --outdir "${OUTDIR}"
done

# Sweep 2: 32 antennas, using max system threads
echo "-> Running with 32 antennas, max threads..."
./benchmark_cuda_tracker_v2 --n-ant 32 --threads 0 --outdir "${OUTDIR}"

echo -e "\n=== Benchmarks Complete ==="
echo "Summary saved to ${OUTDIR}/benchmark_cuda_tracker_v2.csv"