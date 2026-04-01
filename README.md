# LAP
GPU-Resident Preconditioners for Conjugate Gradient and LOBPCG Eigenvalue Solvers

## Overview

LAP provides GPU-accelerated iterative solvers for sparse linear systems and eigenvalue problems:

- **CG (Conjugate Gradient)**: Solves sparse linear systems Ax = b
- **LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient)**: Computes smallest eigenvalues/eigenvectors of sparse matrices
- **SpMV**: Sparse matrix-vector multiplication benchmark

All solvers support multiple preconditioners optimized for GPU execution.

## Directory Structure

```
├── src/     # Common source files (solvers, preconditioners, drivers)
├── inc/     # Common header files
├── hip/     # HIP backend (AMD GPUs)
├── cuda/    # CUDA backend (NVIDIA GPUs)
├── omp/     # OpenMP backend (CPU parallel)
├── cpu/     # CPU-only backend (serial)
├── build/   # Object files (generated)
└── Makefile # Unified build system
```

## Building

### Prerequisites

- **HIP backend**: ROCm 5.0+ (AMD GPUs)
- **CUDA backend**: CUDA 11.0+ (NVIDIA GPUs)
- **OpenMP backend**: GCC with OpenMP support
- **CPU backend**: Any C99 compiler

### Build Commands

```bash
# Build for AMD GPUs (HIP/ROCm)
make BACKEND=hip

# Build for NVIDIA GPUs (CUDA)
make BACKEND=cuda

# Build for CPU with OpenMP
make BACKEND=openmp

# Build for CPU only (serial)
make BACKEND=noacc

# Build specific targets
make BACKEND=hip lobpcg    # LOBPCG eigenvalue solver
make BACKEND=hip cg        # CG solver (general matrices)
make BACKEND=hip laplacian # CG solver (graph Laplacian)
make BACKEND=hip spmv      # SpMV benchmark

# Clean build artifacts
make clean

# Show help
make help
```

### Build Options

```bash
# Specify ROCm path (default: /opt/rocm)
make BACKEND=hip ROCM_PATH=/opt/rocm-5.4

# Specify CUDA architecture (default: sm_80)
make BACKEND=cuda CUDA_ARCH=sm_70

# Debug build
make BACKEND=hip DEBUG=1
```

## Running the Solvers

### Input Format

All tools accept matrices in Matrix Market (.mtx) format.

---

### CG Solver for Graph Laplacian

Solves Lx = b where L is the graph Laplacian computed from an adjacency matrix.

```bash
./lap_hip_laplacian <matrix.mtx> <preconditioner> <tolerance> <max_iter> <M> <K>
```

**Parameters:**
- `matrix.mtx`: Path to adjacency matrix in Matrix Market format
- `preconditioner`: `GS_std`, `GS_it`, `it_jacobi`, `line_jacobi`, `ichol`, or `none`
- `tolerance`: Convergence tolerance (e.g., `1e-6`)
- `max_iter`: Maximum iterations
- `M`: Outer iterations for preconditioner
- `K`: Inner iterations for preconditioner

**Examples:**

```bash
# Solve graph Laplacian with Gauss-Seidel preconditioner
./lap_hip_laplacian road_usa.mtx GS_it 1e-6 10000 6 3

# Using iterative Jacobi preconditioner
./lap_hip_laplacian delaunay_n20.mtx it_jacobi 1e-6 10000 25 25

# Without preconditioner
./lap_hip_laplacian matrix.mtx none 1e-6 10000 1 1
```

---

### CG Solver for General Matrices

Solves Ax = b where A is a sparse symmetric positive definite matrix.

```bash
./lap_hip_cg <matrix.mtx> <preconditioner> <tolerance> <max_iter> <M> <K>
```

**Parameters:**
- `matrix.mtx`: Path to SPD matrix in Matrix Market format
- `preconditioner`: `GS_std`, `GS_it`, `it_jacobi`, `line_jacobi`, `ichol`, or `none`
- `tolerance`: Convergence tolerance (e.g., `1e-6`)
- `max_iter`: Maximum iterations
- `M`: Outer iterations for preconditioner
- `K`: Inner iterations for preconditioner

**Examples:**

```bash
# Solve with Gauss-Seidel preconditioner
./lap_hip_cg thermal2.mtx GS_it 1e-6 10000 6 3

# Solve G3_circuit with iterative Jacobi
./lap_hip_cg G3_circuit.mtx it_jacobi 1e-6 10000 25 25

# Using standard Gauss-Seidel
./lap_hip_cg matrix.mtx GS_std 1e-6 5000 6 3

# Without preconditioner (for comparison)
./lap_hip_cg matrix.mtx none 1e-6 10000 1 1
```

**Example Output (CG):**

```
Solving CG linear system for thermal2.mtx

   Matrix size    : 1228045 x 1228045
   Matrix nnz     : 8580313
   Preconditioner : GS_it
   CG tolerance   : 1e-06
   CG maxit       : 10000
   M (outer)      : 6
   K (inner)      : 3

CG converged in 245 iterations
Final residual: 9.87e-07
Time: 3.45 seconds
```

---

### SpMV Benchmark

Benchmarks sparse matrix-vector multiplication performance.

```bash
./lap_hip_spmv <matrix.mtx> <num_trials>
```

**Parameters:**
- `matrix.mtx`: Path to matrix in Matrix Market format
- `num_trials`: Number of SpMV operations to perform

**Examples:**

```bash
# Run 1000 SpMV iterations on thermal2 matrix
./lap_hip_spmv thermal2.mtx 1000

# Quick benchmark with 100 iterations
./lap_hip_spmv G3_circuit.mtx 100

# Large benchmark
./lap_hip_spmv road_usa.mtx 5000
```

**Example Output (SpMV):**

```
Matrix info:

   Matrix size       : 1228045 x 1228045
   Matrix nnz        : 8580313
   Matrix nnz un     : 8580313
   Number of trials  : 1000

SpMV time: 0.523 seconds
Average time per SpMV: 0.523 ms
Throughput: 32.8 GFLOP/s
```

---

### LOBPCG Solver (Eigenvalue Problems)

Computes the k smallest eigenvalues and eigenvectors of a sparse symmetric matrix.

```bash
./lap_hip_lobpcg <matrix.mtx> <mode> <preconditioner> <tolerance> <max_iter> <M> <K> <nev> <seed> <verbose>
```

**Parameters:**
- `matrix.mtx`: Path to Matrix Market file
- `mode`: `laplacian` (compute graph Laplacian) or `normal` (use matrix as-is)
- `preconditioner`: `GS_std`, `GS_it`, `it_jacobi`, `line_jacobi`, `ichol`, or `none`
- `tolerance`: Convergence tolerance (e.g., `1e-6`)
- `max_iter`: Maximum iterations
- `M`: Outer iterations for preconditioner
- `K`: Inner iterations for preconditioner
- `nev`: Number of eigenvalues to compute
- `seed`: Random seed for initial vectors
- `verbose`: `0` (summary only) or `1` (per-iteration details)

**Examples:**

```bash
# Compute 5 smallest eigenvalues with GS_it preconditioner
./lap_hip_lobpcg thermal2.mtx normal GS_it 1e-6 10000 6 3 5 12345 1

# Compute eigenvalues of graph Laplacian (Fiedler values)
./lap_hip_lobpcg road_usa.mtx laplacian GS_std 1e-6 10000 6 3 5 12345 1

# Using iterative Jacobi (good for some matrices)
./lap_hip_lobpcg G3_circuit.mtx normal it_jacobi 1e-6 10000 25 25 5 12345 1

# Quick test with fewer iterations
./lap_hip_lobpcg matrix.mtx normal GS_it 1e-6 100 6 3 5 12345 1

# Compute 10 eigenvalues with quiet output
./lap_hip_lobpcg matrix.mtx normal GS_std 1e-6 5000 6 3 10 12345 0
```

**Example Output (LOBPCG):**

```
======================================================
LOBPCG Eigenvalue Solver
======================================================
  Matrix file       : thermal2.mtx
  Matrix size       : 1228045 x 1228045
  Num eigenvalues   : 5
  Preconditioner    : GS_it
  Tolerance         : 1e-06
======================================================

it    1  max residual = 2.103e+03
  eigenvalue 1: 4.8888e+02  residual: 2.061e+03
  ...
it  150  max residual = 9.54e-07
  ** Eigenpair 5 converged

======================================================
LOBPCG Summary Results
======================================================
  Iterations        : 150
  Converged         : 5 / 5
  Time (seconds)    : 12.34

Computed Eigenvalues:
  lambda[0] = 1.234567890123456e-03
  lambda[1] = 2.345678901234567e-03
  ...
======================================================
```

---

## Preconditioner Selection Guide

| Preconditioner | Best For | Typical Parameters |
|---------------|----------|-------------------|
| `GS_std` | General matrices | M=6, K=3 |
| `GS_it` | Well-conditioned matrices | M=6, K=3 |
| `it_jacobi` | Diagonally dominant matrices | M=25, K=25 |
| `line_jacobi` | Structured grids | M=1, K=1 |
| `ichol` | SPD matrices | - |
| `none` | Testing/baseline comparison | - |

## Supported Platforms

| Platform | Backend | Tested GPUs |
|----------|---------|-------------|
| AMD | HIP | MI250, MI300X |
| NVIDIA | CUDA | V100, A100 |
| CPU | OpenMP | x86_64 |

## License

LAP is distributed under the [BSD-3 license](https://opensource.org/licenses/BSD-3-Clause).
See [LICENSE](LICENSE.rst) for details.

## Acknowledgments

This research used funding and resources supported by the U.S. DOE Office of Science, Office of Advanced Scientific Computing Research, under award 66150: "CENATE - Center for Advanced Architecture Evaluation". 

The Pacific Northwest National Laboratory is operated by Battelle for the U.S. Department of Energy under contract DE-AC05-76RL01830.

PNNL software release IPIDs: 32592 and 33233-E.
