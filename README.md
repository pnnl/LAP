# LAP
GPU-Resident Preconditioners for Conjugate Gradient and LOBPCG Eigenvalue Solvers

## Overview

LAP provides GPU-accelerated iterative solvers for sparse linear systems and eigenvalue problems:

- **CG (Conjugate Gradient)**: Solves sparse linear systems Ax = b
- **LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient)**: Computes smallest eigenvalues/eigenvectors of sparse matrices

Both solvers support multiple preconditioners optimized for GPU execution.

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

# Build specific target only
make BACKEND=hip lobpcg    # Only LOBPCG solver
make BACKEND=hip cg        # Only CG solver

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

Both solvers accept matrices in Matrix Market (.mtx) format. Matrices should be symmetric for eigenvalue problems.

### CG Solver (Linear Systems)

Solves Ax = b where A is a sparse SPD matrix.

```bash
./lap_hip_cg <matrix.mtx> <mode> <preconditioner> <tolerance> <max_iter> <M> <K>
```

**Parameters:**
- `matrix.mtx`: Path to Matrix Market file
- `mode`: `laplacian` (compute graph Laplacian) or `normal` (use matrix as-is)
- `preconditioner`: `GS_std`, `GS_it`, `it_jacobi`, `line_jacobi`, `ichol`, or `none`
- `tolerance`: Convergence tolerance (e.g., `1e-6`)
- `max_iter`: Maximum iterations
- `M`: Outer iterations for preconditioner
- `K`: Inner iterations for preconditioner

**Examples:**

```bash
# CG with Gauss-Seidel preconditioner on graph Laplacian
./lap_hip_cg thermal2.mtx laplacian GS_it 1e-6 10000 6 3

# CG with iterative Jacobi preconditioner
./lap_hip_cg G3_circuit.mtx normal it_jacobi 1e-6 10000 25 25

# CG without preconditioner
./lap_hip_cg matrix.mtx normal none 1e-6 10000 1 1
```

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
```

### Preconditioner Selection Guide

| Preconditioner | Best For | Parameters |
|---------------|----------|------------|
| `GS_std` | General matrices | M=6, K=3 |
| `GS_it` | Well-conditioned matrices | M=6, K=3 |
| `it_jacobi` | Diagonally dominant matrices | M=25, K=25 |
| `line_jacobi` | Structured grids | M=1, K=1 |
| `ichol` | SPD matrices | - |
| `none` | Testing/comparison | - |

### Example Output (LOBPCG)

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
