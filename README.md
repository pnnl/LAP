# LAP
GPU-Resident Preconditioners for Conjugate Gradient Solver

## Requirements

Depending on your target backend, you will need:

- **CUDA backend**: NVIDIA GPU with CUDA toolkit (cuBLAS, cuSPARSE)
- **HIP backend**: AMD GPU with ROCm (rocBLAS, rocSPARSE)
- **OpenMP backend**: GCC with OpenMP support
- **CPU-only backend**: GCC compiler

## Compilation

The unified Makefile supports multiple backends. Use the `BACKEND` variable to select your target platform.

### Build Commands

```bash
# Build all executables with CUDA (NVIDIA GPUs)
make BACKEND=cuda

# Build all executables with HIP (AMD GPUs)
make BACKEND=hip

# Build all executables with OpenMP (CPU parallel)
make BACKEND=openmp

# Build all executables without accelerators (CPU serial)
make BACKEND=noacc
```

### Build Specific Targets

```bash
# Build only the Laplacian solver
make BACKEND=hip laplacian

# Build only the CG solver
make BACKEND=hip cg

# Build only the SpMV driver
make BACKEND=hip spmv

# Build only the LOBPCG eigenvalue solver
make BACKEND=hip lobpcg
```

### Optional Variables

```bash
# Specify CUDA architecture (default: sm_80)
make BACKEND=cuda CUDA_ARCH=sm_70

# Specify ROCm installation path (default: /opt/rocm)
make BACKEND=hip ROCM_PATH=/opt/rocm-5.0

# Enable debug build
make BACKEND=cuda DEBUG=1
```

### Clean Build Artifacts

```bash
make clean
```

## Usage

### Laplacian Solver

Solves graph Laplacian linear systems (converts adjacency matrix to Laplacian internally).

```bash
./lap_<backend>_laplacian <matrix.mtx> <preconditioner> <tolerance> <maxit> <M> <K>
```

**Arguments:**
- `matrix.mtx` - Adjacency matrix file in Matrix Market format
- `preconditioner` - Preconditioner type: `none`, `it_jacobi`, `line_jacobi`, `GS_it`, `GS_it2`, `GS_std`, `ichol`
- `tolerance` - Convergence tolerance (e.g., `1e-12`)
- `maxit` - Maximum CG iterations
- `M` - Outer iterations for preconditioner
- `K` - Inner iterations for preconditioner

**Example:**
```bash
./lap_hip_laplacian road_usa/road_usa.mtx it_jacobi 1e-12 25000 8 8
```

### CG Solver

Solves general sparse linear systems using Preconditioned Conjugate Gradient.

```bash
./lap_<backend>_cg <matrix.mtx> <preconditioner> <tolerance> <maxit> <M> <K>
```

**Arguments:**
- `matrix.mtx` - SPD matrix file in Matrix Market format
- `preconditioner` - Preconditioner type: `none`, `it_jacobi`, `line_jacobi`, `GS_it`, `GS_it2`, `GS_std`
- `tolerance` - Convergence tolerance (e.g., `1e-12`)
- `maxit` - Maximum CG iterations
- `M` - Outer iterations for preconditioner
- `K` - Inner iterations for preconditioner

**Example:**
```bash
./lap_hip_cg thermal2/thermal2.mtx GS_it 1e-10 10000 4 4
```

### SpMV Driver

Benchmarks Sparse Matrix-Vector multiplication.

```bash
./lap_<backend>_spmv <matrix.mtx> <n_trials>
```

**Arguments:**
- `matrix.mtx` - Matrix file in Matrix Market format
- `n_trials` - Number of SpMV trials to run

**Example:**
```bash
./lap_hip_spmv Hook_1498/Hook_1498.mtx 1000
```

### LOBPCG Eigenvalue Solver

Computes smallest eigenvalues/eigenvectors using the Locally Optimal Block Preconditioned Conjugate Gradient method.

```bash
./lap_<backend>_lobpcg <matrix.mtx> <mode> <preconditioner> <tolerance> <maxit> <M> <K> <nev>
```

**Arguments:**
- `matrix.mtx` - Matrix file in Matrix Market format
- `mode` - Matrix mode: `normal` (use as-is) or `laplacian` (convert to graph Laplacian)
- `preconditioner` - Preconditioner type: `none`, `it_jacobi`, `line_jacobi`, `GS_it`, `GS_it2`, `GS_std`
- `tolerance` - Convergence tolerance (e.g., `1e-8`)
- `maxit` - Maximum LOBPCG iterations
- `M` - Outer iterations for preconditioner
- `K` - Inner iterations for preconditioner
- `nev` - Number of eigenvalues/eigenvectors to compute

**Example:**
```bash
./lap_hip_lobpcg delaunay_n24/delaunay_n24.mtx laplacian it_jacobi 1e-8 500 4 4 10
```

## Preconditioners

| Name | Description |
|------|-------------|
| `none` | No preconditioning |
| `it_jacobi` | Iterative Jacobi |
| `line_jacobi` | Line Jacobi |
| `GS_it` | Iterative Gauss-Seidel |
| `GS_it2` | Iterative Gauss-Seidel (variant 2) |
| `GS_std` | Standard Gauss-Seidel |
| `ichol` | Incomplete Cholesky (Laplacian only) |

## License
LAP comes with [BSD-3 license](https://en.wikipedia.org/wiki/BSD_licenses).
See the [license](https://github.com/pnnl/LAP/blob/develop/LICENSE.rst) for further details.

## Acknowledgments
This research used funding and resources supported by the U.S. DOE Office of Science, Office of Advanced Scientific Computing Research, under award 66150: ``CENATE - Center for Advanced Architecture Evaluation''. The Pacific Northwest National Laboratory is operated by Battelle for the U.S. Department of Energy under contract DE-AC05-76RL01830. PNNL software release IPIDs: 32592 and 33233-E.