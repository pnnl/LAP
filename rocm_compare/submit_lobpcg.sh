#!/bin/bash
# LOBPCG with GS_std and the run_all_MI250_lobpcg.sh settings (M=6, K=3, tol 1e-6, maxit 10000,
# nev 5, seed 12345, verbose 1): custom rocSPARSE build vs legacy ROCm 7.2.1 build.
# SPD matrices run in normal mode, graphs in laplacian mode.
# Build first: make -f Makefile.custom and make -f Makefile.rocm72 (see each makefile for its module).
cd "$(dirname "$0")"
mkdir -p results
export SLURM_MPI_TYPE=none

sbatch -J lobpcg_adaptive   -t 08:00:00 -o results/lobpcg_adaptive.log   run_compare.sbatch lobpcg_adaptive   lobpcg adaptive/adaptive.mtx     laplacian GS_std 1e-6 10000 6 3 5 12345 1
sbatch -J lobpcg_g3_circuit -t 08:00:00 -o results/lobpcg_g3_circuit.log run_compare.sbatch lobpcg_g3_circuit lobpcg G3_circuit/G3_circuit.mtx normal    GS_std 1e-6 10000 6 3 5 12345 1
sbatch -J lobpcg_serena     -t 08:00:00 -o results/lobpcg_serena.log     run_compare.sbatch lobpcg_serena     lobpcg Serena/Serena.mtx         normal    GS_std 1e-6 10000 6 3 5 12345 1
