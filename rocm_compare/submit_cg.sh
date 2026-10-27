#!/bin/bash
# CG with GS_std (M=1, K=1), tol 1e-10: custom rocSPARSE build vs legacy ROCm 7.2.1 build.
# Serena and G3_circuit use lap_hip_cg; adaptive uses lap_hip_laplacian (CG on the graph Laplacian).
# Build first: make -f Makefile.custom and make -f Makefile.rocm72 (see each makefile for its module).
cd "$(dirname "$0")"
mkdir -p results
export SLURM_MPI_TYPE=none

sbatch -J cg_serena     -t 01:00:00 -o results/cg_serena.log     run_compare.sbatch cg_serena     cg        Serena/Serena.mtx         GS_std 1e-10 2000 1 1
sbatch -J cg_g3_circuit -t 01:00:00 -o results/cg_g3_circuit.log run_compare.sbatch cg_g3_circuit cg        G3_circuit/G3_circuit.mtx GS_std 1e-10 2000 1 1
sbatch -J cg_adaptive   -t 02:00:00 -o results/cg_adaptive.log   run_compare.sbatch cg_adaptive   laplacian adaptive/adaptive.mtx     GS_std 1e-10 10000 1 1
