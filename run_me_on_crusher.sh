#!/bin/bash
#SBATCH -A CSC359
#SBATCH -J hipTest
#SBATCH -o %x-%j.out
#SBATCH -t 00:05:00
#SBATCH -p batch
#SBATCH -N 1


#srun -N1 -G1 ./lap_hip chesapeake.mtx GS_it
srun -N1 -G1 ./lap_hip coPapersCiteseer.mtx GS_std 1e-12 250
srun -N1 -G1 ./lap_hip coPapersCiteseer.mtx GS_it 1e-12 250
srun -N1 -G1 ./lap_hip coPapersCiteseer.mtx GS_it2 1e-12 250
srun -N1 -G1 ./lap_hip coPapersCiteseer.mtx line_jacobi 1e-12 250
srun -N1 -G1 ./lap_hip coPapersCiteseer.mtx it_jacobi 1e-12 250

