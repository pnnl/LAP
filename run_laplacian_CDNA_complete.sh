#!/bin/bash
#SBATCH -p MI300x
#SBATCH -N 1 
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=mi300X_Laplacian.txt
#SBATCH --nodelist=TheraC17
module load rocm gcc python

MTX_HOME="/home/kswirydo/LAP_PNNL/"
RES_OUT_HOME="./output_results"
RES_HOME="./power_results"
GPU_NAME="MI300x"

SCRIPT="/home/kswirydo/LAP_PNNL/lap_hip_laplacian " ## This is were the command go
PRECOND=('GS_std' 'GS_it' 'GS_it2' 'it_jacobi' 'line_jacobi' 'ichol')

MATRIX=('delaunay_n24/delaunay_n24.mtx' 't60k/t60k.mtx' 'hugebubbles-00000/hugebubbles-00000.mtx' 'adaptive/adaptive.mtx' 'road_central/road_central.mtx' 'road_usa/road_usa.mtx' 'italy_osm/italy_osm.mtx')
RHS=('' '' '' '' '' '' '' '')

declare -A ARGX
# delaunay
#ARGX+=(["${MATRIX[0]}_${PRECOND[0]}"]='6 3')
ARGX+=(["${MATRIX[0]}_${PRECOND[0]}"]='-1')
ARGX+=(["${MATRIX[0]}_${PRECOND[1]}"]='15 1')
ARGX+=(["${MATRIX[0]}_${PRECOND[2]}"]='6 3')
ARGX+=(["${MATRIX[0]}_${PRECOND[3]}"]='1 1')
ARGX+=(["${MATRIX[0]}_${PRECOND[4]}"]='1 1')
ARGX+=(["${MATRIX[0]}_${PRECOND[5]}"]='-1')
#t60k
ARGX+=(["${MATRIX[1]}_${PRECOND[0]}"]='6 3')
ARGX+=(["${MATRIX[1]}_${PRECOND[1]}"]='50 1')
ARGX+=(["${MATRIX[1]}_${PRECOND[2]}"]='1 1')
ARGX+=(["${MATRIX[1]}_${PRECOND[3]}"]='1 1')
ARGX+=(["${MATRIX[1]}_${PRECOND[4]}"]='1 1')
ARGX+=(["${MATRIX[1]}_${PRECOND[5]}"]='-1')
#hugebubbles
#ARGX+=(["${MATRIX[2]}_${PRECOND[0]}"]='6 3')
ARGX+=(["${MATRIX[2]}_${PRECOND[0]}"]='-1')
ARGX+=(["${MATRIX[2]}_${PRECOND[1]}"]='50 1')
ARGX+=(["${MATRIX[2]}_${PRECOND[2]}"]='6 3')
ARGX+=(["${MATRIX[2]}_${PRECOND[3]}"]='-1')
ARGX+=(["${MATRIX[2]}_${PRECOND[4]}"]='-1')
ARGX+=(["${MATRIX[2]}_${PRECOND[5]}"]='-1')
#adaptive
ARGX+=(["${MATRIX[3]}_${PRECOND[0]}"]='-1')
#ARGX+=(["${MATRIX[3]}_${PRECOND[0]}"]='6 3')
ARGX+=(["${MATRIX[3]}_${PRECOND[1]}"]='-1')
ARGX+=(["${MATRIX[3]}_${PRECOND[2]}"]='2 1')
ARGX+=(["${MATRIX[3]}_${PRECOND[3]}"]='1 1')
ARGX+=(["${MATRIX[3]}_${PRECOND[4]}"]='1 1')
ARGX+=(["${MATRIX[3]}_${PRECOND[5]}"]='-1')

#road central

#ARGX+=(["${MATRIX[4]}_${PRECOND[0]}"]='2 2')
ARGX+=(["${MATRIX[4]}_${PRECOND[0]}"]='-1')
ARGX+=(["${MATRIX[4]}_${PRECOND[1]}"]='4 0')
ARGX+=(["${MATRIX[4]}_${PRECOND[2]}"]='1 1')
ARGX+=(["${MATRIX[4]}_${PRECOND[3]}"]='4 4')
ARGX+=(["${MATRIX[4]}_${PRECOND[4]}"]='-1')
ARGX+=(["${MATRIX[4]}_${PRECOND[5]}"]='-1')
#road_usa

#ARGX+=(["${MATRIX[5]}_${PRECOND[0]}"]='6 3')
ARGX+=(["${MATRIX[5]}_${PRECOND[0]}"]='-1')
ARGX+=(["${MATRIX[5]}_${PRECOND[1]}"]='8 0')
ARGX+=(["${MATRIX[5]}_${PRECOND[2]}"]='3 1')
ARGX+=(["${MATRIX[5]}_${PRECOND[3]}"]='8 8')
ARGX+=(["${MATRIX[5]}_${PRECOND[4]}"]='-1')
ARGX+=(["${MATRIX[5]}_${PRECOND[5]}"]='-1')
#italy_osm

ARGX+=(["${MATRIX[6]}_${PRECOND[0]}"]='-1')
#ARGX+=(["${MATRIX[6]}_${PRECOND[0]}"]='6 6')
ARGX+=(["${MATRIX[6]}_${PRECOND[1]}"]='16 0')
ARGX+=(["${MATRIX[6]}_${PRECOND[2]}"]='4 4')
ARGX+=(["${MATRIX[6]}_${PRECOND[3]}"]='16 16')
ARGX+=(["${MATRIX[6]}_${PRECOND[4]}"]='-1')
ARGX+=(["${MATRIX[6]}_${PRECOND[5]}"]='-1')
NUM_GPUS=1
RES_COLLECTIVE="./rdna_lapl_results/"
sleep 1s
for idx in {1..5}; do
  for precond in "${PRECOND[@]}"; do
    x=0
    for mtx in "${MATRIX[@]}"; do
     cd ${MTX_HOME}
      key="${mtx}_${precond}"
      if [[ ${ARGX[${key}]}  == '-1' ]]; then	
			let x++
	continue; fi
      echo "Preconditioner = ${precond}; Matrix = ${mtx}; Repetition: $idx"
      echo "Creating directories"
      sleep 1s
      mtxs=`dirname $mtx`
      dir_name="REP_${idx}_${GPU_NAME}_LAPLACIAN/${mtxs}/"
      mkdir -p ${dir_name}
      
     echo "Entering directory: " $dir_name
      cd ${dir_name}
      sleep 1s
      mkdir -p ${RES_HOME}
      mkdir -p ${RES_OUT_HOME}
 #     rm -rf ${RES_HOME}/*
 #     rm -rf ${RES_OUT_HOME}/*

      echo "Starting power collecion"
      sleep 1s
      power_sids=()
      for ((i = 0; i < 1 ; i++)); do
	echo "Will be putting results in "${RES_HOME}/${precond}.txt
        /home/kswirydo/power_analysis/power_profiler/power_profiler  ${RES_HOME}/${precond}.txt 25000 20 &
	power_sids+=($!)
      done

      echo "Running the code"
      mtxr=${MTX_HOME}/${mtx}
      hrsxr=''
      sleep 1s
      full_command="${SCRIPT} ${mtxr} ${precond} 1e-12 25000 ${ARGX[${key}]} ${hrsxr}"
      echo "CMD: " + $full_command
      echo "Will be putting (numerical) results in " ${RES_OUT_HOME}/$precond.txt
      ${full_command} > ${RES_OUT_HOME}/$precond.txt

      echo "Killing power collectin"
      sleep 1s
      for sid in "${power_sids[@]}"; do
	kill ${sid}
      done
      pkill tegrastats

      echo "Done"
      sleep 1s
      cd ..
      let x++
    done
  done
done

mkdir -p ${RES_COLLECTIVE}
rm -rf ${RES_COLLECTIVE}/*
mv PC_* ${RES_COLLECTIVE}
