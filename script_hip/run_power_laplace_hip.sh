#!/bin/bash
MTX_HOME="/people/firo017/pact_exp/LAP/lap_matrices/" # "/qfs/projects/eracle/data/"
RES_OUT_HOME="./output_results"
HIP_POWER_SCRIPT="../continous.sh"
RES_HOME="./power_results"
SCRIPT="../../lap_hip_laplacian " ## This is were the command go
PRECOND=('GS_std' 'GS_it' 'GS_it2' 'it_jacobi' 'line_jacobi' 'ichol')
MATRIX=('delaunay_n24/delaunay_n24.mtx' 't60k/t60k.mtx' 'hugebubbles-00000/hugebubbles-00000.mtx' 'adaptive/adaptive.mtx')
#MATRIX=('Fault_639/Fault_639.mtx' 'G3_circuit/G3_circuit.mtx' 'Serena/Serena.mtx' 'thermal2/thermal2.mtx' 'af_0_k101/af_0_k101.mtx' 'Hook_1498/Hook_1498.mtx')
#MATRIX=('Fault_639/Fault_639.mtx')
#RHS=('' '' '' 'thermal2/thermal2_b.mtx' 'af_0_k101/af_0_k101_b.mtx' '')
RHS=('' '' '' '')

declare -A ARGX

ARGX+=(["${MATRIX[0]}_${PRECOND[0]}"]='6 3') 
ARGX+=(["${MATRIX[0]}_${PRECOND[1]}"]='15 1') 
ARGX+=(["${MATRIX[0]}_${PRECOND[2]}"]='6 3') 
ARGX+=(["${MATRIX[0]}_${PRECOND[3]}"]='1 1') 
ARGX+=(["${MATRIX[0]}_${PRECOND[4]}"]='1 1') 
ARGX+=(["${MATRIX[0]}_${PRECOND[5]}"]='-1') 

ARGX+=(["${MATRIX[1]}_${PRECOND[0]}"]='6 3') 
ARGX+=(["${MATRIX[1]}_${PRECOND[1]}"]='50 1') 
ARGX+=(["${MATRIX[1]}_${PRECOND[2]}"]='1 1') 
ARGX+=(["${MATRIX[1]}_${PRECOND[3]}"]='1 1') 
ARGX+=(["${MATRIX[1]}_${PRECOND[4]}"]='1 1')
ARGX+=(["${MATRIX[1]}_${PRECOND[5]}"]='-1') 

ARGX+=(["${MATRIX[2]}_${PRECOND[0]}"]='6 3') 
ARGX+=(["${MATRIX[2]}_${PRECOND[1]}"]='50 1')
ARGX+=(["${MATRIX[2]}_${PRECOND[2]}"]='6 3') 
ARGX+=(["${MATRIX[2]}_${PRECOND[3]}"]='1 1')
ARGX+=(["${MATRIX[2]}_${PRECOND[4]}"]='-1') 
ARGX+=(["${MATRIX[2]}_${PRECOND[5]}"]='-1') 

ARGX+=(["${MATRIX[3]}_${PRECOND[0]}"]='6 3')
ARGX+=(["${MATRIX[3]}_${PRECOND[1]}"]='100 1')
ARGX+=(["${MATRIX[3]}_${PRECOND[2]}"]='2 1')
ARGX+=(["${MATRIX[3]}_${PRECOND[3]}"]='15 15')
ARGX+=(["${MATRIX[3]}_${PRECOND[4]}"]='1 1')
ARGX+=(["${MATRIX[3]}_${PRECOND[5]}"]='-1')


NUM_GPUS=1
RES_COLLECTIVE="./hip_laplacian_results/"
#echo "Setting AMD  devices"
#devices=0
#for ((i = 1; i < ${NUM_GPUS} ; i++)); do
#	devices=${devices},${i}
#done
#export CUDA_VISIBLE_DEVICES=${devices}
#sleep 1s
for idx in {1..5}; do
for precond in "${PRECOND[@]}"; do
	x=-1
	for mtx in "${MATRIX[@]}"; do
		key="${mtx}_${precond}"
		let x++
		if [[ ${ARGX[${key}]}  == '-1' ]]; then continue; fi
		echo "Preconditioner = ${precond}; Matrix = ${mtx}; Repetition: $idx"
		echo "Creating directories"
		sleep 1s
		mtxs=${mtx/\//_}
		dir_name="PC_${precond}_MTX_${mtxs}_REP_${idx}_LAP"
		echo "Creating directory: " $dir_name
		mkdir -p ${dir_name}
		rm -f ${dir_name}/*
		cd ${dir_name}
		sleep 1s
		mkdir -p ${RES_HOME}
		mkdir -p ${RES_OUT_HOME}
		rm -f ${RES_HOME}/*
		rm -f ${RES_OUT_HOME}/*
		
		echo "Starting rocm-smi"
		sleep 1s
		power_sids=()
		for ((i = 0; i < ${NUM_GPUS} ; i++)); do
			${HIP_POWER_SCRIPT} > ${RES_HOME}/gpu_${i}.txt &
			power_sids+=($!)
		done
		
		echo "Running the code"
		mtxr=${MTX_HOME}/${mtx}

                hrsxr=''
                if [[ ${RHS[$x]}  != '' ]]; then
                        hrsxr=${MTX_HOME}/${RHS[$x]}
                fi
		sleep 1s
		full_command="${SCRIPT} ${mtxr} ${precond} 1e-12 25000 ${ARGX[${key}]} ${hrsxr}"
		echo "CMD: " + $full_command
		${full_command} > ${RES_OUT_HOME}/output.txt
		
		echo "Killing all rocm-smi"
		sleep 1s
		for sid in "${power_sids[@]}"; do
			kill ${sid}
		done
		pkill rocm-smi
		
		echo "Done"
		sleep 5s
		cd ..
	done
done
done

mkdir -p ${RES_COLLECTIVE}
rm -rf ${RES_COLLECTIVE}/*
mv PC_* ${RES_COLLECTIVE}
