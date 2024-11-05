
#!/bin/bash


MTX_HOME="/home/kswirydo/LAP_PNNL"
RES_HOME="./power_results"


PRECOND=('GS_std' 'GS_it' 'GS_it2' 'it_jacobi' 'line_jacobi' 'ichol')
MATRIX=('Fault_639/Fault_639.mtx' 'G3_circuit/G3_circuit.mtx' 'Serena/Serena.mtx' 'thermal2/thermal2.mtx' 'af_0_k101/af_0_k101.mtx' 'Hook_1498/Hook_1498.mtx' 'af_5_k101/af_5_k101.mtx' )
declare -a time_res
mkdir -p plots_cg
cd plots_cg
for idx in {1..2}; do
  
  echo "Matrix,Preconditioner,Execution Time (in seconds)" >> /home/kswirydo/LAP_PNNL/CG_runtimes/runtime_MI300x_REP_${idx}.csv
  for mtx in "${MATRIX[@]}"; do

  mtxs=`dirname $mtx`
  dir_name="/home/kswirydo/LAP_PNNL/REP_${idx}_MI300x_CG/${mtxs}/power_results/"
  fig_name="${mtxs}_REP${idx}_MI300X"
  fig_title='"AMD MI300X: '${mtxs}'"'
  command="python3  /home/kswirydo/power_analysis/plot_power_profile_multi.py --folder ${dir_name} --figure $fig_name  --title ${fig_title}"
  echo $command
  echo ${fig_title}    
  eval $command
#check for num iters and time
  dir_name_output="/home/kswirydo/LAP_PNNL/REP_${idx}_MI300x_CG/${mtxs}/output_results/"
  mapfile -t my_array < <( grep -A4 summary $dir_name_output/* | awk '$4{print $4 $5}' )
  echo ${my_array[@]}
  num_files=$(ls -1q $dir_name_output | wc -l)
  ii=0;
#Gauss-Seidel 0
#Two Stage GS 1
#Two Stage Jac+GS 2
#line Jacobi 3
#iterative Jacobi 4
#Incomple Cholesky 5

  for ((i=0; i < 6; i++)); do 
    time_res[$i]="";
  done
  for ((i=0; i < num_files; i++)); do
    let "ii=i*4"
    
    if [[ "${my_array[ii + 3]}" == "GS_std" ]]
    then
    time_res[0]=${my_array[ii + 1]}
    fi
    
    if [[ "${my_array[ii + 3]}" == "GS_it" ]]
    then
    time_res[1]=${my_array[ii + 1]}
    fi
 
    if [[ "${my_array[ii + 3]}" == "GS_it2" ]]
    then
    time_res[2]=${my_array[ii + 1]}
    fi
    
    if [[ "${my_array[ii + 3]}" == "line_jacobi" ]]
    then
    time_res[3]=${my_array[ii + 1]}
    fi
    
    if [[ "${my_array[ii + 3]}" == "it_jacobi" ]]
    then
    time_res[4]=${my_array[ii + 1]}
    fi

    if [[ "${my_array[ii + 3]}" == "ichol" ]]
    then
    time_res[5]=${my_array[ii + 1]}
    fi

    echo "Matrix name $mtxs Precon name:  ${my_array[ii + 3]} Time ${my_array[ii + 1]} Iters ${my_array[ii + 0]}   " 
    echo
   done
   echo $mtxs '| Gauss-Seidel |'  ${time_res[0]}
   echo $mtxs '| Two Stage GS |'  ${time_res[1]}
   echo $mtxs '| Two Stage Jac+GS |'  ${time_res[2]}
   echo $mtxs '| line Jacobi |' ${time_res[3]}
   echo $mtxs '| iterative Jacobi |' ${time_res[4]}
   echo $mtxs '| Incomple Cholesky |' ${time_res[5]}
   echo
   echo
   echo $mtxs ',Gauss-Seidel,'  ${time_res[0]} >> /home/kswirydo/LAP_PNNL/CG_runtimes/runtime_MI300x_REP_${idx}.csv
   echo $mtxs ',Two Stage GS,'  ${time_res[1]} >> /home/kswirydo/LAP_PNNL/CG_runtimes/runtime_MI300x_REP_${idx}.csv
   echo $mtxs ',Two Stage Jac+GS,'  ${time_res[2]} >> /home/kswirydo/LAP_PNNL/CG_runtimes/runtime_MI300x_REP_${idx}.csv
   echo $mtxs ',line Jacobi,' ${time_res[3]} >> /home/kswirydo/LAP_PNNL/CG_runtimes/runtime_MI300x_REP_${idx}.csv
   echo $mtxs ',iterative Jacobi,' ${time_res[4]} >> /home/kswirydo/LAP_PNNL/CG_runtimes/runtime_MI300x_REP_${idx}.csv
   echo $mtxs ',Incomple Cholesky,' ${time_res[5]} >> /home/kswirydo/LAP_PNNL/CG_runtimes/runtime_MI300x_REP_${idx}.csv
done
rm /home/kswirydo/LAP_PNNL/CG_energy/aaa_REP${idx}.csv
echo "Matrix,Preconditioner,Energy" >> /home/kswirydo/LAP_PNNL/CG_energy/aaa_REP${idx}.csv
cd /home/kswirydo/LAP_PNNL/CG_energy
rm energy_MI300x_REP_${idx}.csv
#cat *_REP${idx}.csv 
cat *_REP${idx}.csv >> energy_MI300x_REP_${idx}.csv
cd /home/kswirydo/LAP_PNNL/plots_cg/
done
