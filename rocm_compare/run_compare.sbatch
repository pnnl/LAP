#!/bin/bash
#SBATCH -p MI355x
#SBATCH -N 1
#SBATCH --gpus-per-node 1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
# Submit from this directory:
#   sbatch -J <name> -t <time> -o results/<name>.log run_compare.sbatch <tag> <driver> <driver args...>
# <driver> is laplacian, cg, spmv or lobpcg. Runs build_<config>/lap_hip_<config>_<driver> for each
# config in $CONFIGS (default "custom rocm72") one after another on the same GPU, each in its own
# module environment, and writes results/<tag>_<config>.txt.
# PROF=1 runs each under rocprofv3 (kernel trace and stats) into results/<tag>_<config>_prof/.
TAG=$1; DRIVER=$2; shift 2
DIR=$SLURM_SUBMIT_DIR
REPO=$(cd $DIR/.. && pwd)
mkdir -p $DIR/results
cd $REPO
echo "node: $(hostname)"

for cfg in ${CONFIGS:-custom rocm72}; do
  (
    module purge
    if [ "$cfg" == rocm72 ]; then module load rocm/7.2 gcc; else module load rocm/10.1.0; fi
    bin=$DIR/build_$cfg/lap_hip_${cfg}_$DRIVER
    out=$DIR/results/${TAG}_$cfg.txt
    prof=""
    if [ -n "$PROF" ]; then
      prof="rocprofv3 --kernel-trace --stats -S -u msec -f csv -d $DIR/results/${TAG}_${cfg}_prof -o run --"
    fi
    echo "$(ldd $bin | grep -o 'librocsparse.so.1 => [^ ]*')" > $out
    t0=$(date +%s.%N)
    stdbuf -oL $prof $bin "$@" >> $out 2>&1
    rc=$?
    t1=$(date +%s.%N)
    awk -v a=$t0 -v b=$t1 'BEGIN{printf "wall_total_s=%.1f\n", b-a}' >> $out
    echo "exit=$rc" >> $out
  )
done
echo "done"
