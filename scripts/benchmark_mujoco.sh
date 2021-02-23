envs=(
  Hopper-v2
  Walker2d-v2
  HalfCheetah-v2
  Ant-v2
)

algos=$1

seeds=(
  110 111 112 113 114 115 116 117 118 119
)

for env in "${envs[@]}"; do
  for algo in "${algos[@]}"; do
    for seed in "${seeds[@]}"; do
      python -m rlutils.run $algo --env_name $env --seed $seed --logger_path 'benchmark_results' --epochs 600
    done
  done
done
