seeds=(
  110 111 112 113 114
)

for seed in "${seeds[@]}"; do
  python examples/memr.py --env_name Hopper-v2 --seed $seed --logger_path 'benchmark_results' --epochs 100
done

for seed in "${seeds[@]}"; do
  python examples/memr.py --env_name Walker2d-v2 --seed $seed --logger_path 'benchmark_results' --epochs 300
done

for seed in "${seeds[@]}"; do
  python examples/memr.py --env_name AntTruncatedObs-v2 --seed $seed --logger_path 'benchmark_results' --epochs 300
done

for seed in "${seeds[@]}"; do
  python examples/memr.py --env_name HalfCheetah-v2 --seed $seed --logger_path 'benchmark_results' --epochs 300
done
