envs=(
  pybullet_envs:HopperBulletEnv-v0
  pybullet_envs:Walker2DBulletEnv-v0
  pybullet_envs:HalfCheetahBulletEnv-v0
  pybullet_envs:AntBulletEnv-v0
)

algos=(ppo trpo sac td3)

seeds=(
  110 111 112 113 114 115 116 117 118 119
)

for env in "${envs[@]}"; do
  for algo in "${algos[@]}"; do
    for seed in "${seeds[@]}"; do
      python -m rlutils.run $algo --env_name $env --seed $seed
    done
  done
done
