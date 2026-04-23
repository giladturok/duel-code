# batch sampling jobs (300 total)
for SEED in $(seq 1 12); do
  for LEN in 1024 2048; do
    sbatch ./scripts/gen/bd3lm_owt.sh $LEN $SEED 16
    sbatch ./scripts/gen/bd3lm_owt.sh $LEN $SEED 8
    sbatch ./scripts/gen/bd3lm_owt.sh $LEN $SEED 4
    sbatch ./scripts/gen/mdlm_owt.sh $LEN $SEED
    sbatch ./scripts/ar/gen_owt.sh $LEN $SEED
  done
  sbatch ./scripts/gen/sedd_owt.sh $SEED
done

python ./scripts/log_sample_stats.py
