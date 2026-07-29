#!/bin/bash
#SBATCH -J cola_score                   # Job name
#SBATCH -o watch_folder/%x_%A_%a.out    # log file (out)
#SBATCH -e watch_folder/%x_%A_%a.err    # log file (err)
#SBATCH -N 1                            # Total number of nodes requested
#SBATCH --get-user-env                  # retrieve the users login environment
#SBATCH --mem=48G                       # server memory requested (per node)
#SBATCH -t 8:00:00                      # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov,gpu         # kuleshov first: it schedules immediately
#SBATCH --exclude=sun-compute-03,snavely-compute-02,portal-compute-01,lancer-compute-01,sun-compute-01
#SBATCH --constraint="[a6000|a5000|a100|h100|h200|3090|a40|6000ada]"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                    # Type/number of GPUs needed
#SBATCH --open-mode=append              # Do not overwrite logs
#SBATCH --requeue                       # Requeue upon preemption
#SBATCH --array=0-20%10

# CoLA linguistic-acceptability scoring of the 17 BD3-LM sampler cells plus 4 anchors.
# One array task per cell; each task scores both segmentation schemes with both
# gate-validated checkpoints (primary deberta-v3-large-cola MCC 0.7193, secondary
# textattack/roberta-base-CoLA MCC 0.6383). See cola_gate.py for the validation gate.

set -u

PY=/home/gt345/.conda/envs/bd3lm/bin/python
ROOT=/home/gt345/projects/scaling/duel
CQ=${ROOT}/scripts/cola_quality
OUT=${CQ}/out/scores
ANCHORS=${CQ}/out/anchors.jsonl

mkdir -p "${OUT}" "${ROOT}/watch_folder"

CELLS=(
  block-greedy_k1 block-greedy_k2 block-greedy_k4 block-greedy_k8
  block-left-to-right_k1 block-left-to-right_k2 block-left-to-right_k4 block-left-to-right_k8
  block-probability-margin_k1 block-probability-margin_k2 block-probability-margin_k4 block-probability-margin_k8
  block-confidence-threshold_k0.045 block-confidence-threshold_k0.08
  block-confidence-threshold_k0.16 block-confidence-threshold_k1.0
  uniform
  anchor-owt-real anchor-owt-token-shuffled anchor-owt-word-shuffled anchor-owt-repeat
)

CELL=${CELLS[${SLURM_ARRAY_TASK_ID}]}
echo "=== array task ${SLURM_ARRAY_TASK_ID}: cell=${CELL} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "${ROOT}" || exit 1

for MODEL in yiiino/deberta-v3-large-cola textattack/roberta-base-CoLA; do
  for SCHEME in sentence window; do
    TAG="${CELL}__${MODEL//\//_}__${SCHEME}"
    if [ -s "${OUT}/${TAG}.npz" ]; then
      echo "--- SKIP ${TAG} (already present) ---"
      continue
    fi
    echo "--- ${TAG} ---"
    ${PY} -u "${CQ}/cola_score.py" \
      --cell "${CELL}" \
      --model "${MODEL}" \
      --scheme "${SCHEME}" \
      --anchors "${ANCHORS}" \
      --out "${OUT}" \
      --batch-size 128 || { echo "FAILED ${TAG}"; exit 1; }
  done
done

echo "=== done ${CELL} ==="
