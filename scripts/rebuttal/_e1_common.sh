# Shared helpers for the E1 (110M held-out likelihood cost) rebuttal tables.
# Sourced by e1_duel.sh / e1_elbo.sh. NOT executable on its own.
#
# NeurIPS 2026 rebuttal, Reviewer 2 W2/Q1:
#   "How much additional wall-clock time or GPU memory does DUEL require to
#    obtain the reported gap-closed estimates when using MDLM?"
#
# Design constraints (see duel-manuscript/rebuttal/e1_110m_results.md):
#  * ZERO edits to model/eval source. Peak memory comes from an out-of-process
#    nvidia-smi poller; per-batch timing comes from timestamping Lightning's
#    tqdm refreshes. Nothing here can change a reported number.
#  * loader.eval_batch_size is PINNED at 32 for every arm in every table.
#    exact_likelihood._pad_block_logits_to_full (:656-659) allocates
#    [B, 1024, 50258] fp32 = 206 MB/sequence -> 6.6 GB at B=32, 26 GB at B=128.
#  * strategy=single, trainer.devices=1. configs/strategy/ddp.yaml would build a
#    DDPStrategy even at devices=1 and NCCL-init-fails on several nodes.
#  * PYTORCH_CUDA_ALLOC_CONF identical on every arm or the memory column is void.

# Deliberately NO `set -e`: run_arm must survive a failing arm and keep sweeping.
set -uo pipefail

export TRITON_NUM_STAGES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
# `wandb=null` is NOT usable here. main.py:140 passes logger=wandb_logger to the
# Trainer; with None, this Lightning falls back to a CSVLogger that dies with
# "dict contains fields not in fieldnames: 'val/ppl_step'" the moment a step
# metric appears mid-run. WANDB_MODE=disabled keeps the paper's exact logger
# object (a real WandbLogger) while doing no network I/O.
# offline, not disabled: val/num_decoding_steps is logged on_step=True,
# on_epoch=False, prog_bar=False (diffusion.py:841-847), so it appears in NO
# console output at all -- only in the logger. Offline wandb gives us
# wandb-summary.json without any network I/O. WANDB_DIR keeps it beside the arm.
export WANDB_MODE=offline
export WANDB_SILENT=true

E1_ROOT="${E1_ROOT:-${PWD}/logs/e1_cost}"
mkdir -p "${E1_ROOT}" "${PWD}/watch_folder"

# ---------------------------------------------------------------- GPU index --
# Under this cluster's cgroup isolation the allocated device is renumbered to 0,
# but do not assume it: resolve from CUDA_VISIBLE_DEVICES and fall back.
e1_gpu_index() {
    local idx="${CUDA_VISIBLE_DEVICES:-0}"
    idx="${idx%%,*}"
    case "${idx}" in ''|*[!0-9]*) idx=0 ;; esac
    if ! nvidia-smi -i "${idx}" --query-gpu=memory.used \
         --format=csv,noheader,nounits >/dev/null 2>&1; then
        idx=0
    fi
    echo "${idx}"
}
E1_GPU="$(e1_gpu_index)"

# ------------------------------------------------------------------ run_arm --
# run_arm <arm_name> <hydra args...>
# Writes into ${E1_ROOT}/<arm_name>/:
#   run.log    stdout+stderr, \r expanded, every line prefixed with epoch seconds
#   mem.csv    whole-GPU memory.used (MiB), polled every 0.5 s
#   procmem.csv per-compute-process memory (MiB), contamination-free
#   meta.txt   node, gpu name, exit code, hydra command line
run_arm() {
    local arm="$1"; shift
    local dir="${E1_ROOT}/${arm}"
    mkdir -p "${dir}"

    {
        echo "arm=${arm}"
        echo "host=$(hostname)"
        echo "slurm_job=${SLURM_JOB_ID:-none}"
        echo "gpu_index=${E1_GPU}"
        echo "gpu_name=$(nvidia-smi -i "${E1_GPU}" --query-gpu=name --format=csv,noheader)"
        echo "alloc_conf=${PYTORCH_CUDA_ALLOC_CONF}"
        echo "started=$(date -Is)"
        echo "cmd=python -u main.py $*"
    } > "${dir}/meta.txt"

    echo "=== [$(date -Is)] ARM ${arm} on $(hostname) gpu${E1_GPU} ==="

    # Co-tenancy. Filtering on GPU model does NOT make nodes comparable: two
    # cards both reporting "RTX A6000" have been measured 2.4x apart on
    # identical work, because a co-tenant process on the same card (or the same
    # PCIe/NUMA domain) steals bandwidth. Slurm here hands out GPUs, not nodes,
    # so we cannot exclude that by request without --exclusive, which will not
    # schedule under current a6000 contention. Instead: record it. Any nonzero
    # baseline, or any compute-app PID that is not ours, means the timing on
    # this arm is contended and must be reported as such.
    nvidia-smi -i "${E1_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits \
        > "${dir}/mem_baseline.txt" 2>/dev/null
    nvidia-smi -i "${E1_GPU}" --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader 2>/dev/null > "${dir}/cotenants_start.csv"
    {
        echo "node=$(hostname -s)"
        echo "cotenants_at_start=$(wc -l < "${dir}/cotenants_start.csv" | tr -d ' ')"
        echo "slurm_exclusive=${SLURM_JOB_EXCLUSIVE:-no}"
    } >> "${dir}/meta.txt"

    nvidia-smi -i "${E1_GPU}" --query-gpu=memory.used \
        --format=csv,noheader,nounits -lms 500 > "${dir}/mem.csv" 2>/dev/null &
    local SMI=$!
    ( while kill -0 ${SMI} 2>/dev/null; do
        nvidia-smi -i "${E1_GPU}" --query-compute-apps=pid,used_gpu_memory \
            --format=csv,noheader,nounits 2>/dev/null
        sleep 1
      done ) > "${dir}/procmem.csv" 2>/dev/null &
    local PSMI=$!

    # Every stdout/stderr line is stamped with epoch seconds AFTER \r expansion,
    # so each tqdm refresh becomes its own timestamped line. That is where the
    # per-batch timings come from -- tqdm's own [MM:SS] field only has 1 s
    # resolution, which is too coarse for the <2% spread gate.
    # hydra.run.dir override, not WANDB_DIR: WandbLogger passes save_dir='.'
    # explicitly, which beats WANDB_DIR, and configs/config.yaml:91 chdirs into
    # the hydra run dir -- so pinning that dir is what puts wandb-summary.json
    # (the ONLY place val/num_decoding_steps survives) beside the arm.
    ( python -u main.py "$@" wandb.name="${arm}" \
        hydra.run.dir="${dir}/hydra" 2>&1 ) \
        | stdbuf -oL tr '\r' '\n' \
        | stdbuf -oL python3 -u -c 'import sys,time
for l in sys.stdin:
    sys.stdout.write("%.3f %s" % (time.time(), l))' \
        > "${dir}/run.log"
    local rc=${PIPESTATUS[0]}

    kill ${SMI} ${PSMI} 2>/dev/null || true
    wait ${SMI} ${PSMI} 2>/dev/null || true

    # Distinct PIDs seen on this card across the whole arm, minus our own: the
    # honest co-tenancy count, since a neighbour can start mid-arm.
    local ours cot
    ours=$(sort -u "${dir}/procmem.csv" 2>/dev/null | wc -l | tr -d ' ')
    cot=$(awk -F, '{gsub(/ /,"",$1); print $1}' "${dir}/procmem.csv" 2>/dev/null \
          | sort -u | grep -c . )
    {
        echo "finished=$(date -Is)"
        echo "exit_code=${rc}"
        echo "peak_mem_mib=$(sort -n "${dir}/mem.csv" 2>/dev/null | tail -1)"
        echo "baseline_mem_mib=$(cat "${dir}/mem_baseline.txt" 2>/dev/null)"
        echo "distinct_gpu_pids=${cot}"
        echo "cotenants=$(( cot > 1 ? cot - 1 : 0 ))"
    } >> "${dir}/meta.txt"
    : "${ours}"

    echo "=== [$(date -Is)] ARM ${arm} rc=${rc} peak=$(sort -n "${dir}/mem.csv" 2>/dev/null | tail -1) MiB ==="
    return 0
}

# --------------------------------------------------------------- checkpoints --
# MDLM has no native block size: its L' is purely an *evaluation*-time blocking
# of the decoding order, so one checkpoint serves every L' column.
e1_ckpt() {  # e1_ckpt <mdlm|bd3lm> <block>
    if [ "$1" = "mdlm" ]; then echo "kuleshov-group/mdlm-owt";
    else echo "kuleshov-group/bd3lm-owt-block_size$2"; fi
}

# block_size MUST always be passed explicitly: configs/config.yaml:16 defaults it
# to ${model.length} and nothing asserts it matches the checkpoint.
e1_common_args() {  # e1_common_args <mdlm|bd3lm> <block> <eval_bs>
    local algo="$1" block="$2" bs="$3"
    echo "loader.eval_batch_size=${bs}"
    echo "loader.num_workers=4"
    echo "model=small"
    echo "algo=${algo}"
    echo "algo.backbone=hf_dit"
    echo "data=openwebtext-split"
    echo "model.length=1024"
    echo "block_size=${block}"
    echo "eval.checkpoint_path=$(e1_ckpt "${algo}" "${block}")"
    echo "trainer.devices=1"
    echo "strategy=single"
    echo "trainer.num_sanity_val_steps=0"
    # seed=null is load-bearing, not cosmetic. main.py:145 passes valid_seed=
    # config.seed, and dataloader.py:644-656 SHUFFLES the validation set
    # whenever that is non-None. With seed=null the split is served in fixed
    # order, so every arm -- DUEL and ELBO, every k, every repeat -- scores the
    # SAME first `limit_val_batches` sequences and the xELBO column is exact.
    #
    # It does NOT randomise anything: this Lightning logs "No seed found, seed
    # set to 0" and is fully deterministic, which is why two ELBO repeats at
    # seed=null returned bit-identical NLL (3.287830114364624 twice). Vary the
    # MC draw with PL_GLOBAL_SEED instead -- seed_everything(None) reads it when
    # config.seed is None, so the t draws move while the data order does not.
    echo "seed=null"

    if [ "${algo}" = "mdlm" ]; then echo "algo.ignore_bos=false"; fi
}

# ------------------------------------------------------------- calibration --
# Node-speed probe. Slurm gives us a GPU, not a node, so arms in different jobs
# land on different (possibly contended) cards. Every job runs this IDENTICAL
# arm first; its s/batch is the per-node yardstick that makes cross-table
# numbers auditable, and a large drift between jobs means the tables were not
# measured on comparable hardware. ~1 min.
e1_calibrate() {
    # Job id in the name: several E1 jobs land on the same node at once (four
    # did on nikola-compute-18), and a hostname-only name makes them share one
    # arm directory and clobber each other's meta/procmem.
    local arm="calib__bd3lm16__elbo__bs32__sdpa__$(hostname -s)__j${SLURM_JOB_ID:-0}"
    if grep -q "^exit_code=0$" "${E1_ROOT}/${arm}/meta.txt" 2>/dev/null; then
        echo "SKIP ${arm} (already complete)"; return 0
    fi
    mapfile -t C < <(e1_common_args bd3lm 16 32)
    run_arm "${arm}" "${C[@]}" mode=elbo_ppl model.attn_backend=sdpa \
        trainer.limit_val_batches=8
}
