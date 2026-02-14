#!/usr/bin/env bash
set -euo pipefail

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No python interpreter found in PATH."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RUN_TS="${RUN_TS:-$(date +%Y-%m-%d_%H-%M-%S)}"
TASK="${TASK:-overcooked-v1/cramped_room}"
LAYOUT="${LAYOUT:-cramped_room}"
LABEL_PREFIX="${LABEL_PREFIX:-adhoc_teamwork_po}"

NUM_SEEDS="${NUM_SEEDS:-1}"
COMEDI_NUM_ENVS="${COMEDI_NUM_ENVS:-8}"
BR_NUM_ENVS="${BR_NUM_ENVS:-8}"
BR_EGO_ACTOR_TYPE="${BR_EGO_ACTOR_TYPE:-mlp}"
BR_NUM_EGO_TRAIN_SEEDS="${BR_NUM_EGO_TRAIN_SEEDS:-1}"
BR_PARTNER_IDX_LIST="${BR_PARTNER_IDX_LIST:-[[0,0]]}"
RENDER_EPISODES="${RENDER_EPISODES:-2}"
WANDB_LOGGER_MODE="${WANDB_LOGGER_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-aht-benchmark}"
WANDB_ENTITY="${WANDB_ENTITY:-aht-project}"

# Partial observability knobs
PO_MODE="${PO_MODE:-cone}"                    # none | cone
PO_FOV_RANGE="${PO_FOV_RANGE:-5}"
PO_FOV_SLOPE="${PO_FOV_SLOPE:-0.7}"
PO_USE_OCCLUSION="${PO_USE_OCCLUSION:-true}" # true | false
PO_SOFT_VIEW="${PO_SOFT_VIEW:-false}"        # true | false
PO_DIST_SIGMA="${PO_DIST_SIGMA:-3.0}"
PO_ANG_SIGMA="${PO_ANG_SIGMA:-1.5}"

COMEDI_TOTAL_TIMESTEPS_PER_ITERATION="${COMEDI_TOTAL_TIMESTEPS_PER_ITERATION:-6e6}"
BR_TOTAL_TIMESTEPS="${BR_TOTAL_TIMESTEPS:-6e7}"
NUM_HELDOUT_EVAL_EPISODES="${NUM_HELDOUT_EVAL_EPISODES:-64}"

RUN_ROOT="results/manual_runs/${RUN_TS}"
LOG_DIR="${RUN_ROOT}/logs"
COMEDI_RUN="${RUN_ROOT}/comedi_${LAYOUT}_po"
BR_RUN="${RUN_ROOT}/ppo_br_${LAYOUT}_po"
mkdir -p "${LOG_DIR}"

export WANDB_MODE="${WANDB_LOGGER_MODE}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export LAYOUT PO_MODE PO_FOV_RANGE PO_FOV_SLOPE PO_USE_OCCLUSION PO_SOFT_VIEW PO_DIST_SIGMA PO_ANG_SIGMA

PO_OVERRIDES=(
  "ENV_KWARGS.po_mode=${PO_MODE}"
  "ENV_KWARGS.fov_range=${PO_FOV_RANGE}"
  "ENV_KWARGS.fov_slope=${PO_FOV_SLOPE}"
  "ENV_KWARGS.use_occlusion=${PO_USE_OCCLUSION}"
  "ENV_KWARGS.soft_view=${PO_SOFT_VIEW}"
  "ENV_KWARGS.dist_sigma=${PO_DIST_SIGMA}"
  "ENV_KWARGS.ang_sigma=${PO_ANG_SIGMA}"
)

echo "============================================================"
echo "Ad Hoc Teamwork Overcooked v1 PO workflow"
echo "Root dir: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Task: ${TASK}"
echo "Layout: ${LAYOUT}"
echo "Run root: ${RUN_ROOT}"
echo "W&B mode: ${WANDB_LOGGER_MODE}"
echo "W&B project/entity: ${WANDB_PROJECT}/${WANDB_ENTITY}"
echo "PO config:"
echo "  - mode: ${PO_MODE}"
echo "  - range: ${PO_FOV_RANGE}"
echo "  - slope: ${PO_FOV_SLOPE}"
echo "  - occlusion: ${PO_USE_OCCLUSION}"
echo "  - soft_view: ${PO_SOFT_VIEW}"
echo "  - dist_sigma: ${PO_DIST_SIGMA}"
echo "  - ang_sigma: ${PO_ANG_SIGMA}"
echo "============================================================"

if [[ "${WANDB_LOGGER_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "Warning: WANDB_LOGGER_MODE=online but WANDB_API_KEY is not set."
  echo "Run 'wandb login' or set WANDB_API_KEY before launching."
fi

if [[ "${WANDB_LOGGER_MODE}" == "offline" ]]; then
  echo "Note: WANDB_LOGGER_MODE=offline. Set WANDB_LOGGER_MODE=online for rented GPU machines to sync runs automatically."
fi

echo
echo "[1/7] Environment and dependency sanity check"
"${PYTHON_BIN}" --version 2>&1 | tee "${LOG_DIR}/00_python_version.log"
"${PYTHON_BIN}" -c "import jax; print('jax', jax.__version__); devs = jax.devices(); print(devs); gpu_devs = [d for d in devs if d.platform == 'gpu']; print(f'{len(gpu_devs)} GPU(s) detected') if gpu_devs else print('WARNING: No GPU detected — training will be very slow on CPU.')" 2>&1 | tee "${LOG_DIR}/01_jax_devices.log"
"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${LOG_DIR}/02_deps.log"
import importlib
import sys

mods = ["hydra", "wandb", "flax", "moviepy", "jaxmarl", "jumanji"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)

if missing:
    print("Missing Python modules:", ", ".join(missing))
    sys.exit(1)

print("deps_ok")
PY

echo
echo "[2/7] Download heldout evaluation data (if missing)"
if [[ ! -d "eval_teammates" ]]; then
  "${PYTHON_BIN}" download_eval_data.py 2>&1 | tee "${LOG_DIR}/03_download_eval_data.log"
else
  echo "eval_teammates/ already exists, skipping download." | tee "${LOG_DIR}/03_download_eval_data.log"
fi

echo
echo "[3/7] Overcooked heuristic render + debug print test"
"${PYTHON_BIN}" tests/test_overcooked_agents.py 2>&1 | tee "${LOG_DIR}/04_test_overcooked_agents.log"

echo
echo "[4/7] Overcooked wrapper print test"
"${PYTHON_BIN}" tests/test_overcooked_wrapper.py 2>&1 | tee "${LOG_DIR}/05_test_overcooked_wrapper.log"

echo
echo "[4b/7] Partial-observability wrapper smoke test"
"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${LOG_DIR}/05b_test_overcooked_po_wrapper.log"
import os
import jax
from envs import make_env

to_bool = lambda x: str(x).strip().lower() == "true"

env = make_env(
    env_name="overcooked-v1",
    env_kwargs={
        "layout": os.environ["LAYOUT"],
        "random_obj_state": True,
        "do_reward_shaping": True,
        "po_mode": os.environ["PO_MODE"],
        "fov_range": int(os.environ["PO_FOV_RANGE"]),
        "fov_slope": float(os.environ["PO_FOV_SLOPE"]),
        "use_occlusion": to_bool(os.environ["PO_USE_OCCLUSION"]),
        "soft_view": to_bool(os.environ["PO_SOFT_VIEW"]),
        "dist_sigma": float(os.environ["PO_DIST_SIGMA"]),
        "ang_sigma": float(os.environ["PO_ANG_SIGMA"]),
    },
)

key = jax.random.PRNGKey(0)
key, sk = jax.random.split(key)
obs, state = env.reset(sk)
actions = {a: 4 for a in env.agents}
key, sk = jax.random.split(key)
obs, state, rew, dones, info = env.step(sk, state, actions)
print("po_wrapper_smoke_ok", {a: obs[a].shape for a in env.agents}, rew, dones["__all__"])
PY

echo
echo "[5/7] CoMeDi teammate population generation (PO)"
if [[ -d "${COMEDI_RUN}/saved_train_run" ]]; then
  echo "Skipping: ${COMEDI_RUN}/saved_train_run already exists (previous run completed)."
else
  comedi_cmd=(
    "${PYTHON_BIN}" teammate_generation/run.py
    "task=${TASK}"
    "algorithm=comedi/${TASK}"
    "label=${LABEL_PREFIX}_comedi"
    "train_ego=false"
    "run_heldout_eval=false"
    "logger.mode=${WANDB_LOGGER_MODE}"
    "logger.project=${WANDB_PROJECT}"
    "logger.entity=${WANDB_ENTITY}"
    "logger.verbose=true"
    "algorithm.NUM_SEEDS=${NUM_SEEDS}"
    ${PARTNER_POP_SIZE:+"algorithm.PARTNER_POP_SIZE=${PARTNER_POP_SIZE}"}
    "algorithm.NUM_ENVS=${COMEDI_NUM_ENVS}"
    "algorithm.TOTAL_TIMESTEPS_PER_ITERATION=${COMEDI_TOTAL_TIMESTEPS_PER_ITERATION}"
    "hydra.run.dir=${COMEDI_RUN}"
    "${PO_OVERRIDES[@]}"
  )
  "${comedi_cmd[@]}" 2>&1 | tee "${LOG_DIR}/06_comedi_po.log"

  if [[ ! -d "${COMEDI_RUN}/saved_train_run" ]]; then
    echo "CoMeDi output not found at ${COMEDI_RUN}/saved_train_run"
    exit 1
  fi
fi

echo
echo "[6/7] PPO best-response ego training against one CoMeDi teammate (PO)"
if [[ -d "${BR_RUN}/ego_train_run" ]]; then
  echo "Skipping: ${BR_RUN}/ego_train_run already exists (previous run completed)."
else
  if [[ "${BR_EGO_ACTOR_TYPE}" != "mlp" ]]; then
    echo "This script currently renders only mlp ego checkpoints. Set BR_EGO_ACTOR_TYPE=mlp."
    exit 1
  fi

  # PPO-BR only supports 1 partner (asserted in ppo_br.py:125).
  # POP_SIZE falls back to the Hydra base default (4) when PARTNER_POP_SIZE is unset.
  BR_POP_SIZE="${PARTNER_POP_SIZE:-4}"
  PARTNER_CFG="{comedi_seed0_member0:{path:${COMEDI_RUN}/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,idx_list:${BR_PARTNER_IDX_LIST},POP_SIZE:${BR_POP_SIZE},test_mode:false}}"

  br_cmd=(
    "${PYTHON_BIN}" ego_agent_training/run.py
    "task=${TASK}"
    "algorithm=ppo_br/${TASK}"
    "label=${LABEL_PREFIX}_ppo_br"
    "run_heldout_eval=false"
    "logger.mode=${WANDB_LOGGER_MODE}"
    "logger.project=${WANDB_PROJECT}"
    "logger.entity=${WANDB_ENTITY}"
    "logger.verbose=true"
    "algorithm.EGO_ACTOR_TYPE=${BR_EGO_ACTOR_TYPE}"
    "algorithm.NUM_EGO_TRAIN_SEEDS=${BR_NUM_EGO_TRAIN_SEEDS}"
    "algorithm.NUM_ENVS=${BR_NUM_ENVS}"
    "algorithm.TOTAL_TIMESTEPS=${BR_TOTAL_TIMESTEPS}"
    "algorithm.partner_agent=${PARTNER_CFG}"
    "hydra.run.dir=${BR_RUN}"
    "${PO_OVERRIDES[@]}"
  )
  "${br_cmd[@]}" 2>&1 | tee "${LOG_DIR}/07_ppo_br_po.log"

  if [[ ! -d "${BR_RUN}/ego_train_run" ]]; then
    echo "PPO-BR output not found at ${BR_RUN}/ego_train_run"
    exit 1
  fi
fi

echo
echo "[7/7] Heldout evaluation + MP4 rendering (PO)"
export BR_RUN TASK NUM_HELDOUT_EVAL_EPISODES RENDER_EPISODES LAYOUT
export PO_MODE PO_FOV_RANGE PO_FOV_SLOPE PO_USE_OCCLUSION PO_SOFT_VIEW PO_DIST_SIGMA PO_ANG_SIGMA

"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${LOG_DIR}/08_heldout_eval_po.log"
import os
from hydra import initialize, compose
from omegaconf import OmegaConf
from evaluation.heldout_evaluator import run_heldout_evaluation

task = os.environ["TASK"]
br_run = os.environ["BR_RUN"]
num_eval_eps = int(os.environ["NUM_HELDOUT_EVAL_EPISODES"])

po_mode = os.environ["PO_MODE"]
po_overrides = [
    f"ENV_KWARGS.po_mode={po_mode}",
    f"ENV_KWARGS.fov_range={os.environ['PO_FOV_RANGE']}",
    f"ENV_KWARGS.fov_slope={os.environ['PO_FOV_SLOPE']}",
    f"ENV_KWARGS.use_occlusion={os.environ['PO_USE_OCCLUSION']}",
    f"ENV_KWARGS.soft_view={os.environ['PO_SOFT_VIEW']}",
    f"ENV_KWARGS.dist_sigma={os.environ['PO_DIST_SIGMA']}",
    f"ENV_KWARGS.ang_sigma={os.environ['PO_ANG_SIGMA']}",
]

# Disable return normalization under PO — performance bounds were calibrated
# under full observability and are meaningless under partial observability.
heldout_overrides = [
    f"global_heldout_settings.NUM_EVAL_EPISODES={num_eval_eps}",
]
if po_mode != "none":
    heldout_overrides.append("global_heldout_settings.NORMALIZE_RETURNS=false")

with initialize(version_base=None, config_path="evaluation/configs"):
    cfg = compose(
        config_name="heldout_ego",
        overrides=[
            f"task={task}",
            f"ego_agent.path={br_run}/ego_train_run",
            "ego_agent.actor_type=mlp",
            "ego_agent.ckpt_key=final_params",
            "ego_agent.idx_list=[0]",
            "ego_agent.test_mode=true",
            *heldout_overrides,
            *po_overrides,
        ],
    )

print(OmegaConf.to_yaml(cfg, resolve=True))
cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
run_heldout_evaluation(cfg_dict, print_metrics=True)
PY

"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${LOG_DIR}/09_render_br_video_po.log"
import os
import jax
from envs import make_env
from agents.initialize_agents import initialize_mlp_agent
from common.save_load_utils import load_checkpoints
from evaluation.vis_episodes import save_video

br_run = os.environ["BR_RUN"]
layout = os.environ["LAYOUT"]
render_episodes = int(os.environ["RENDER_EPISODES"])

to_bool = lambda x: str(x).strip().lower() == "true"

env = make_env(
    "overcooked-v1",
    {
        "layout": layout,
        "random_obj_state": True,
        "do_reward_shaping": True,
        "reward_shaping_params": {
            "PLACEMENT_IN_POT_REW": 0.5,
            "PLATE_PICKUP_REWARD": 0.1,
            "SOUP_PICKUP_REWARD": 1.0,
            "ONION_PICKUP_REWARD": 0.1,
            "COUNTER_PICKUP_REWARD": 0.0,
            "COUNTER_DROP_REWARD": 0.0,
        },
        "po_mode": os.environ["PO_MODE"],
        "fov_range": int(os.environ["PO_FOV_RANGE"]),
        "fov_slope": float(os.environ["PO_FOV_SLOPE"]),
        "use_occlusion": to_bool(os.environ["PO_USE_OCCLUSION"]),
        "soft_view": to_bool(os.environ["PO_SOFT_VIEW"]),
        "dist_sigma": float(os.environ["PO_DIST_SIGMA"]),
        "ang_sigma": float(os.environ["PO_ANG_SIGMA"]),
    },
)

ego_ckpts = load_checkpoints(f"{br_run}/ego_train_run", ckpt_key="checkpoints")
ego_params = jax.tree.map(lambda x: x[0, -1], ego_ckpts)

rng = jax.random.PRNGKey(0)
rng, init_rng0, init_rng1 = jax.random.split(rng, 3)
ego_policy_0, _ = initialize_mlp_agent({}, env, init_rng0)
ego_policy_1, _ = initialize_mlp_agent({}, env, init_rng1)

save_path = save_video(
    env=env,
    env_name="overcooked-v1",
    agent_0_param=ego_params,
    agent_0_policy=ego_policy_0,
    agent_1_param=ego_params,
    agent_1_policy=ego_policy_1,
    max_episode_steps=400,
    num_eps=render_episodes,
    savevideo=True,
    save_dir=f"{br_run}/videos",
    save_name="ppo_br_ego_vs_self_po",
)
print(f"Saved render video to: {save_path}")
PY

echo
echo "Workflow complete."
echo "CoMeDi run: ${COMEDI_RUN}"
echo "PPO-BR run: ${BR_RUN}"
echo "Logs: ${LOG_DIR}"
echo "PO video: ${BR_RUN}/videos/ppo_br_ego_vs_self_po.mp4"
