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
LABEL_PREFIX="${LABEL_PREFIX:-adhoc_teamwork}"
SMOKE="${SMOKE:-1}"

NUM_SEEDS="${NUM_SEEDS:-1}"
PARTNER_POP_SIZE="${PARTNER_POP_SIZE:-4}"
COMEDI_NUM_ENVS="${COMEDI_NUM_ENVS:-8}"
BR_NUM_ENVS="${BR_NUM_ENVS:-8}"
BR_EGO_ACTOR_TYPE="${BR_EGO_ACTOR_TYPE:-mlp}"
BR_NUM_EGO_TRAIN_SEEDS="${BR_NUM_EGO_TRAIN_SEEDS:-1}"
RENDER_EPISODES="${RENDER_EPISODES:-2}"
WANDB_LOGGER_MODE="${WANDB_LOGGER_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-aht-benchmark}"
WANDB_ENTITY="${WANDB_ENTITY:-aht-project}"

if [[ "${SMOKE}" == "1" ]]; then
  COMEDI_TOTAL_TIMESTEPS_PER_ITERATION="${COMEDI_TOTAL_TIMESTEPS_PER_ITERATION:-2.56e5}"
  BR_TOTAL_TIMESTEPS="${BR_TOTAL_TIMESTEPS:-2.56e5}"
  NUM_HELDOUT_EVAL_EPISODES="${NUM_HELDOUT_EVAL_EPISODES:-16}"
else
  COMEDI_TOTAL_TIMESTEPS_PER_ITERATION="${COMEDI_TOTAL_TIMESTEPS_PER_ITERATION:-6e6}"
  BR_TOTAL_TIMESTEPS="${BR_TOTAL_TIMESTEPS:-6e7}"
  NUM_HELDOUT_EVAL_EPISODES="${NUM_HELDOUT_EVAL_EPISODES:-64}"
fi

RUN_ROOT="results/manual_runs/${RUN_TS}"
LOG_DIR="${RUN_ROOT}/logs"
COMEDI_RUN="${RUN_ROOT}/comedi_${LAYOUT}"
BR_RUN="${RUN_ROOT}/ppo_br_${LAYOUT}"
mkdir -p "${LOG_DIR}"

export WANDB_MODE="${WANDB_LOGGER_MODE}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "============================================================"
echo "Ad Hoc Teamwork Overcooked v1 workflow"
echo "Root dir: ${ROOT_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Task: ${TASK}"
echo "Layout: ${LAYOUT}"
echo "Run root: ${RUN_ROOT}"
echo "SMOKE: ${SMOKE}"
echo "W&B mode: ${WANDB_LOGGER_MODE}"
echo "W&B project/entity: ${WANDB_PROJECT}/${WANDB_ENTITY}"
echo "============================================================"

if [[ "${WANDB_LOGGER_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "Warning: WANDB_LOGGER_MODE=online but WANDB_API_KEY is not set."
  echo "Run 'wandb login' or set WANDB_API_KEY before launching."
fi

echo
echo "[1/7] Environment and dependency sanity check"
"${PYTHON_BIN}" --version 2>&1 | tee "${LOG_DIR}/00_python_version.log"
"${PYTHON_BIN}" -c "import jax; print('jax', jax.__version__); print(jax.devices())" 2>&1 | tee "${LOG_DIR}/01_jax_devices.log"
"${PYTHON_BIN}" -c "import hydra, wandb, flax, moviepy, jaxmarl, jumanji; print('deps_ok')" 2>&1 | tee "${LOG_DIR}/02_deps.log"

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
echo "[5/7] CoMeDi teammate population generation"
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
  "algorithm.PARTNER_POP_SIZE=${PARTNER_POP_SIZE}"
  "algorithm.NUM_ENVS=${COMEDI_NUM_ENVS}"
  "algorithm.TOTAL_TIMESTEPS_PER_ITERATION=${COMEDI_TOTAL_TIMESTEPS_PER_ITERATION}"
  "hydra.run.dir=${COMEDI_RUN}"
)
"${comedi_cmd[@]}" 2>&1 | tee "${LOG_DIR}/06_comedi.log"

if [[ ! -d "${COMEDI_RUN}/saved_train_run" ]]; then
  echo "CoMeDi output not found at ${COMEDI_RUN}/saved_train_run"
  exit 1
fi

echo
echo "[6/7] PPO best-response ego training against one CoMeDi teammate"
if [[ "${BR_EGO_ACTOR_TYPE}" != "mlp" ]]; then
  echo "This script currently renders only mlp ego checkpoints. Set BR_EGO_ACTOR_TYPE=mlp."
  exit 1
fi

PARTNER_CFG="{comedi_seed0_member0:{path:${COMEDI_RUN}/saved_train_run,actor_type:actor_with_conditional_critic,ckpt_key:final_params_conf,idx_list:[[0,0]],POP_SIZE:${PARTNER_POP_SIZE},test_mode:false}}"

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
)
"${br_cmd[@]}" 2>&1 | tee "${LOG_DIR}/07_ppo_br.log"

if [[ ! -d "${BR_RUN}/ego_train_run" ]]; then
  echo "PPO-BR output not found at ${BR_RUN}/ego_train_run"
  exit 1
fi

echo
echo "[7/7] Heldout evaluation + MP4 rendering"
export BR_RUN TASK NUM_HELDOUT_EVAL_EPISODES RENDER_EPISODES LAYOUT

"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${LOG_DIR}/08_heldout_eval.log"
import os
from hydra import initialize, compose
from omegaconf import OmegaConf
from evaluation.heldout_evaluator import run_heldout_evaluation

task = os.environ["TASK"]
br_run = os.environ["BR_RUN"]
num_eval_eps = int(os.environ["NUM_HELDOUT_EVAL_EPISODES"])

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
            f"global_heldout_settings.NUM_EVAL_EPISODES={num_eval_eps}",
        ],
    )

print(OmegaConf.to_yaml(cfg, resolve=True))
cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
run_heldout_evaluation(cfg_dict, print_metrics=True)
PY

"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${LOG_DIR}/09_render_br_video.log"
import os
import jax
from envs import make_env
from agents.initialize_agents import initialize_mlp_agent
from common.save_load_utils import load_checkpoints
from evaluation.vis_episodes import save_video

br_run = os.environ["BR_RUN"]
layout = os.environ["LAYOUT"]
render_episodes = int(os.environ["RENDER_EPISODES"])

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
    save_name="ppo_br_ego_vs_self",
)
print(f"Saved render video to: {save_path}")
PY

echo
echo "Workflow complete."
echo "CoMeDi run: ${COMEDI_RUN}"
echo "PPO-BR run: ${BR_RUN}"
echo "Logs: ${LOG_DIR}"
echo "Videos:"
echo "  - tests/test_overcooked_agents.py output under results/overcooked-v1/videos/"
echo "  - PPO-BR render at ${BR_RUN}/videos/ppo_br_ego_vs_self.mp4"
