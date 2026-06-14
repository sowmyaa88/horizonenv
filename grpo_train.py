# grpo_train.py
#
# GRPO fine-tuning for the HorizonEnv / TempoRL inventory-management agent,
# using Unsloth (fast LoRA + 4-bit) and TRL's GRPOTrainer.
#
# Reward = the real environment reward from InventoryEnv.step(), computed by
# replaying each prompt's prior-day actions (see generate_grpo_data.py) and
# then applying the action the model just generated.
#
# Includes the same "GRPO training tricks" as the OpenEnv reference repo:
#   - DAPO asymmetric clipping (epsilon / epsilon_high)
#   - Dr. GRPO loss (loss_type="dr_grpo") -> removes response-length bias
#   - Truncation masking (mask_truncated_completions=True)
#
# Run on a free Colab/Kaggle T4 (16GB) with:
#   python generate_grpo_data.py --seeds-per-task 20 --sample-every 3
#   python grpo_train.py

import json
import re

from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

from models import InventoryAction
from server.inventory_env import InventoryEnv


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 3072          # prompt + completion
MAX_PROMPT_LENGTH = 2048
MAX_COMPLETION_LENGTH = 768

DATA_FILE = "grpo_data.jsonl"
OUTPUT_DIR = "qwen-quartermaster-grpo"

LORA_RANK = 16
NUM_GENERATIONS = 4            # group size G — keep small on a single T4
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 1


# ─────────────────────────────────────────────────────────────────────────────
# MODEL + LoRA (Unsloth)
# ─────────────────────────────────────────────────────────────────────────────

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,         # vLLM-backed generation for GRPO rollouts
    gpu_memory_utilization=0.6,  # leave room for training activations
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_RANK * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

dataset = load_dataset("json", data_files=DATA_FILE, split="train")


def to_chat_prompt(example):
    example["prompt"] = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user", "content": example["prompt_text"]},
    ]
    return example


dataset = dataset.map(to_chat_prompt)


# ─────────────────────────────────────────────────────────────────────────────
# ACTION PARSING
# ─────────────────────────────────────────────────────────────────────────────

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text: str):
    """Parse the model's raw completion into an InventoryAction.
    Returns (action, was_valid_json)."""
    raw = text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    # Fall back to grabbing the first {...} block if there's extra prose
    match = JSON_BLOCK_RE.search(raw)
    if match:
        raw = match.group(0)

    try:
        data = json.loads(raw)
        return InventoryAction(**data), True
    except Exception:
        return InventoryAction(), False


def get_completion_text(completion) -> str:
    """Completions arrive as a list of chat messages for conversational
    datasets; pull out the assistant's text content."""
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


# ─────────────────────────────────────────────────────────────────────────────
# REWARD FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

INVALID_JSON_PENALTY = -1.5


def env_reward(prompts, completions, task_name, seed, day, replay_actions, **kwargs):
    """Replay each example up to `day - 1` using the stored actions, then
    apply the freshly generated action and return InventoryEnv.step()'s
    reward (penalised if the completion wasn't valid JSON)."""
    rewards = []

    for completion, t_name, s, d, replay_json in zip(
        completions, task_name, seed, day, replay_actions
    ):
        text = get_completion_text(completion)
        action, valid_json = parse_action(text)

        env = InventoryEnv(task_name=t_name, seed=s)
        env.reset()

        for action_dict in json.loads(replay_json):
            env.step(InventoryAction(**action_dict))

        _, reward, _ = env.step(action)

        if not valid_json:
            reward += INVALID_JSON_PENALTY

        rewards.append(float(reward))

    return rewards


def format_reward(completions, **kwargs):
    """Small auxiliary reward: +0.1 if the completion parses as valid JSON,
    0.0 otherwise. Helps the model converge on the required output format
    quickly, on top of the (sparser) environment signal."""
    rewards = []
    for completion in completions:
        text = get_completion_text(completion)
        _, valid_json = parse_action(text)
        rewards.append(0.1 if valid_json else 0.0)
    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# GRPO CONFIG — Dr. GRPO + DAPO tricks
# ─────────────────────────────────────────────────────────────────────────────

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_generations=NUM_GENERATIONS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),

    # --- DAPO asymmetric clipping ---
    # Widen the upper clip bound so low-probability (exploratory) tokens
    # aren't suppressed as aggressively as in symmetric PPO-style clipping.
    epsilon=0.2,
    epsilon_high=0.28,

    # --- Dr. GRPO loss ---
    # Fixed-denominator normalisation removes the response-length bias that
    # gives long incorrect completions softer gradients than short ones.
    loss_type="dr_grpo",

    # --- Truncation masking ---
    # Completions that run past max_completion_length are masked out of the
    # loss entirely rather than penalised, so "ran out of tokens" isn't
    # confused with "bad reasoning".
    mask_truncated_completions=True,

    logging_steps=1,
    save_steps=50,
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[env_reward, format_reward],
    args=training_args,
    train_dataset=dataset,
)


if __name__ == "__main__":
    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/final_lora")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_lora")
    print(f"Saved LoRA adapter to {OUTPUT_DIR}/final_lora")
