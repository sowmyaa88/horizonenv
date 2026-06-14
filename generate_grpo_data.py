# generate_grpo_data.py
#
# Builds a dataset of "decision points" for GRPO training.
#
# Each decision point is one (task_name, seed, day) triple. To reach that
# exact environment state deterministically, we also store the list of
# actions taken on every prior day ("replay_actions"). During GRPO training,
# the reward function can recreate the environment with the same seed,
# replay those actions, then apply the policy's freshly-generated action for
# `day` and read off env.step()'s reward.
#
# States are collected by running a heuristic agent with some randomised
# "noise" actions mixed in, so the dataset covers both good and mediocre
# situations (not just the heuristic's optimal trajectory).

from __future__ import annotations

import json
import os
import random
from typing import Dict, List

# inference.py instantiates an OpenAI/Groq client at import time, which
# raises if no API key is set. We don't need that client here — only
# build_prompt() / SYSTEM_PROMPT — so set a placeholder key before importing.
os.environ.setdefault("GROQ_API_KEY", "unused-for-data-generation")

from models import InventoryAction
from server.inventory_env import InventoryEnv
from server.constants import PRODUCTS, SHIPPING, TASKS
from inference import build_prompt, SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# POLICY USED TO GENERATE TRAJECTORIES (heuristic + noise)
# ─────────────────────────────────────────────────────────────────────────────

def heuristic_action(env: InventoryEnv, obs_dict: Dict, rng: random.Random) -> InventoryAction:
    """Same idea as server/grader.py's heuristic baseline: keep ~2 days of
    stock, use medium shipping, set prices based on elasticity."""
    buy, methods, prices = {}, {}, {}

    for product, props in PRODUCTS.items():
        current_stock = sum(b[0] for b in env.state.inventory.get(product, []))
        incoming = sum(
            d.quantity for d in env.state.pending_deliveries if d.product == product
        )
        target_stock = props["base_demand"] * 2
        needed = max(0, target_stock - current_stock - incoming)

        capacity = obs_dict["remaining_capacity"].get(product, 0)
        order_qty = min(needed, capacity)

        if order_qty > 0:
            cost = order_qty * (props["cost"] + SHIPPING["medium"]["cost_per_unit"])
            if cost <= env.state.cash * 0.7:
                buy[product] = order_qty
                methods[product] = "medium"

        elasticity = props["elasticity"]
        if elasticity < 0.6:
            prices[product] = 1.2
        elif elasticity < 1.0:
            prices[product] = 1.1
        else:
            prices[product] = 1.0

    return InventoryAction(
        buy_quantities=buy,
        delivery_methods=methods,
        price_multipliers=prices,
        notes_to_self=f"Day {obs_dict['current_day']}: heuristic restock, prices set by elasticity.",
        weekly_plan="Maintain ~2 days of stock per product; watch directives and grocery expiry.",
    )


def noisy_action(env: InventoryEnv, obs_dict: Dict, rng: random.Random) -> InventoryAction:
    """A deliberately imperfect action: random buys / prices / occasional
    idleness. Used to diversify the states the policy will see."""
    buy, methods, prices = {}, {}, {}

    for product, props in PRODUCTS.items():
        if rng.random() < 0.5:
            capacity = obs_dict["remaining_capacity"].get(product, 0)
            buy[product] = rng.randint(0, max(0, min(capacity, props["base_demand"] * 3)))
            methods[product] = rng.choice(["slow", "medium", "fast"])
        prices[product] = round(rng.uniform(0.6, 1.4), 2)

    return InventoryAction(
        buy_quantities={k: v for k, v in buy.items() if v > 0},
        delivery_methods=methods,
        price_multipliers=prices,
        notes_to_self=f"Day {obs_dict['current_day']}: exploring a random restock/pricing strategy.",
    )


def rollout_policy(env: InventoryEnv, obs_dict: Dict, rng: random.Random) -> InventoryAction:
    if rng.random() < 0.3:
        return noisy_action(env, obs_dict, rng)
    return heuristic_action(env, obs_dict, rng)


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTION
# ─────────────────────────────────────────────────────────────────────────────

def collect_records(task_name: str, seeds: List[int], sample_every: int = 3) -> List[Dict]:
    records = []

    for seed in seeds:
        env = InventoryEnv(task_name=task_name, seed=seed)
        obs = env.reset()
        rng = random.Random(seed * 1000 + 7)
        replay_actions: List[Dict] = []

        for day in range(1, env.state.total_days + 1):
            obs_dict = obs.model_dump()

            if day == 1 or (day - 1) % sample_every == 0:
                records.append({
                    "task_name": task_name,
                    "seed": seed,
                    "day": day,
                    "replay_actions": json.dumps(replay_actions),
                    "prompt_text": build_prompt(obs_dict),
                })

            action = rollout_policy(env, obs_dict, rng)
            obs, reward, done = env.step(action)
            replay_actions.append(action.model_dump())

            if done:
                break

    return records


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["easy", "medium", "hard"],
                         choices=list(TASKS.keys()))
    parser.add_argument("--seeds-per-task", type=int, default=20,
                         help="Number of distinct episode seeds per task")
    parser.add_argument("--sample-every", type=int, default=3,
                         help="Sample a decision point every N days")
    parser.add_argument("--out", type=str, default="grpo_data.jsonl")
    args = parser.parse_args()

    all_records = []
    for task_name in args.tasks:
        seeds = list(range(args.seeds_per_task))
        recs = collect_records(task_name, seeds, sample_every=args.sample_every)
        print(f"{task_name}: {len(recs)} decision points from {len(seeds)} seeds")
        all_records.extend(recs)

    with open(args.out, "w") as f:
        for r in all_records:
            r["system_prompt"] = SYSTEM_PROMPT
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(all_records)} total decision points to {args.out}")


if __name__ == "__main__":
    main()
