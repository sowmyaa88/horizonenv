# server/grader.py

from typing import Tuple


def compute_baselines(task_name: str) -> Tuple[float, float]:
    """
    Compute floor and ceiling profits for a task.
    Both are deterministic (seeded) so scores are reproducible.

    Floor  = passive agent (never buys, sells initial stock until depleted)
    Ceiling = heuristic agent (knows demand, buys optimally every day)
    """
    floor = _run_passive_agent(task_name)
    ceiling = _run_heuristic_agent(task_name)
    return floor, ceiling


def score_episode(task_name: str, agent_profit: float) -> float:
    """
    Score an agent's profit between 0.0 and 1.0.
    0.002 = worst possible, 0.998 = best possible.
    """
    floor, ceiling = compute_baselines(task_name)
    if ceiling == floor:
        return 0.5
    raw = (agent_profit - floor) / (ceiling - floor)
    return round(max(0.002, min(0.998, raw)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# PASSIVE AGENT (floor baseline)
# ─────────────────────────────────────────────────────────────────────────────

def _run_passive_agent(task_name: str) -> float:
    """
    Never buys anything. Just sells whatever starting stock it has
    until it runs out. No pricing strategy, no planning.
    """
    from server.inventory_env import InventoryEnv
    from models import InventoryAction

    env = InventoryEnv(task_name=task_name, seed=42)
    env.reset()

    done = False
    while not done:
        action = InventoryAction()   # empty action — do nothing
        _, _, done = env.step(action)

    return env.state.total_profit


# ─────────────────────────────────────────────────────────────────────────────
# HEURISTIC AGENT (ceiling baseline)
# ─────────────────────────────────────────────────────────────────────────────

def _run_heuristic_agent(task_name: str) -> float:
    """
    A rule-based agent with perfect demand knowledge.
    Buys optimally every day using medium shipping.
    Sets prices to maximize revenue given elasticity.
    """
    from server.inventory_env import InventoryEnv
    from server.constants import PRODUCTS, SHIPPING
    from models import InventoryAction

    env = InventoryEnv(task_name=task_name, seed=42)
    obs = env.reset()

    done = False
    while not done:
        buy = {}
        methods = {}
        prices = {}

        for product, props in PRODUCTS.items():
            # Check how much stock we have
            current_stock = sum(
                b[0] for b in env.state.inventory.get(product, [])
            )

            # Check incoming deliveries
            incoming = sum(
                d.quantity for d in env.state.pending_deliveries
                if d.product == product
            )

            # Target: keep 2 days of demand in stock
            target_stock = props["base_demand"] * 2
            needed = max(0, target_stock - current_stock - incoming)

            # Check warehouse capacity
            capacity = obs.remaining_capacity.get(product, 0)
            order_qty = min(needed, capacity)

            if order_qty > 0:
                # Only order if we can afford it
                cost = order_qty * (props["cost"] + SHIPPING["medium"]["cost_per_unit"])
                if cost <= env.state.cash * 0.7:   # don't spend more than 70% of cash
                    buy[product] = order_qty
                    methods[product] = "medium"

            # Optimal price: slightly above 1.0 for low-elasticity products
            elasticity = props["elasticity"]
            if elasticity < 0.6:
                prices[product] = 1.2    # groceries: inelastic, can charge more
            elif elasticity < 1.0:
                prices[product] = 1.1
            else:
                prices[product] = 1.0    # elastic products: keep at base price

        action = InventoryAction(
            buy_quantities=buy,
            delivery_methods=methods,
            price_multipliers=prices,
            notes_to_self="Heuristic baseline agent.",
        )

        obs, _, done = env.step(action)

    return env.state.total_profit