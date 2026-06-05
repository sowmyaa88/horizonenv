# inference.py

import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from server.inventory_env import InventoryEnv
from models import InventoryAction

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CLIENT SETUP
# ─────────────────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.3-70b-versatile"


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert retail store manager playing a 90-day inventory management game.

Each day you receive the current store state and must decide:
- What to buy and how much
- Which shipping speed to use (slow=$2/unit 3-7days, medium=$5/unit 2-4days, fast=$10/unit 1day)
- What to liquidate (dispose of expiring/excess stock)
- How to price products (0.5x to 1.5x multiplier affects demand)
- Notes to remember important information across days

PRODUCTS:
- electronics: sell $150, cost $100, margin $50, no expiry, elasticity 1.2
- clothing: sell $40, cost $25, margin $15, no expiry, elasticity 1.5
- groceries: sell $10, cost $5, margin $5, EXPIRES IN 5 DAYS, elasticity 0.4
- furniture: sell $200, cost $130, margin $70, no expiry, elasticity 0.8
- toys: sell $25, cost $12, margin $13, no expiry, elasticity 1.3

CRITICAL RULES:
1. Directives are shown ONCE — write them in notes_to_self immediately
2. Groceries expire in 5 days — never overstock them
3. Fast shipping costs 5x more but guarantees next-day delivery
4. Conflicting directives exist — calculate which penalty is lower to violate
5. Your notes_to_self persists to the next day — use it as your memory

You must respond with ONLY a valid JSON object, no explanation, no markdown:
{
  "buy_quantities": {"product": amount},
  "delivery_methods": {"product": "slow|medium|fast"},
  "liquidate": {"product": amount},
  "price_multipliers": {"product": 0.5-1.5},
  "notes_to_self": "your memory notes tracking directives, plans, warnings",
  "weekly_plan": "your strategic plan for the week",
  "take_loan": false
}"""


def build_prompt(obs: dict) -> str:
    """Format observation into a clear prompt for the LLM."""
    
    # Format inventory
    inventory_lines = []
    for product, batches in obs["updated_inventory"].items():
        total = sum(b[0] for b in batches)
        if batches and batches[0][1] is not None:
            expiry_info = f" (batches expiring: {[b[1] for b in batches]} days)"
        else:
            expiry_info = ""
        inventory_lines.append(f"  {product}: {total} units{expiry_info}")

    # Format events
    event_lines = []
    for event_id, countdown in obs["updated_events"].items():
        if countdown > 0:
            event_lines.append(f"  {event_id}: in {countdown} days")
        elif countdown <= 0:
            event_lines.append(f"  {event_id}: ACTIVE NOW (day {countdown})")

    # Format directives
    directive_lines = []
    for d in obs.get("new_directives", []):
        directive_lines.append(f"  NEW: {d['text']}")
    for did in obs.get("active_directive_ids", []):
        if not any(d["id"] == did for d in obs.get("new_directives", [])):
            directive_lines.append(f"  ACTIVE: {did} (check your notes for details)")

    # Format violations
    violation_lines = []
    for v in obs.get("directive_violations_last_step", []):
        violation_lines.append(f"  VIOLATED {v['id']}: {v['message']} (penalty -{v['penalty']})")

    # Format milestones
    milestone_lines = []
    for mid, m in obs.get("milestones", {}).items():
        status = "ACHIEVED" if m.get("achieved") else f"deadline day {m['deadline']}"
        milestone_lines.append(f"  {mid}: target={m['target']} [{status}]")

    # Format deliveries
    delivery_lines = []
    for d in obs.get("updated_deliveries", []):
        delivery_lines.append(f"  {d['product']}: {d['quantity']} units arriving day {d['arrives_day']}")

    prompt = f"""=== DAY {obs['current_day']} / {obs['total_days']} ===

FINANCIALS:
  Cash: ${obs['total_cash']}
  Today's profit: ${obs['day_profit']}
  Total profit: ${obs['total_profit']}
  Loan balance: ${obs['loan_balance']}

INVENTORY:
{chr(10).join(inventory_lines)}

WAREHOUSE CAPACITY REMAINING:
{chr(10).join(f"  {p}: {c} units" for p, c in obs['remaining_capacity'].items())}

YESTERDAY'S DEMAND:
{chr(10).join(f"  {p}: {d} units" for p, d in obs['demand_today'].items())}

INCOMING DELIVERIES:
{chr(10).join(delivery_lines) if delivery_lines else "  None"}

UPCOMING EVENTS:
{chr(10).join(event_lines) if event_lines else "  None"}

DIRECTIVES:
{chr(10).join(directive_lines) if directive_lines else "  None active"}

VIOLATIONS LAST STEP:
{chr(10).join(violation_lines) if violation_lines else "  None"}

MILESTONES:
{chr(10).join(milestone_lines)}

YOUR NOTES FROM YESTERDAY:
{obs.get('agent_notes', 'None') or 'None'}

YOUR WEEKLY PLAN:
{obs.get('agent_weekly_plan', 'None') or 'None'}

Respond with ONLY a JSON action object."""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_action(obs: dict) -> InventoryAction:
    """Ask the LLM what to do, parse its response into an InventoryAction."""
    prompt = build_prompt(obs)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        return InventoryAction(**data)
    except Exception as e:
        print(f"  [parse error] {e} — using empty action")
        return InventoryAction()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(task_name: str = "easy") -> float:
    """Run one full 90-day episode with the LLM as the agent."""
    from server.grader import score_episode

    print(f"\n{'='*60}")
    print(f"  HorizonEnv — Task: {task_name.upper()}")
    print(f"{'='*60}")

    env = InventoryEnv(task_name=task_name, seed=42)
    obs = env.reset()
    obs_dict = obs.model_dump()

    total_reward = 0.0
    done = False
    day = 1

    while not done:
        print(f"\n--- Day {day} | Cash: ${obs_dict['total_cash']} | "
              f"Profit: ${obs_dict['total_profit']} ---")

        # Get LLM action
        action = get_llm_action(obs_dict)

        print(f"  Buying: {action.buy_quantities or 'nothing'}")
        print(f"  Prices: {action.price_multipliers or 'default'}")
        if action.notes_to_self:
            print(f"  Notes: {action.notes_to_self[:80]}...")

        # Step environment
        obs, reward, done = env.step(action)
        obs_dict = obs.model_dump()
        total_reward += reward

        if obs_dict.get("directive_violations_last_step"):
            for v in obs_dict["directive_violations_last_step"]:
                print(f"  ⚠ VIOLATION: {v['message']}")

        day += 1

    # Final score
    final_profit = env.state.total_profit
    score = score_episode(task_name, final_profit)

    print(f"\n{'='*60}")
    print(f"  EPISODE COMPLETE")
    print(f"  Final profit:  ${final_profit:.2f}")
    print(f"  Total reward:  {total_reward:.4f}")
    print(f"  Score:         {score:.4f} / 1.0")
    print(f"{'='*60}\n")

    return score


if __name__ == "__main__":
    # Run all 3 tasks
    scores = {}
    for task in ["easy", "medium", "hard"]:
        scores[task] = run_episode(task)

    print("\nFINAL SCORES:")
    for task, score in scores.items():
        print(f"  {task}: {score:.4f}")