from __future__ import annotations
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# ─── Base classes (OpenEnv protocol) ────────────────────────────────────────

class Action(BaseModel):
    """Every env action inherits from this."""
    pass

class Observation(BaseModel):
    """Every env observation inherits from this."""
    pass

class State(BaseModel):
    """Internal env state (not sent to agent)."""
    pass


# ─── Agent's ACTION (what the LLM sends each day) ───────────────────────────

class InventoryAction(Action):
    buy_quantities: Dict[str, int] = Field(
        default_factory=dict,
        description="Product name → units to order. E.g. {'electronics': 10}"
    )
    delivery_methods: Dict[str, Literal["slow", "medium", "fast"]] = Field(
        default_factory=dict,
        description="Product name → shipping speed. Defaults to slow if omitted."
    )
    liquidate: Dict[str, int] = Field(
        default_factory=dict,
        description="Product name → units to dispose of (no revenue)."
    )
    price_multipliers: Dict[str, float] = Field(
        default_factory=dict,
        description="Product name → price multiplier (0.5 to 1.5). Default 1.0."
    )
    notes_to_self: str = Field(
        default="",
        description="Agent's scratchpad. Persists and is returned next step."
    )
    weekly_plan: Optional[str] = Field(
        default=None,
        description="Persistent plan shown every step until overwritten."
    )
    take_loan: bool = Field(
        default=False,
        description="Request a $500 loan. Only works when cash < $100 and loans < 2."
    )


# ─── What the agent OBSERVES each day ───────────────────────────────────────

class InventoryObservation(Observation):
    current_day: int
    total_days: int
    total_cash: float
    day_profit: float
    total_profit: float

    # Yesterday's actual sales demand per product
    demand_today: Dict[str, int] = Field(default_factory=dict)

    # Inventory as batches: {"groceries": [[qty, days_left], ...]}
    updated_inventory: Dict[str, List[List[Optional[int]]]] = Field(default_factory=dict)

    # How much more of each product the warehouse can hold
    remaining_capacity: Dict[str, int] = Field(default_factory=dict)

    # Seasonal events with countdowns (negative = currently active)
    updated_events: Dict[str, int] = Field(default_factory=dict)

    # Shipments in transit: [{"product": "electronics", "qty": 10, "arrives_day": 5}]
    updated_deliveries: List[Dict] = Field(default_factory=list)

    # Full directive text — shown ONLY on the day it arrives
    new_directives: List[Dict] = Field(default_factory=list)

    # After arrival day, only IDs are shown (agent must remember the rest)
    active_directive_ids: List[str] = Field(default_factory=list)

    # Which rules were broken last step and their penalties
    directive_violations_last_step: List[Dict] = Field(default_factory=list)

    # Time-bound targets: {"milestone_id": {"target": ..., "deadline": ..., "progress": ...}}
    milestones: Dict[str, Dict] = Field(default_factory=dict)

    # Returned from agent's previous notes_to_self
    agent_notes: str = ""
    agent_weekly_plan: str = ""

    # Loan tracking
    loan_balance: float = 0.0
    loans_taken: int = 0
    loans_remaining: int = 2


# ─── Internal STATE (env tracks this, never sent directly to agent) ─────────

class DeliveryBatch(BaseModel):
    product: str
    quantity: int
    arrives_day: int
    delivery_method: str

class InventoryState(State):
    current_day: int = 1
    total_days: int = 90
    cash: float = 2000.0
    total_profit: float = 0.0

    # {"electronics": [[qty, days_left], ...], "groceries": [...], ...}
    inventory: Dict[str, List[List[Optional[int]]]] = Field(default_factory=dict)

    # In-transit orders
    pending_deliveries: List[DeliveryBatch] = Field(default_factory=list)

    # Agent memory (persisted across steps)
    agent_notes: str = ""
    agent_weekly_plan: str = ""

    # Active directive IDs
    active_directive_ids: List[str] = Field(default_factory=list)

    # Directive violation history
    directive_violations: List[Dict] = Field(default_factory=list)

    # Milestone progress
    milestone_progress: Dict[str, Dict] = Field(default_factory=dict)

    # Loan state
    loan_balance: float = 0.0
    loans_taken: int = 0

    # For tracking idle days (3+ idle = penalty)
    consecutive_idle_days: int = 0

    # Weekly counters (reset every 7 days)
    weekly_spend: float = 0.0
    weekly_waste: int = 0

    # Per-step demand tracking
    last_demand: Dict[str, int] = Field(default_factory=dict)
    last_day_profit: float = 0.0
    # Event countdown tracking
    event_countdowns: Dict[str, int] = Field(default_factory=dict)