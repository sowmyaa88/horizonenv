# server/constants.py

# ─── Products ────────────────────────────────────────────────────────────────

PRODUCTS = {
    "electronics": {
        "sell_price": 150,
        "cost": 100,
        "margin": 50,
        "shelf_life": None,        # None = never expires
        "elasticity": 1.2,         # demand drops 1.2% per 1% price increase
        "base_demand": 8,
        "warehouse_capacity": 200,
    },
    "clothing": {
        "sell_price": 40,
        "cost": 25,
        "margin": 15,
        "shelf_life": None,
        "elasticity": 1.5,
        "base_demand": 15,
        "warehouse_capacity": 300,
    },
    "groceries": {
        "sell_price": 10,
        "cost": 5,
        "margin": 5,
        "shelf_life": 5,           # expires after 5 days
        "elasticity": 0.4,         # people buy groceries regardless of price
        "base_demand": 30,
        "warehouse_capacity": 400,
    },
    "furniture": {
        "sell_price": 200,
        "cost": 130,
        "margin": 70,
        "shelf_life": None,
        "elasticity": 0.8,
        "base_demand": 4,
        "warehouse_capacity": 100,
    },
    "toys": {
        "sell_price": 25,
        "cost": 12,
        "margin": 13,
        "shelf_life": None,
        "elasticity": 1.3,
        "base_demand": 12,
        "warehouse_capacity": 250,
    },
}


# ─── Shipping ─────────────────────────────────────────────────────────────────

SHIPPING = {
    "slow":   {"cost_per_unit": 2,  "base_days": 5, "jitter": 2},  # arrives day +3 to +7
    "medium": {"cost_per_unit": 5,  "base_days": 3, "jitter": 1},  # arrives day +2 to +4
    "fast":   {"cost_per_unit": 10, "base_days": 1, "jitter": 0},  # guaranteed next day
}


# ─── Seasonal Events ──────────────────────────────────────────────────────────
# Each event: what day it starts, how long it lasts, which products it affects

EVENTS = [
    {
        "id": "black_friday",
        "name": "Black Friday",
        "start_day": 25,
        "duration": 3,
        "demand_multipliers": {
            "electronics": 3.5,
            "clothing": 2.5,
            "toys": 3.0,
            "furniture": 2.0,
            "groceries": 1.2,
        },
    },
    {
        "id": "back_to_school",
        "name": "Back to School",
        "start_day": 10,
        "duration": 3,
        "demand_multipliers": {
            "clothing": 2.0,
            "toys": 1.5,
            "electronics": 1.8,
            "furniture": 1.0,
            "groceries": 1.0,
        },
    },
    {
        "id": "supply_disruption",
        "name": "Supply Disruption",
        "start_day": 40,
        "duration": 4,
        "demand_multipliers": {          # disruption REDUCES demand
            "electronics": 0.5,
            "furniture": 0.6,
            "clothing": 0.8,
            "groceries": 1.0,
            "toys": 0.7,
        },
    },
    {
        "id": "holiday_season",
        "name": "Holiday Season",
        "start_day": 70,
        "duration": 5,
        "demand_multipliers": {
            "toys": 3.0,
            "electronics": 2.5,
            "clothing": 2.0,
            "furniture": 1.5,
            "groceries": 1.3,
        },
    },
    {
        "id": "competitor_launch",
        "name": "Competitor Launch",
        "start_day": 55,
        "duration": 3,
        "demand_multipliers": {          # competitor HURTS your sales
            "electronics": 0.4,
            "clothing": 0.6,
            "toys": 0.5,
            "furniture": 0.8,
            "groceries": 1.0,
        },
    },
]


# ─── Directive Templates ──────────────────────────────────────────────────────
# These get instantiated into actual directives per task config

DIRECTIVE_TEMPLATES = {
    "min_stock": {
        "description": "Maintain minimum stock level for a product",
        "penalty_range": (0.5, 5.0),
    },
    "budget_cap": {
        "description": "Daily spending capped at a limit",
        "penalty_range": (0.5, 1.0),
    },
    "shipping_rule": {
        "description": "A product must use a specific shipping method",
        "penalty_range": (0.5, 1.0),
    },
    "price_range": {
        "description": "A product must be priced within a multiplier range",
        "penalty_range": (0.5, 1.5),
    },
    "force_liquidate": {
        "description": "Liquidate all of a product by a deadline",
        "penalty_range": (3.0, 5.0),
    },
    "order_freeze": {
        "description": "No new purchase orders allowed",
        "penalty_range": (1.0, 3.0),
    },
    "waste_limit": {
        "description": "Zero waste allowed for a product during a window",
        "penalty_range": (1.0, 3.0),
    },
    "min_cash": {
        "description": "Maintain cash above a minimum threshold",
        "penalty_range": (2.0, 2.0),
    },
    "order_limit": {
        "description": "Max number of distinct products ordered per day",
        "penalty_range": (0.5, 1.0),
    },
}


# ─── Task Configs ─────────────────────────────────────────────────────────────

TASKS = {
    "easy": {
        "starting_cash": 2000,
        "starting_stock_multiplier": 1.0,   # full starting stock
        "warehouse_capacity_multiplier": 1.0,
        "num_directives": 5,
        "num_events": 0,
        "has_deceptive_directive": False,
        "has_conflicting_directives": False,
        "base_demand_multiplier": 1.0,
        "description": "Basic compliance and memory. No events, few directives.",
    },
    "medium": {
        "starting_cash": 1500,
        "starting_stock_multiplier": 0.6,
        "warehouse_capacity_multiplier": 1.0,
        "num_directives": 15,
        "num_events": 6,
        "has_deceptive_directive": True,
        "has_conflicting_directives": False,
        "base_demand_multiplier": 1.1,
        "description": "Directive modifications + seasonal planning.",
    },
    "hard": {
        "starting_cash": 1000,
        "starting_stock_multiplier": 0.3,
        "warehouse_capacity_multiplier": 0.8,  # tighter warehouse
        "num_directives": 27,
        "num_events": 12,
        "has_deceptive_directive": True,
        "has_conflicting_directives": True,
        "base_demand_multiplier": 1.2,
        "description": "Strategic violation + error recovery. 27 directives, packed events.",
    },
}


# ─── Reward Weights ───────────────────────────────────────────────────────────

REWARD_WEIGHTS = {
    "directives":   0.40,
    "planning":     0.20,
    "revenue":      0.15,
    "fulfillment":  0.15,
    "waste":        0.10,
}


# ─── Loan Config ─────────────────────────────────────────────────────────────

LOAN_CONFIG = {
    "amount": 500,
    "daily_interest_rate": 0.03,    # 3% compound daily
    "auto_repay_rate": 0.15,        # 15% of daily revenue goes to repayment
    "max_loans": 2,
    "eligibility_threshold": 100,   # cash must be below this to take a loan
}


# ─── Misc ─────────────────────────────────────────────────────────────────────

TOTAL_DAYS = 90
BANKRUPTCY_THRESHOLD = 10          # cash below this = true bankruptcy
IDLE_PENALTY_DAYS = 3              # consecutive idle days before penalty
PRICE_MULTIPLIER_MIN = 0.5
PRICE_MULTIPLIER_MAX = 1.5
WEEKEND_DEMAND_BOOST = 1.3         # demand 30% higher on weekends