# server/directives.py

import random
from typing import Dict, List, Optional
from server.constants import PRODUCTS, TASKS


class DirectiveEngine:
    """
    Handles everything related to corporate directives:
    - Generating directives for a task
    - Issuing them on the right day
    - Expiring / modifying them
    - Checking compliance each step
    """

    def __init__(self, task_config: Dict, rng: random.Random):
        self.task_config = task_config
        self.rng = rng
        self.all_directives: List[Dict] = []      # every directive for this episode
        self.active: Dict[str, Dict] = {}          # id → directive (currently active)
        self._generated = False

    # ─────────────────────────────────────────────────────────────────────────
    # GENERATE
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_directives(self):
        """Build the full directive list for this episode."""
        cfg = self.task_config
        n = cfg["num_directives"]
        directives = []
        used_ids = set()

        def make_id(prefix):
            i = 1
            while f"{prefix}{i:02d}" in used_ids:
                i += 1
            uid = f"{prefix}{i:02d}"
            used_ids.add(uid)
            return uid

        # Spread issue days across 90 days (first one always day 1)
        issue_days = [1] + sorted(self.rng.sample(range(2, 80), min(n - 1, 78)))

        product_list = list(PRODUCTS.keys())

        for i, issue_day in enumerate(issue_days[:n]):
            dtype = self.rng.choice([
                "min_stock", "budget_cap", "shipping_rule",
                "price_range", "order_freeze", "waste_limit",
                "min_cash", "order_limit"
            ])

            did = make_id("D")
            product = self.rng.choice(product_list)
            expiry_day = None

            if dtype == "min_stock":
                min_qty = self.rng.choice([20, 30, 50, 80])
                d = {
                    "id": did,
                    "type": dtype,
                    "issue_day": issue_day,
                    "expiry_day": expiry_day,
                    "product": product,
                    "min_qty": min_qty,
                    "penalty": self.rng.choice([0.5, 1.0, 2.0]),
                    "text": (
                        f"[{did}] DIRECTIVE: Maintain at least {min_qty} units of "
                        f"{product} in stock at all times. "
                        f"Penalty: -{self.rng.choice([0.5, 1.0, 2.0])} per step violated."
                    ),
                    "active": False,
                    "superseded_by": None,
                }

            elif dtype == "budget_cap":
                cap = self.rng.choice([300, 350, 400, 500])
                d = {
                    "id": did,
                    "type": dtype,
                    "issue_day": issue_day,
                    "expiry_day": expiry_day,
                    "cap": cap,
                    "penalty": 0.5,
                    "text": (
                        f"[{did}] DIRECTIVE: Daily spending is capped at ${cap}. "
                        f"Penalty: -0.5 per step exceeded."
                    ),
                    "active": False,
                    "superseded_by": None,
                }

            elif dtype == "shipping_rule":
                method = self.rng.choice(["fast", "medium"])
                d = {
                    "id": did,
                    "type": dtype,
                    "issue_day": issue_day,
                    "expiry_day": expiry_day,
                    "product": product,
                    "required_method": method,
                    "penalty": 0.5,
                    "text": (
                        f"[{did}] DIRECTIVE: All {product} orders must use "
                        f"{method} shipping. Penalty: -0.5 per order violated."
                    ),
                    "active": False,
                    "superseded_by": None,
                }

            elif dtype == "price_range":
                min_mult = self.rng.choice([0.8, 1.0, 1.2])
                max_mult = min_mult + self.rng.choice([0.2, 0.3])
                d = {
                    "id": did,
                    "type": dtype,
                    "issue_day": issue_day,
                    "expiry_day": expiry_day,
                    "product": product,
                    "min_mult": min_mult,
                    "max_mult": max_mult,
                    "penalty": 0.5,
                    "text": (
                        f"[{did}] DIRECTIVE: {product.capitalize()} must be priced "
                        f"between {min_mult}x and {max_mult}x. "
                        f"Penalty: -0.5 per step violated."
                    ),
                    "active": False,
                    "superseded_by": None,
                }

            elif dtype == "order_freeze":
                duration = self.rng.randint(5, 15)
                expiry_day = issue_day + duration
                d = {
                    "id": did,
                    "type": dtype,
                    "issue_day": issue_day,
                    "expiry_day": expiry_day,
                    "penalty": 1.0,
                    "text": (
                        f"[{did}] DIRECTIVE: No new purchase orders allowed "
                        f"from day {issue_day} to day {expiry_day}. "
                        f"Penalty: -1.0 per day violated."
                    ),
                    "active": False,
                    "superseded_by": None,
                }

            elif dtype == "waste_limit":
                duration = self.rng.randint(7, 20)
                expiry_day = issue_day + duration
                d = {
                    "id": did,
                    "type": dtype,
                    "issue_day": issue_day,
                    "expiry_day": expiry_day,
                    "product": "groceries",
                    "penalty": 1.0,
                    "text": (
                        f"[{did}] DIRECTIVE: Zero grocery waste allowed "
                        f"from day {issue_day} to day {expiry_day}. "
                        f"Penalty: -1.0 per step with waste."
                    ),
                    "active": False,
                    "superseded_by": None,
                }

            elif dtype == "min_cash":
                threshold = self.rng.choice([200, 300, 400])
                d = {
                    "id": did,
                    "type": dtype,
                    "issue_day": issue_day,
                    "expiry_day": expiry_day,
                    "threshold": threshold,
                    "penalty": 2.0,
                    "text": (
                        f"[{did}] DIRECTIVE: Maintain cash above ${threshold} at all times. "
                        f"Penalty: -2.0 per step violated."
                    ),
                    "active": False,
                    "superseded_by": None,
                }

            elif dtype == "order_limit":
                max_products = self.rng.choice([2, 3, 4])
                d = {
                    "id": did,
                    "type": dtype,
                    "issue_day": issue_day,
                    "expiry_day": expiry_day,
                    "max_products": max_products,
                    "penalty": 0.5,
                    "text": (
                        f"[{did}] DIRECTIVE: Order at most {max_products} distinct "
                        f"products per day. Penalty: -0.5 per day exceeded."
                    ),
                    "active": False,
                    "superseded_by": None,
                }

            else:
                continue

            directives.append(d)

        # Add conflicting directives for hard mode
        if cfg.get("has_conflicting_directives"):
            directives += self._make_conflicting_pair(used_ids, issue_days)

        # Add deceptive directive for medium/hard
        if cfg.get("has_deceptive_directive"):
            directives += self._make_deceptive_directive(used_ids, issue_days)

        # Sort by issue day
        directives.sort(key=lambda d: d["issue_day"])
        self.all_directives = directives

    def _make_conflicting_pair(self, used_ids, issue_days) -> List[Dict]:
        """Two directives that are impossible to satisfy simultaneously."""
        freeze_id = "CF01"
        stock_id = "CF02"
        used_ids.add(freeze_id)
        used_ids.add(stock_id)
        issue_day = self.rng.randint(20, 40)

        freeze = {
            "id": freeze_id,
            "type": "order_freeze",
            "issue_day": issue_day,
            "expiry_day": issue_day + 10,
            "penalty": 2.0,
            "text": (
                f"[{freeze_id}] DIRECTIVE: No new purchase orders allowed "
                f"from day {issue_day} to day {issue_day + 10}. "
                f"Penalty: -2.0 per day violated. "
                f"NOTE: This conflicts with {stock_id}."
            ),
            "active": False,
            "superseded_by": None,
            "is_conflict": True,
        }
        stock = {
            "id": stock_id,
            "type": "min_stock",
            "issue_day": issue_day,
            "expiry_day": issue_day + 10,
            "product": "electronics",
            "min_qty": 80,
            "penalty": 1.0,
            "text": (
                f"[{stock_id}] DIRECTIVE: Maintain 80+ electronics in stock "
                f"from day {issue_day} to day {issue_day + 10}. "
                f"Penalty: -1.0 per step. "
                f"NOTE: This may conflict with {freeze_id}."
            ),
            "active": False,
            "superseded_by": None,
            "is_conflict": True,
        }
        return [freeze, stock]

    def _make_deceptive_directive(self, used_ids, issue_days) -> List[Dict]:
        """Looks helpful but triggers cascading waste penalties."""
        did = "DC01"
        used_ids.add(did)
        issue_day = self.rng.randint(5, 20)
        return [{
            "id": did,
            "type": "min_stock",
            "issue_day": issue_day,
            "expiry_day": None,
            "product": "groceries",
            "min_qty": 200,
            "penalty": 0.5,
            "text": (
                f"[{did}] URGENT DIRECTIVE: Stock 200+ groceries immediately "
                f"for the summer push! Penalty only -0.5 if ignored. "
                f"(Note: groceries expire in 5 days.)"
            ),
            "active": False,
            "superseded_by": None,
            "is_deceptive": True,
        }]
    
    # ─────────────────────────────────────────────────────────────────────────
    # TICK (called every step)
    # ─────────────────────────────────────────────────────────────────────────

    def tick(self, state) -> List[Dict]:
        """
        Issue new directives, expire old ones.
        Returns list of NEW directives issued today (full text shown once).
        """
        if not self._generated:
            self._generate_directives()
            self._generated = True

        new_directives = []
        current_day = state.current_day

        for d in self.all_directives:
            # Issue directive on its start day
            if d["issue_day"] == current_day and not d["active"] and not d.get("superseded_by"):
                d["active"] = True
                self.active[d["id"]] = d
                new_directives.append(d)

            # Expire directive if it has an expiry day
            if d.get("expiry_day") and current_day > d["expiry_day"] and d["active"]:
                d["active"] = False
                self.active.pop(d["id"], None)

        # Update state's active directive IDs
        state.active_directive_ids = list(self.active.keys())

        return new_directives

    # ─────────────────────────────────────────────────────────────────────────
    # COMPLIANCE CHECK (called every step after actions are processed)
    # ─────────────────────────────────────────────────────────────────────────

    def check_compliance(self, state, action) -> List[Dict]:
        """Check all active directives. Returns list of violations."""
        violations = []

        for did, d in self.active.items():
            dtype = d["type"]
            violation = None

            if dtype == "min_stock":
                product = d["product"]
                current_stock = sum(
                    b[0] for b in state.inventory.get(product, [])
                )
                if current_stock < d["min_qty"]:
                    violation = {
                        "id": did,
                        "type": dtype,
                        "message": (
                            f"{product} stock is {current_stock}, "
                            f"below required {d['min_qty']}"
                        ),
                        "penalty": d["penalty"],
                    }

            elif dtype == "budget_cap":
                if state.weekly_spend > d["cap"]:
                    violation = {
                        "id": did,
                        "type": dtype,
                        "message": (
                            f"Daily spend ${state.weekly_spend:.0f} "
                            f"exceeded cap ${d['cap']}"
                        ),
                        "penalty": d["penalty"],
                    }

            elif dtype == "shipping_rule":
                product = d["product"]
                if product in action.buy_quantities and action.buy_quantities[product] > 0:
                    used_method = action.delivery_methods.get(product, "slow")
                    if used_method != d["required_method"]:
                        violation = {
                            "id": did,
                            "type": dtype,
                            "message": (
                                f"{product} shipped via {used_method}, "
                                f"required {d['required_method']}"
                            ),
                            "penalty": d["penalty"],
                        }

            elif dtype == "price_range":
                product = d["product"]
                used_mult = action.price_multipliers.get(product, 1.0)
                if not (d["min_mult"] <= used_mult <= d["max_mult"]):
                    violation = {
                        "id": did,
                        "type": dtype,
                        "message": (
                            f"{product} price multiplier {used_mult} "
                            f"outside allowed range [{d['min_mult']}, {d['max_mult']}]"
                        ),
                        "penalty": d["penalty"],
                    }

            elif dtype == "order_freeze":
                if action.buy_quantities:
                    violation = {
                        "id": did,
                        "type": dtype,
                        "message": "Purchased during an active order freeze",
                        "penalty": d["penalty"],
                    }

            elif dtype == "waste_limit":
                if state.weekly_waste > 0:
                    violation = {
                        "id": did,
                        "type": dtype,
                        "message": f"Grocery waste detected: {state.weekly_waste} units",
                        "penalty": d["penalty"],
                    }

            elif dtype == "min_cash":
                if state.cash < d["threshold"]:
                    violation = {
                        "id": did,
                        "type": dtype,
                        "message": (
                            f"Cash ${state.cash:.0f} below "
                            f"required ${d['threshold']}"
                        ),
                        "penalty": d["penalty"],
                    }

            elif dtype == "order_limit":
                num_products_ordered = len([
                    p for p, q in action.buy_quantities.items() if q > 0
                ])
                if num_products_ordered > d["max_products"]:
                    violation = {
                        "id": did,
                        "type": dtype,
                        "message": (
                            f"Ordered {num_products_ordered} products, "
                            f"max allowed is {d['max_products']}"
                        ),
                        "penalty": d["penalty"],
                    }

            if violation:
                violations.append(violation)

        # Save to state for observation
        state.directive_violations = violations
        return violations

    # ─────────────────────────────────────────────────────────────────────────
    # MODIFICATION (supersede an old directive with a new one)
    # ─────────────────────────────────────────────────────────────────────────

    def supersede(self, old_id: str, new_directive: Dict):
        """
        Mark old_id as superseded, activate new_directive.
        Used when a later directive UPDATES an earlier one.
        """
        if old_id in self.active:
            self.active[old_id]["active"] = False
            self.active[old_id]["superseded_by"] = new_directive["id"]
            del self.active[old_id]

        new_directive["active"] = True
        self.active[new_directive["id"]] = new_directive

    def get_all_directives(self) -> List[Dict]:
        """Return all directives (for /tasks endpoint)."""
        if not self._generated:
            self._generate_directives()
            self._generated = True
        return self.all_directives