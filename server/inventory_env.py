# server/inventory_env.py

import random
from typing import Dict, List, Optional, Tuple
from models import (
    InventoryAction, InventoryObservation,
    InventoryState, DeliveryBatch
)
from server.constants import (
    PRODUCTS, SHIPPING, EVENTS, TASKS,
    REWARD_WEIGHTS, LOAN_CONFIG,
    TOTAL_DAYS, BANKRUPTCY_THRESHOLD,
    IDLE_PENALTY_DAYS, PRICE_MULTIPLIER_MIN,
    PRICE_MULTIPLIER_MAX, WEEKEND_DEMAND_BOOST
)


class InventoryEnv:
    """
    The Quartermaster RL environment.
    Manages a retail store over 90 days.
    """

    def __init__(self, task_name: str = "easy", seed: int = 42):
        self.task_name = task_name
        self.task_config = TASKS[task_name]
        self.seed = seed
        self.rng = random.Random(seed)
        self.state: Optional[InventoryState] = None
        self.directives: List[Dict] = []       # will be filled by directives engine

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> InventoryObservation:
        """Start a fresh episode. Returns the first observation."""
        self.rng = random.Random(self.seed)
        cfg = self.task_config

        # Build starting inventory — scale by task difficulty
        starting_inventory = {}
        for product, props in PRODUCTS.items():
            qty = int(20 * cfg["starting_stock_multiplier"])
            shelf_life = props["shelf_life"]
            days_left = shelf_life if shelf_life else None
            starting_inventory[product] = [[qty, days_left]]

        self.state = InventoryState(
            current_day=1,
            total_days=TOTAL_DAYS,
            cash=cfg["starting_cash"],
            total_profit=0.0,
            inventory=starting_inventory,
            pending_deliveries=[],
            agent_notes="",
            agent_weekly_plan="",
            active_directive_ids=[],
            directive_violations=[],
            milestone_progress=self._init_milestones(),
            loan_balance=0.0,
            loans_taken=0,
            consecutive_idle_days=0,
            weekly_spend=0.0,
            weekly_waste=0,
            last_demand={p: 0 for p in PRODUCTS},
            last_day_profit=0.0,
        )

        return self._build_observation(new_directives=[])

    # ─────────────────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action: InventoryAction) -> Tuple[InventoryObservation, float, bool]:
        """
        Process one day. Returns (observation, reward, done).
        Order matters — see step execution order in README.
        """
        s = self.state
        violations = []
        sparse_reward = 0.0

        # 1. Save agent memory
        s.agent_notes = action.notes_to_self
        if action.weekly_plan is not None:
            s.agent_weekly_plan = action.weekly_plan

        # 2. Process loan request
        if action.take_loan:
            sparse_reward += self._process_loan()

        # 3. Compound interest on outstanding loan
        if s.loan_balance > 0:
            s.loan_balance *= (1 + LOAN_CONFIG["daily_interest_rate"])

        # 4. Weekly reset (every 7 days)
        if s.current_day % 7 == 1:
            s.weekly_spend = 0.0
            s.weekly_waste = 0

        # 5. Issue new directives, expire old ones (handled by directives engine)
        new_directives = self._tick_directives()

        # 6. Tick event countdowns
        self._tick_events()

        # 7. Expire groceries
        expired_waste = self._expire_inventory()
        s.weekly_waste += expired_waste

        # 8. Receive arriving deliveries
        self._receive_deliveries()

        # 9. Process purchase orders
        order_penalty = self._process_orders(action)
        sparse_reward += order_penalty

        # 10. Generate demand
        demand = self._generate_demand(action.price_multipliers)
        s.last_demand = demand

        # 11. Sell products FIFO
        revenue, units_sold, units_demanded = self._sell_products(demand)

        # 12. Process liquidation
        liquidated = self._process_liquidation(action.liquidate)
        s.weekly_waste += liquidated

        # 13. Auto-repay loan
        if s.loan_balance > 0 and revenue > 0:
            repayment = min(revenue * LOAN_CONFIG["auto_repay_rate"], s.loan_balance)
            s.loan_balance -= repayment

        # 14. Calculate day profit
        day_profit = revenue - self._calculate_costs(action)
        s.last_day_profit = day_profit
        s.total_profit += day_profit
        s.cash += day_profit

        # 15. Check directive compliance
        violations = self._check_directives(action)

        # 16. Check milestones
        milestone_bonus = self._check_milestones()
        sparse_reward += milestone_bonus

        # 17. Check idle penalty
        if not action.buy_quantities and not action.liquidate:
            s.consecutive_idle_days += 1
            if s.consecutive_idle_days >= IDLE_PENALTY_DAYS:
                sparse_reward -= 1.0
        else:
            s.consecutive_idle_days = 0

        # 18. Check bankruptcy
        done = False
        if s.cash < BANKRUPTCY_THRESHOLD and s.loans_taken >= LOAN_CONFIG["max_loans"]:
            sparse_reward -= 2.0
            done = True

        # 19. Compute dense reward
        dense_reward = self._compute_reward(
            violations=violations,
            action=action,
            units_sold=units_sold,
            units_demanded=units_demanded,
            expired_waste=expired_waste,
            liquidated=liquidated,
        )

        total_reward = dense_reward + sparse_reward

        # 20. Advance day
        s.current_day += 1
        if s.current_day > s.total_days:
            # End of episode: subtract remaining loan balance
            s.total_profit -= s.loan_balance
            done = True

        obs = self._build_observation(new_directives=new_directives)
        return obs, total_reward, done

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _init_milestones(self) -> Dict:
        return {
            "profit_by_day50": {"target": 2000, "deadline": 50, "achieved": False, "bonus": 2.0},
            "zero_grocery_waste_14days": {"target": 14, "deadline": 90, "achieved": False, "bonus": 4.0, "streak": 0},
            "stock_toys_by_day79": {"target": 100, "deadline": 79, "achieved": False, "bonus": 3.0},
        }

    def _tick_directives(self) -> List[Dict]:
        """Delegate to directives engine. Returns new directives issued today."""
        from server.directives import DirectiveEngine
        if not hasattr(self, '_directive_engine'):
            self._directive_engine = DirectiveEngine(self.task_config, self.rng)
        return self._directive_engine.tick(self.state)

    def _tick_events(self):
        """Advance event countdowns."""
        s = self.state
        if not s.event_countdowns:
            for event in EVENTS:
                days_until = event["start_day"] - s.current_day
                s.event_countdowns[event["id"]] = days_until

        for event in EVENTS:
            s.event_countdowns[event["id"]] -= 1

    def _get_active_event_multipliers(self) -> Dict[str, float]:
        """Returns combined demand multipliers from all currently active events."""
        s = self.state
        multipliers = {p: 1.0 for p in PRODUCTS}
        if not s.event_countdowns:
            return multipliers

        for event in EVENTS:
            eid = event["id"]
            countdown = s.event_countdowns.get(eid, 999)
            if -event["duration"] < countdown <= 0:
                for product, mult in event["demand_multipliers"].items():
                    multipliers[product] *= mult

        return multipliers

    def _expire_inventory(self) -> int:
        """Tick down shelf life, remove expired batches. Returns units wasted."""
        s = self.state
        total_expired = 0
        for product, batches in s.inventory.items():
            shelf_life = PRODUCTS[product]["shelf_life"]
            if shelf_life is None:
                continue
            surviving = []
            for batch in batches:
                qty, days_left = batch
                days_left -= 1
                if days_left <= 0:
                    total_expired += qty
                else:
                    surviving.append([qty, days_left])
            s.inventory[product] = surviving
        return total_expired

    def _receive_deliveries(self):
        """Move arriving shipments into inventory."""
        s = self.state
        still_transit = []
        for delivery in s.pending_deliveries:
            if delivery.arrives_day <= s.current_day:
                shelf_life = PRODUCTS[delivery.product]["shelf_life"]
                batch = [delivery.quantity, shelf_life] if shelf_life else [delivery.quantity, None]
                s.inventory[delivery.product].append(batch)
            else:
                still_transit.append(delivery)
        s.pending_deliveries = still_transit

    def _process_orders(self, action: InventoryAction) -> float:
        """Place purchase orders. Returns penalty if order is unaffordable."""
        s = self.state
        penalty = 0.0
        for product, qty in action.buy_quantities.items():
            if qty <= 0 or product not in PRODUCTS:
                continue
            cost_per_unit = PRODUCTS[product]["cost"]
            ship_method = action.delivery_methods.get(product, "slow")
            ship_cost = SHIPPING[ship_method]["cost_per_unit"]
            total_cost = qty * (cost_per_unit + ship_cost)

            if total_cost > s.cash:
                penalty -= 1.0   # attempted unaffordable order
                continue

            s.cash -= total_cost
            s.weekly_spend += total_cost

            # Schedule delivery with jitter
            base_days = SHIPPING[ship_method]["base_days"]
            jitter = SHIPPING[ship_method]["jitter"]
            arrival_offset = base_days + self.rng.randint(-jitter, jitter)
            arrival_day = s.current_day + max(1, arrival_offset)

            s.pending_deliveries.append(DeliveryBatch(
                product=product,
                quantity=qty,
                arrives_day=arrival_day,
                delivery_method=ship_method,
            ))

        return penalty

    def _generate_demand(self, price_multipliers: Dict[str, float]) -> Dict[str, int]:
        """Calculate today's demand per product."""
        s = self.state
        cfg = self.task_config
        event_multipliers = self._get_active_event_multipliers()
        is_weekend = s.current_day % 7 in (0, 6)
        demand = {}

        for product, props in PRODUCTS.items():
            base = props["base_demand"] * cfg["base_demand_multiplier"]

            # Weekend boost
            if is_weekend:
                base *= WEEKEND_DEMAND_BOOST

            # Event multiplier
            base *= event_multipliers.get(product, 1.0)

            # Price elasticity: higher price = lower demand
            price_mult = price_multipliers.get(product, 1.0)
            price_mult = max(PRICE_MULTIPLIER_MIN, min(PRICE_MULTIPLIER_MAX, price_mult))
            elasticity = props["elasticity"]
            price_effect = 1 - (elasticity * (price_mult - 1))
            base *= max(0.1, price_effect)

            # Add some randomness (+/- 20%)
            noise = self.rng.uniform(0.8, 1.2)
            demand[product] = max(0, int(base * noise))

        return demand

    def _sell_products(self, demand: Dict[str, int]) -> Tuple[float, Dict, Dict]:
        """Sell using FIFO. Returns (revenue, units_sold, units_demanded)."""
        s = self.state
        total_revenue = 0.0
        units_sold = {}
        units_demanded = dict(demand)

        for product, demanded in demand.items():
            batches = s.inventory.get(product, [])
            sold = 0
            remaining_demand = demanded
            new_batches = []

            for batch in batches:
                if remaining_demand <= 0:
                    new_batches.append(batch)
                    continue
                qty, days_left = batch
                sell_qty = min(qty, remaining_demand)
                sold += sell_qty
                remaining_demand -= sell_qty
                if qty - sell_qty > 0:
                    new_batches.append([qty - sell_qty, days_left])

            s.inventory[product] = new_batches
            units_sold[product] = sold
            price = PRODUCTS[product]["sell_price"]
            total_revenue += sold * price

        return total_revenue, units_sold, units_demanded

    def _process_liquidation(self, liquidate: Dict[str, int]) -> int:
        """Dispose of stock. Returns total units liquidated."""
        s = self.state
        total = 0
        for product, qty in liquidate.items():
            if qty <= 0 or product not in s.inventory:
                continue
            remaining = qty
            new_batches = []
            for batch in s.inventory[product]:
                if remaining <= 0:
                    new_batches.append(batch)
                    continue
                b_qty, days_left = batch
                remove = min(b_qty, remaining)
                remaining -= remove
                total += remove
                if b_qty - remove > 0:
                    new_batches.append([b_qty - remove, days_left])
            s.inventory[product] = new_batches
        return total

    def _calculate_costs(self, action: InventoryAction) -> float:
        """Sum of all costs this day (already deducted from cash, just for profit calc)."""
        total = 0.0
        for product, qty in action.buy_quantities.items():
            if product not in PRODUCTS:
                continue
            ship_method = action.delivery_methods.get(product, "slow")
            cost = PRODUCTS[product]["cost"] + SHIPPING[ship_method]["cost_per_unit"]
            total += qty * cost
        return total

    def _check_directives(self, action: InventoryAction) -> List[Dict]:
        """Delegate compliance check to directive engine."""
        if not hasattr(self, '_directive_engine'):
            return []
        return self._directive_engine.check_compliance(self.state, action)

    def _check_milestones(self) -> float:
        """Check and award milestone bonuses."""
        s = self.state
        bonus = 0.0
        mp = s.milestone_progress

        # Profit by day 50
        m = mp["profit_by_day50"]
        if not m["achieved"] and s.current_day <= m["deadline"]:
            if s.total_profit >= m["target"]:
                m["achieved"] = True
                bonus += m["bonus"]

        # Zero grocery waste streak
        m = mp["zero_grocery_waste_14days"]
        if not m["achieved"]:
            if s.weekly_waste == 0:
                m["streak"] = m.get("streak", 0) + 1
            else:
                m["streak"] = 0
            if m["streak"] >= m["target"]:
                m["achieved"] = True
                bonus += m["bonus"]

        # Stock 100+ toys by day 79
        m = mp["stock_toys_by_day79"]
        if not m["achieved"] and s.current_day <= m["deadline"]:
            toy_stock = sum(b[0] for b in s.inventory.get("toys", []))
            if toy_stock >= m["target"]:
                m["achieved"] = True
                bonus += m["bonus"]

        return bonus

    def _compute_reward(self, violations, action, units_sold,
                        units_demanded, expired_waste, liquidated) -> float:
        """Compute the dense per-step reward (range -1 to +1)."""
        s = self.state
        weights = REWARD_WEIGHTS

        # R_directives: compliance score
        if not violations:
            r_directives = 1.0
        else:
            total_penalty = sum(v.get("penalty", 0.5) for v in violations)
            r_directives = max(-1.0, 1.0 - total_penalty)

        # R_planning: quality of notes (basic version — directives engine scores this)
        r_planning = self._score_planning(action)

        # R_revenue: revenue vs max possible
        max_revenue = sum(
            units_demanded.get(p, 0) * PRODUCTS[p]["sell_price"]
            for p in PRODUCTS
        )
        actual_revenue = sum(
            units_sold.get(p, 0) * PRODUCTS[p]["sell_price"]
            for p in PRODUCTS
        )
        r_revenue = (actual_revenue / max_revenue) if max_revenue > 0 else 0.0
        r_revenue = r_revenue * 2 - 1   # scale to [-1, 1]

        # R_fulfillment: units sold / units demanded
        total_demanded = sum(units_demanded.values())
        total_sold = sum(units_sold.values())
        r_fulfillment = (total_sold / total_demanded) if total_demanded > 0 else 1.0
        r_fulfillment = r_fulfillment * 2 - 1

        # R_waste: penalize waste
        total_waste = expired_waste + liquidated
        r_waste = 1.0 if total_waste == 0 else max(-1.0, 1.0 - total_waste * 0.1)

        reward = (
            weights["directives"]   * r_directives +
            weights["planning"]     * r_planning +
            weights["revenue"]      * r_revenue +
            weights["fulfillment"]  * r_fulfillment +
            weights["waste"]        * r_waste
        )
        return max(-1.0, min(1.0, reward))

    def _score_planning(self, action: InventoryAction) -> float:
        """Content-aware planning note scorer."""
        s = self.state
        notes = action.notes_to_self + (action.weekly_plan or "")

        if not notes.strip():
            return -1.0

        score = 0.0

        # 1. Directive tracking: does agent mention active directive IDs?
        if s.active_directive_ids:
            mentioned = sum(1 for did in s.active_directive_ids if did in notes)
            score += 0.5 * (mentioned / len(s.active_directive_ids))

        # 2. Situational awareness: mentions products or quantities
        product_mentions = sum(1 for p in PRODUCTS if p in notes.lower())
        score += 0.3 * min(1.0, product_mentions / 3)

        # 3. Note evolution: penalize copy-paste from last step
        if notes.strip() == s.agent_notes.strip() and notes.strip():
            score -= 0.3
        else:
            score += 0.3

        # 4. Violation acknowledgment
        if s.directive_violations and any(
            v.get("id", "") in notes for v in s.directive_violations
        ):
            score += 0.2

        # 5. Plan structure: bullet points, numbers
        has_structure = any(c in notes for c in ["-", "*", "1.", "2.", "\n"])
        if has_structure:
            score += 0.2

        return max(-1.0, min(1.0, score))

    def _build_observation(self, new_directives: List[Dict]) -> InventoryObservation:
        """Package current state into an observation for the agent."""
        s = self.state
        cfg = self.task_config

        # Remaining warehouse capacity
        remaining_capacity = {}
        for product, props in PRODUCTS.items():
            cap = int(props["warehouse_capacity"] * cfg["warehouse_capacity_multiplier"])
            current_stock = sum(b[0] for b in s.inventory.get(product, []))
            remaining_capacity[product] = max(0, cap - current_stock)

        # Event countdowns
        event_countdowns = dict(s.event_countdowns)

        # Pending deliveries as dicts
        deliveries = [
            {"product": d.product, "quantity": d.quantity, "arrives_day": d.arrives_day}
            for d in s.pending_deliveries
        ]

        return InventoryObservation(
            current_day=s.current_day,
            total_days=s.total_days,
            total_cash=round(s.cash, 2),
            day_profit=round(s.last_day_profit, 2),
            total_profit=round(s.total_profit, 2),
            demand_today=s.last_demand,
            updated_inventory=s.inventory,
            remaining_capacity=remaining_capacity,
            updated_events=event_countdowns,
            updated_deliveries=deliveries,
            new_directives=new_directives,
            active_directive_ids=s.active_directive_ids,
            directive_violations_last_step=s.directive_violations,
            milestones=s.milestone_progress,
            agent_notes=s.agent_notes,
            agent_weekly_plan=s.agent_weekly_plan,
            loan_balance=round(s.loan_balance, 2),
            loans_taken=s.loans_taken,
            loans_remaining=LOAN_CONFIG["max_loans"] - s.loans_taken,
        )

    def _process_loan(self) -> float:
        """Handle loan request. Returns penalty if ineligible."""
        s = self.state
        if (s.cash < LOAN_CONFIG["eligibility_threshold"] and
                s.loans_taken < LOAN_CONFIG["max_loans"]):
            s.cash += LOAN_CONFIG["amount"]
            s.loan_balance += LOAN_CONFIG["amount"]
            s.loans_taken += 1
            return 0.0
        return 0.0   # no penalty for failed loan request, just ignored

    def get_state_summary(self) -> Dict:
        """Quick summary for the /state API endpoint."""
        s = self.state
        return {
            "day": s.current_day,
            "cash": round(s.cash, 2),
            "total_profit": round(s.total_profit, 2),
            "inventory": {p: sum(b[0] for b in batches) for p, batches in s.inventory.items()},
            "loans_taken": s.loans_taken,
            "loan_balance": round(s.loan_balance, 2),
        }