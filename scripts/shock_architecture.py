"""
Four-layer shock architecture for SupplySim.

Layers:
1. TopologyConfig — network structure (regions, chokepoints, capacity heterogeneity)
2. ShockGenerationConfig — how disruptions are seeded
3. ShockDynamicsConfig — how shocks evolve over time
4. PolicyInterfaceConfig — friction and constraints policies face

Also implements:
- ShockState — runtime tracking of individual shock events
- ShockEngine — manages shock lifecycle each timestep
- PolicyInterfaceLayer — enforces rerouting constraints and nonlinear costs
- ArchitectureConfig — bundles all four layer configs
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Layer 1 — Network Topology
# ---------------------------------------------------------------------------

@dataclass
class TopologyConfig:
    num_inner_layers: int = 2
    num_per_layer: int = 10

    # Regional clustering
    num_regions: int = 3
    region_assignment: str = "random"  # "random" | "clustered" | "tiered"

    # Chokepoint structure
    chokepoint_fraction: float = 0.2
    chokepoint_in_degree_multiplier: float = 2.5

    # Supplier capacity heterogeneity
    supplier_capacity_cv: float = 0.4

    # Path redundancy
    min_num_suppliers: int = 2
    max_num_suppliers: int = 3

    # Lead-time variance
    lead_time_mean: int = 2
    lead_time_cv: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TopologyConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Layer 2 — Shock Generation
# ---------------------------------------------------------------------------

@dataclass
class ShockGenerationConfig:
    shock_prob: float = 0.20

    event_type_probs: Dict[str, float] = field(default_factory=lambda: {
        "localized": 0.5,
        "regional": 0.3,
        "cascade": 0.2,
    })

    magnitude_dist: str = "beta"  # "fixed" | "uniform" | "beta"
    magnitude_mean: float = 0.85
    magnitude_cv: float = 0.20

    epicenter_bias: str = "region"  # "uniform" | "centrality" | "region"

    max_concurrent_shocks: int = 3
    shock_overlap_prob: float = 0.3

    # Fraction of firms per product to hit in a localized shock.
    # E.g., 0.5 means hit half the firms, keeping the rest as reroute targets.
    localized_firm_fraction: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShockGenerationConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Layer 3 — Shock Dynamics
# ---------------------------------------------------------------------------

@dataclass
class ShockDynamicsConfig:
    duration_dist: str = "geometric"
    duration_mean: float = 15.0

    contagion_radius: int = 2
    contagion_prob_per_hop: float = 0.4

    recovery_shape: str = "linear"  # "instant" | "linear" | "concave"
    recovery_steps: int = 8
    recovery_rate: float = 1.05

    demand_surge_enabled: bool = False
    demand_surge_multiplier: float = 1.3
    demand_surge_lag: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShockDynamicsConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Layer 4 — Policy Interface
# ---------------------------------------------------------------------------

@dataclass
class PolicyInterfaceConfig:
    reroute_setup_delay: int = 0
    reroute_capacity_fraction: float = 1.0

    expedite_budget: float = 50_000
    expedite_cost_model: str = "linear"  # "linear" | "convex" | "step"
    expedite_convexity: float = 0.0
    expedite_capacity_cap: float = 1.0

    # Reroute supply bonus: when rerouting away from a shocked supplier,
    # the new supplier provides bonus supply units to the buyer (models
    # emergency inventory/expedited production from the new relationship).
    # Only triggers when old supplier's product has shock severity > 0.
    reroute_supply_bonus: float = 150.0

    observability: str = "full"  # "full" | "delayed" | "noisy"
    observation_delay: int = 0
    observation_noise_cv: float = 0.0

    max_order_age: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyInterfaceConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureConfig:
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    shock_generation: ShockGenerationConfig = field(default_factory=ShockGenerationConfig)
    shock_dynamics: ShockDynamicsConfig = field(default_factory=ShockDynamicsConfig)
    policy_interface: PolicyInterfaceConfig = field(default_factory=PolicyInterfaceConfig)

    def to_dict(self) -> dict:
        return {
            "topology": self.topology.to_dict(),
            "shock_generation": self.shock_generation.to_dict(),
            "shock_dynamics": self.shock_dynamics.to_dict(),
            "policy_interface": self.policy_interface.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ArchitectureConfig":
        return cls(
            topology=TopologyConfig.from_dict(d.get("topology", {})),
            shock_generation=ShockGenerationConfig.from_dict(d.get("shock_generation", {})),
            shock_dynamics=ShockDynamicsConfig.from_dict(d.get("shock_dynamics", {})),
            policy_interface=PolicyInterfaceConfig.from_dict(d.get("policy_interface", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ArchitectureConfig":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Supply Graph — topology-aware wrapper
# ---------------------------------------------------------------------------

class SupplyGraph:
    """
    Wraps the static graph structures and adds region/chokepoint/capacity info.
    Built once at episode start; policies cannot modify it.
    """

    def __init__(
        self,
        firms: list,
        products: list,
        prod_graph,  # DataFrame
        firm2prods: dict,
        prod2firms: dict,
        inputs2supplier: dict,
        topology_config: TopologyConfig,
        rng: np.random.Generator,
    ):
        self.firms = firms
        self.products = products
        self.prod_graph = prod_graph
        self.firm2prods = firm2prods
        self.prod2firms = prod2firms
        self.inputs2supplier = dict(inputs2supplier)

        self.exog_prods = sorted(
            set(prod_graph.source.values) - set(prod_graph.dest.values)
        )
        self.consumer_prods = sorted(
            set(prod_graph.dest.values) - set(prod_graph.source.values)
        )

        # Build networkx DAG
        self._nx_graph = nx.DiGraph()
        for row in prod_graph.itertuples(index=False):
            self._nx_graph.add_edge(row.source, row.dest, units=float(row.units))

        # Firm indices
        self.firm2idx = {f: i for i, f in enumerate(firms)}

        # --- Region assignment ---
        self.num_regions = max(1, topology_config.num_regions)
        self.firm_region: Dict[str, int] = {}
        if topology_config.region_assignment == "tiered":
            # Assign firms by layer grouping
            for i, f in enumerate(firms):
                self.firm_region[f] = i % self.num_regions
        elif topology_config.region_assignment == "clustered":
            # Cluster nearby firms (simple hash-based for reproducibility)
            for i, f in enumerate(firms):
                self.firm_region[f] = rng.integers(0, self.num_regions)
        else:  # "random"
            for f in firms:
                self.firm_region[f] = int(rng.integers(0, self.num_regions))

        # Region -> firms
        self.region_firms: Dict[int, List[str]] = {r: [] for r in range(self.num_regions)}
        for f, r in self.firm_region.items():
            self.region_firms[r].append(f)

        # --- Chokepoints ---
        # Identify inner-layer firms with high connectivity
        self.is_chokepoint: Dict[str, bool] = {}
        firm_connectivity = []
        for f in firms:
            n_prods = len(firm2prods.get(f, []))
            firm_connectivity.append((f, n_prods))
        firm_connectivity.sort(key=lambda x: -x[1])
        n_chokepoints = max(1, int(len(firms) * topology_config.chokepoint_fraction))
        chokepoint_set = set(f for f, _ in firm_connectivity[:n_chokepoints])
        for f in firms:
            self.is_chokepoint[f] = f in chokepoint_set

        # --- Supplier capacities ---
        cv = topology_config.supplier_capacity_cv
        self.supplier_capacity: Dict[str, float] = {}
        if cv > 0:
            for f in firms:
                # Lognormal with mean=1, cv=cv
                sigma = np.sqrt(np.log(1 + cv ** 2))
                mu = -sigma ** 2 / 2
                cap_mult = float(rng.lognormal(mu, sigma))
                self.supplier_capacity[f] = max(0.1, cap_mult)
        else:
            for f in firms:
                self.supplier_capacity[f] = 1.0

        # --- Topological neighbors (for contagion) ---
        # Build a firm-level adjacency: two firms are neighbors if they supply
        # the same product or trade with each other
        self.firm_neighbors: Dict[str, Set[str]] = {f: set() for f in firms}
        for p, p_firms in prod2firms.items():
            for i, f1 in enumerate(p_firms):
                for f2 in p_firms[i + 1:]:
                    self.firm_neighbors[f1].add(f2)
                    self.firm_neighbors[f2].add(f1)
        # Also add supplier-buyer edges
        for (buyer, product), supplier in inputs2supplier.items():
            if buyer in self.firm_neighbors and supplier in self.firm_neighbors:
                self.firm_neighbors[buyer].add(supplier)
                self.firm_neighbors[supplier].add(buyer)

        # --- Centrality (for epicenter bias) ---
        # Use product count as proxy for node centrality
        self.firm_centrality: Dict[str, float] = {}
        max_prods = max(len(firm2prods.get(f, [])) for f in firms) if firms else 1
        for f in firms:
            self.firm_centrality[f] = len(firm2prods.get(f, [])) / max(max_prods, 1)

    def get_exog_firms(self) -> List[Tuple[str, str]]:
        """Return all (firm, product) keys for exogenous products."""
        result = []
        for p in self.exog_prods:
            for f in self.prod2firms[p]:
                result.append((f, p))
        return result

    def firms_in_region(self, region_id: int) -> List[str]:
        return self.region_firms.get(region_id, [])

    def product_depth(self, product: str) -> int:
        """Topological depth from exog sources."""
        try:
            preds = list(self._nx_graph.predecessors(product))
            if not preds:
                return 0
            return 1 + max(self.product_depth(p) for p in preds)
        except nx.NetworkXError:
            return 0


# ---------------------------------------------------------------------------
# ShockState — runtime tracking of a single shock event
# ---------------------------------------------------------------------------

@dataclass
class ShockState:
    shock_id: int
    event_type: str  # "localized" | "regional" | "cascade"
    epicenter_firm: str
    epicenter_product: str
    affected_firms: List[str]
    affected_products: List[str]
    magnitude: float  # fraction of supply removed
    onset_step: int
    expected_duration: int
    contagion_front: Set[str]  # firms at the spreading edge
    recovering: bool = False
    recovery_step_start: int = 0
    ended: bool = False

    def to_dict(self) -> dict:
        return {
            "shock_id": self.shock_id,
            "event_type": self.event_type,
            "epicenter_firm": self.epicenter_firm,
            "epicenter_product": self.epicenter_product,
            "affected_firms": list(self.affected_firms),
            "affected_products": list(self.affected_products),
            "magnitude": self.magnitude,
            "onset_step": self.onset_step,
            "expected_duration": self.expected_duration,
            "recovering": self.recovering,
            "recovery_step_start": self.recovery_step_start,
            "ended": self.ended,
        }


# ---------------------------------------------------------------------------
# ShockEngine — manages shock lifecycle
# ---------------------------------------------------------------------------

class ShockEngine:
    """
    Manages shock lifecycle each timestep:
    1. Age existing shocks — transition to recovering when duration expires
    2. Apply contagion — spread active shocks to neighbors
    3. Sample new shock events
    4. Compute supply_multipliers from all active shocks
    """

    def __init__(
        self,
        gen_config: ShockGenerationConfig,
        dyn_config: ShockDynamicsConfig,
        topology: SupplyGraph,
        rng: np.random.Generator,
    ):
        self.gen_config = gen_config
        self.dyn_config = dyn_config
        self.topology = topology
        self.rng = rng

        self.active_shocks: List[ShockState] = []
        self._next_shock_id = 0
        self._warmup_steps = 0  # set externally if needed

        # Pre-generate shock event sequence for reproducibility
        # (policies cannot influence shock generation)
        self._pregenerated = False

    def set_warmup_steps(self, n: int):
        self._warmup_steps = n

    def _sample_magnitude(self) -> float:
        cfg = self.gen_config
        if cfg.magnitude_dist == "fixed":
            return cfg.magnitude_mean
        elif cfg.magnitude_dist == "uniform":
            spread = cfg.magnitude_cv * cfg.magnitude_mean
            lo = max(0.05, cfg.magnitude_mean - spread)
            hi = min(0.99, cfg.magnitude_mean + spread)
            return float(self.rng.uniform(lo, hi))
        else:  # "beta"
            mean = cfg.magnitude_mean
            cv = cfg.magnitude_cv
            var = (cv * mean) ** 2
            var = min(var, mean * (1 - mean) - 1e-6)
            if var <= 0:
                return mean
            alpha = mean * (mean * (1 - mean) / var - 1)
            beta_param = (1 - mean) * (mean * (1 - mean) / var - 1)
            alpha = max(0.5, alpha)
            beta_param = max(0.5, beta_param)
            return float(np.clip(self.rng.beta(alpha, beta_param), 0.05, 0.99))

    def _sample_duration(self) -> int:
        cfg = self.dyn_config
        if cfg.duration_dist == "geometric":
            # Geometric with mean = duration_mean
            p = 1.0 / max(1.0, cfg.duration_mean)
            dur = int(self.rng.geometric(p))
            return max(1, min(dur, int(cfg.duration_mean * 4)))
        else:
            return max(1, int(cfg.duration_mean))

    def _select_event_type(self) -> str:
        probs = self.gen_config.event_type_probs
        types = list(probs.keys())
        p = np.array([probs[t] for t in types], dtype=float)
        p /= p.sum()
        return str(self.rng.choice(types, p=p))

    def _select_epicenter(self) -> Tuple[str, str]:
        """Select (firm, product) as shock epicenter."""
        topo = self.topology
        exog_keys = topo.get_exog_firms()
        if not exog_keys:
            # Fallback: any firm/product
            f = self.rng.choice(topo.firms)
            prods = topo.firm2prods.get(f, [])
            p = self.rng.choice(prods) if prods else topo.products[0]
            return f, p

        bias = self.gen_config.epicenter_bias
        if bias == "centrality":
            # Weight by centrality
            weights = np.array([topo.firm_centrality.get(f, 0.1) for f, p in exog_keys])
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(exog_keys)) / len(exog_keys)
            idx = int(self.rng.choice(len(exog_keys), p=weights))
        elif bias == "region":
            # Pick a random region, then a random exog key within it
            region = int(self.rng.integers(0, topo.num_regions))
            region_firms_set = set(topo.firms_in_region(region))
            region_exog = [(f, p) for f, p in exog_keys if f in region_firms_set]
            if not region_exog:
                region_exog = exog_keys
            idx_in_region = int(self.rng.integers(0, len(region_exog)))
            return region_exog[idx_in_region]
        else:  # "uniform"
            idx = int(self.rng.integers(0, len(exog_keys)))
        return exog_keys[idx]

    def _create_shock(self, t: int) -> ShockState:
        event_type = self._select_event_type()
        epicenter_firm, epicenter_product = self._select_epicenter()
        magnitude = self._sample_magnitude()
        duration = self._sample_duration()
        topo = self.topology

        # Shocks must be ASYMMETRIC: a subset of firms are hit,
        # leaving others as viable rerouting targets.
        affected_products = [epicenter_product]

        if event_type == "localized":
            # Hit epicenter + fraction of other firms for this product.
            # Keeps at least 1 firm unaffected as a reroute target.
            product_firms = list(topo.prod2firms.get(epicenter_product, []))
            frac = self.gen_config.localized_firm_fraction
            n_to_affect = max(1, int(len(product_firms) * frac))
            # Always keep at least 1 firm unaffected for rerouting
            n_to_affect = min(n_to_affect, max(1, len(product_firms) - 1))

            other_firms = [f for f in product_firms if f != epicenter_firm]
            n_additional = n_to_affect - 1  # already have epicenter

            if n_additional > 0 and other_firms:
                # Prefer same-region firms (geographic clustering of disruption)
                region = topo.firm_region.get(epicenter_firm, 0)
                same_region = [f for f in other_firms if topo.firm_region.get(f, -1) == region]
                diff_region = [f for f in other_firms if f not in same_region]
                pool = same_region + diff_region
                affected_firms = [epicenter_firm] + pool[:n_additional]
            else:
                affected_firms = [epicenter_firm]
            contagion_front = set(affected_firms)

        elif event_type == "regional":
            # Hit ALL firms in the epicenter's region across all their exog products.
            # Firms in OTHER regions are unaffected — rerouting to them is valuable.
            region = topo.firm_region.get(epicenter_firm, 0)
            region_firms = topo.firms_in_region(region)
            affected_firms = list(region_firms)
            # Collect all exog products supplied by firms in this region
            for f in region_firms:
                for p in topo.exog_prods:
                    if f in topo.prod2firms.get(p, []) and p not in affected_products:
                        affected_products.append(p)
            contagion_front = set(affected_firms)

        elif event_type == "cascade":
            # Start at epicenter + nearby firms, contagion will spread aggressively
            product_firms = list(topo.prod2firms.get(epicenter_product, []))
            frac = self.gen_config.localized_firm_fraction
            n_to_affect = max(1, int(len(product_firms) * frac))
            n_to_affect = min(n_to_affect, max(1, len(product_firms) - 1))

            other_firms = [f for f in product_firms if f != epicenter_firm]
            n_additional = n_to_affect - 1

            if n_additional > 0 and other_firms:
                region = topo.firm_region.get(epicenter_firm, 0)
                same_region = [f for f in other_firms if topo.firm_region.get(f, -1) == region]
                diff_region = [f for f in other_firms if f not in same_region]
                pool = same_region + diff_region
                affected_firms = [epicenter_firm] + pool[:n_additional]
            else:
                affected_firms = [epicenter_firm]
            contagion_front = set(affected_firms)

        else:
            affected_firms = [epicenter_firm]
            contagion_front = {epicenter_firm}

        shock = ShockState(
            shock_id=self._next_shock_id,
            event_type=event_type,
            epicenter_firm=epicenter_firm,
            epicenter_product=epicenter_product,
            affected_firms=affected_firms,
            affected_products=affected_products,
            magnitude=magnitude,
            onset_step=t,
            expected_duration=duration,
            contagion_front=contagion_front,
        )
        self._next_shock_id += 1
        return shock

    def _apply_contagion(self, shock: ShockState, t: int):
        """
        Spread shock to neighboring products in the product DAG.
        When a new product is affected, ALL its suppliers are added.
        """
        dyn = self.dyn_config
        if dyn.contagion_radius <= 0 or dyn.contagion_prob_per_hop <= 0:
            return

        topo = self.topology
        if shock.recovering or shock.ended:
            return

        steps_since_onset = t - shock.onset_step
        if steps_since_onset > dyn.contagion_radius:
            return

        # Spread at the product level: for each affected product,
        # check neighboring products in the DAG
        affected_products_set = set(shock.affected_products)
        new_products = []

        for p in list(shock.affected_products):
            # Get downstream products (successors in the product DAG)
            if p in topo._nx_graph:
                for neighbor_p in topo._nx_graph.successors(p):
                    if neighbor_p not in affected_products_set:
                        if self.rng.random() < dyn.contagion_prob_per_hop:
                            new_products.append(neighbor_p)
                            affected_products_set.add(neighbor_p)

        # Add new products but only firms in AFFECTED REGIONS (asymmetric contagion).
        # This preserves rerouting value: firms in unaffected regions remain viable.
        affected_firms_set = set(shock.affected_firms)
        affected_regions = set(topo.firm_region.get(f, 0) for f in shock.affected_firms)
        for p in new_products:
            shock.affected_products.append(p)
            for f in topo.prod2firms.get(p, []):
                if f not in affected_firms_set and topo.firm_region.get(f, -1) in affected_regions:
                    shock.affected_firms.append(f)
                    affected_firms_set.add(f)

        # Update contagion front to the newly affected firms
        shock.contagion_front = affected_firms_set - set(shock.contagion_front)

    def _compute_recovery_multiplier(self, shock: ShockState, t: int) -> float:
        """
        Compute the supply multiplier for a recovering shock.
        Returns fraction of supply RETAINED (1 - magnitude adjusted for recovery).
        """
        dyn = self.dyn_config
        if not shock.recovering:
            # Active shock: apply full magnitude
            return 1.0 - shock.magnitude

        steps_into_recovery = t - shock.recovery_step_start
        total_recovery = dyn.recovery_steps

        if total_recovery <= 0 or dyn.recovery_shape == "instant":
            return 1.0  # fully recovered

        progress = min(1.0, steps_into_recovery / max(1, total_recovery))

        if dyn.recovery_shape == "concave":
            # Fast initial recovery, slow tail: sqrt curve
            recovery_frac = math.sqrt(progress)
        else:  # "linear"
            recovery_frac = progress

        # Supply retained = (1 - magnitude) + magnitude * recovery_fraction
        return (1.0 - shock.magnitude) + shock.magnitude * recovery_frac

    def step(self, t: int) -> Tuple[Dict[Tuple[str, str], float], List[dict]]:
        """
        Advance shock state by one timestep.

        Returns:
            supply_multipliers: {(firm, product): multiplier} for all affected nodes
            events: list of event dicts (for logging)
        """
        events = []
        topo = self.topology

        # --- 1. Age existing shocks ---
        for shock in self.active_shocks:
            if shock.ended:
                continue
            age = t - shock.onset_step
            if not shock.recovering and age >= shock.expected_duration:
                shock.recovering = True
                shock.recovery_step_start = t
                events.append({
                    "type": "shock_recovery_start",
                    "shock_id": shock.shock_id,
                    "t": t,
                })

            # Check if recovery is complete
            if shock.recovering:
                dyn = self.dyn_config
                if dyn.recovery_shape == "instant":
                    shock.ended = True
                    events.append({
                        "type": "shock_ended",
                        "shock_id": shock.shock_id,
                        "t": t,
                    })
                elif (t - shock.recovery_step_start) >= dyn.recovery_steps:
                    shock.ended = True
                    events.append({
                        "type": "shock_ended",
                        "shock_id": shock.shock_id,
                        "t": t,
                    })

        # Remove ended shocks
        self.active_shocks = [s for s in self.active_shocks if not s.ended]

        # --- 2. Apply contagion ---
        for shock in self.active_shocks:
            if not shock.recovering:
                self._apply_contagion(shock, t)

        # --- 3. Sample new shock events ---
        if t >= self._warmup_steps:
            n_active = len([s for s in self.active_shocks if not s.recovering])
            can_add = n_active < self.gen_config.max_concurrent_shocks

            if can_add and self.rng.random() < self.gen_config.shock_prob:
                # Check overlap constraint
                if n_active == 0 or self.rng.random() < self.gen_config.shock_overlap_prob:
                    new_shock = self._create_shock(t)
                    self.active_shocks.append(new_shock)
                    events.append({
                        "type": "shock_onset",
                        "shock_id": new_shock.shock_id,
                        "event_type": new_shock.event_type,
                        "epicenter_firm": new_shock.epicenter_firm,
                        "epicenter_product": new_shock.epicenter_product,
                        "magnitude": new_shock.magnitude,
                        "duration": new_shock.expected_duration,
                        "affected_firms": len(new_shock.affected_firms),
                        "t": t,
                    })

        # --- 4. Compute supply multipliers ---
        supply_multipliers: Dict[Tuple[str, str], float] = {}
        for shock in self.active_shocks:
            mult = self._compute_recovery_multiplier(shock, t)
            for firm in shock.affected_firms:
                for product in shock.affected_products:
                    if firm in topo.prod2firms.get(product, []):
                        key = (firm, product)
                        if key in supply_multipliers:
                            # Overlapping shocks compound multiplicatively
                            supply_multipliers[key] *= mult
                        else:
                            supply_multipliers[key] = mult

        # Clamp multipliers
        for key in supply_multipliers:
            supply_multipliers[key] = max(0.0, min(1.0, supply_multipliers[key]))

        return supply_multipliers, events

    def get_active_shock_states(self) -> List[dict]:
        return [s.to_dict() for s in self.active_shocks]

    def get_demand_surge_multiplier(self, t: int) -> float:
        """Return demand multiplier if surge is active."""
        dyn = self.dyn_config
        if not dyn.demand_surge_enabled:
            return 1.0
        # Check if any shock is active and past the lag period
        for shock in self.active_shocks:
            if not shock.ended and (t - shock.onset_step) >= dyn.demand_surge_lag:
                return dyn.demand_surge_multiplier
        return 1.0


# ---------------------------------------------------------------------------
# PolicyInterfaceLayer
# ---------------------------------------------------------------------------

class PolicyInterfaceLayer:
    """
    Wraps env step to enforce rerouting constraints and nonlinear costs.
    """

    def __init__(self, config: PolicyInterfaceConfig, topology: SupplyGraph):
        self.config = config
        self.topology = topology

        # Track reroute setup delays: (buyer, product, new_supplier) -> step_initiated
        self._reroute_pending: Dict[Tuple[str, str, str], int] = {}
        # Track per-supplier load this step
        self._supplier_load: Dict[str, float] = {}

    def reset(self):
        self._reroute_pending = {}
        self._supplier_load = {}

    def apply_reroute_constraints(
        self,
        proposed_reroutes: List[Tuple[str, str, str]],
        t: int,
    ) -> List[Tuple[str, str, str]]:
        """
        Filter reroutes that exceed supplier capacity fraction or are still in setup delay.
        Returns filtered list of approved reroutes.
        """
        cfg = self.config
        topo = self.topology
        approved = []

        # Reset supplier load tracking for this step
        self._supplier_load = {}

        for buyer, product, new_supplier in proposed_reroutes:
            # Check setup delay
            if cfg.reroute_setup_delay > 0:
                key = (buyer, product, new_supplier)
                if key in self._reroute_pending:
                    if (t - self._reroute_pending[key]) < cfg.reroute_setup_delay:
                        continue  # Still in setup period
                    else:
                        # Setup complete, allow the reroute
                        del self._reroute_pending[key]
                else:
                    # Initiate setup delay
                    self._reroute_pending[key] = t
                    continue  # Can't reroute yet

            # Check capacity fraction
            if cfg.reroute_capacity_fraction < 1.0:
                cap = topo.supplier_capacity.get(new_supplier, 1.0)
                current_load = self._supplier_load.get(new_supplier, 0.0)
                if current_load >= cap * cfg.reroute_capacity_fraction:
                    continue  # Supplier at capacity
                self._supplier_load[new_supplier] = current_load + 1.0

            approved.append((buyer, product, new_supplier))

        return approved

    def compute_expedite_cost(
        self,
        units_requested: float,
        unit_cost: float,
        current_step_spend: float,
    ) -> Tuple[float, float]:
        """
        Apply convex cost model.
        Returns (actual_units_granted, total_cost).
        """
        cfg = self.config

        # Apply capacity cap
        max_units = units_requested
        if cfg.expedite_capacity_cap < 1.0:
            max_units = min(units_requested, units_requested * cfg.expedite_capacity_cap)

        if cfg.expedite_cost_model == "linear" or cfg.expedite_convexity <= 0:
            cost = max_units * unit_cost
            return max_units, cost

        elif cfg.expedite_cost_model == "convex":
            # Convex cost: cost = unit_cost * units^(1 + convexity)
            conv = cfg.expedite_convexity
            cost = unit_cost * (max_units ** (1.0 + conv))
            return max_units, cost

        elif cfg.expedite_cost_model == "step":
            # Step function: first 50% at normal cost, rest at 3x
            half = max_units / 2.0
            if max_units <= half:
                cost = max_units * unit_cost
            else:
                cost = half * unit_cost + (max_units - half) * unit_cost * 3.0
            return max_units, cost

        # Default: linear
        return max_units, max_units * unit_cost

    def observe(
        self,
        true_obs: dict,
        t: int,
        obs_history: Optional[List[dict]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Apply observation delay or noise. Return policy-visible state."""
        cfg = self.config

        if cfg.observability == "full":
            return true_obs

        if cfg.observability == "delayed" and obs_history is not None:
            delay = cfg.observation_delay
            if delay > 0 and len(obs_history) > delay:
                return obs_history[-delay]
            elif delay > 0:
                # Not enough history; return earliest available
                return obs_history[0] if obs_history else true_obs
            return true_obs

        if cfg.observability == "noisy" and rng is not None:
            noisy_obs = copy.deepcopy(true_obs)
            cv = cfg.observation_noise_cv
            if cv > 0 and "inventories" in noisy_obs:
                inv = noisy_obs["inventories"]
                noise = rng.normal(1.0, cv, size=inv.shape)
                noisy_obs["inventories"] = np.clip(inv * noise, 0, None)
            return noisy_obs

        return true_obs
