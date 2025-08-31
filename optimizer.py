#!/usr/bin/env python3
"""
Lean ROP & Safety Stock Optimizer (Pyomo + NEOS)
-------------------------------------------------
Computes optimal (cost-minimizing) Safety Stock (SS) just sufficient to meet a
cycle service level target, then sets ROP = mean demand over protection period + SS.

• Protection period = lead time + review period (periodic review); if review_period_days
  is 0 or missing, this reduces to continuous review (s, Q).
• Demand during protection period is approximated Normal with:
    mu_P  = avg_daily_demand * (avg_lead_time_days + review_period_days)
    sig_P = sqrt( (avg_lead_time_days + review_period_days) * std_daily_demand^2
                  + (avg_daily_demand^2) * (std_lead_time_days^2) )
• Safety stock enforces cycle service level (CSL) >= service_level_target via
    SS >= z * sig_P, where z = Phi^{-1}(service_level_target)
• Objective is to minimize total expected daily holding cost of the safety stock:
    Minimize sum_i h_i * SS_i
  where h_i is per-unit-per-day holding cost. You can provide either
  `holding_cost_per_unit_per_day` or `holding_cost_rate_annual` (fraction of unit_cost per year).

This uses Pyomo and submits the tiny LP to the free NEOS server (no local solver needed).
It will fall back to the closed-form solution if NEOS is unreachable.

Input: a CSV with at least these columns (names must match):
    sku_id
    location_id
    unit_of_measure                (e.g., EA, case)              [not used in math]
    avg_daily_demand               (units/day)
    std_daily_demand               (units/day)
    avg_lead_time_days
    std_lead_time_days
    service_level_target           (e.g., 0.95)
    unit_cost                      (per unit)
    holding_cost_rate_annual       (fraction of unit_cost per YEAR)  # optional if you give next
    holding_cost_per_unit_per_day  (absolute per-day)                # optional if you gave above
    order_cost_fixed               (per PO/line)                 [not used in this lean model]
    moq_units                      (minimum order)               [not used in this lean model]
    lot_size_units                 (pack multiple; 1 if none)    [used only for optional rounding]
    review_period_days             (0 for continuous review)

Output: a CSV mirroring input plus these columns:
    protection_days, z_value, mu_protection, sigma_protection,
    safety_stock_units, rop_units,
    safety_stock_units_rounded, rop_units_rounded,
    holding_cost_per_unit_per_day, daily_holding_cost_safety_stock

Usage (Mac / Unix):
    pip install pyomo pandas
    export NEOS_EMAIL="you@example.com"   # NEOS requires an email

    python optimize_rop_safety_stock.py input.csv \
        --out results.csv \
        --round-to-lot  # optional: round SS & ROP up to lot_size_units

Notes:
- Service level must be in (0,1). Values at 0 or 1 are clipped to [1e-6, 1-1e-6].
- If both holding cost fields are present, `holding_cost_per_unit_per_day` wins.
- This script purposely does NOT optimize order quantity Q. If you later want
  to include EOQ or discrete Q with MOQ/lot-size, this file is structured so
  it is straightforward to extend the Pyomo model.
"""
from __future__ import annotations
from pathlib import Path
import os
import sys
import math
import argparse
from typing import Tuple
import pandas as pd
from statistics import NormalDist

# Pyomo imports are done inside try-except so the script can still do closed-form
# if Pyomo isn't present (e.g., quick sanity checks without solver).
try:
    from pyomo.environ import (ConcreteModel, Var, NonNegativeReals, Objective,
                               Constraint, Param, Set, value, summation)
    from pyomo.opt import SolverManagerFactory, TerminationCondition
    PYOMO_AVAILABLE = True
except Exception as e:
    PYOMO_AVAILABLE = False


def _compute_protection_stats(row: pd.Series) -> Tuple[float, float, float, float]:
    """Return (protection_days, z, mu_P, sigma_P) for a record.
    Clips service level to (1e-6, 1-1e-6) to avoid infinities.
    """
    review = 0.0 if pd.isna(row.get("review_period_days", 0.0)) else float(row["review_period_days"])  # days
    lead_mu = float(row["avg_lead_time_days"])                        # days
    lead_sigma = float(row["std_lead_time_days"])                     # days
    d_mu = float(row["avg_daily_demand"])                             # units/day
    d_sigma = float(row["std_daily_demand"])                          # units/day

    protection_days = lead_mu + review

    # Normal approximation for demand during random lead time + deterministic review period
    mu_P = d_mu * protection_days
    var_P = (protection_days) * (d_sigma ** 2) + (d_mu ** 2) * (lead_sigma ** 2)
    sigma_P = math.sqrt(max(var_P, 0.0))

    # Inverse CDF for standard normal
    p = float(row["service_level_target"]) if not pd.isna(row["service_level_target"]) else 0.95
    p = max(min(p, 1 - 1e-6), 1e-6)
    z = NormalDist().inv_cdf(p)

    return protection_days, z, mu_P, sigma_P


def _resolve_holding_cost_per_day(row: pd.Series) -> float:
    """Compute per-unit per-day holding cost from either absolute or annual rate."""
    # Prefer explicit per-day cost if provided
    if "holding_cost_per_unit_per_day" in row and not pd.isna(row["holding_cost_per_unit_per_day"]) and row["holding_cost_per_unit_per_day"] != "":
        return float(row["holding_cost_per_unit_per_day"]).__abs__()
    # Else derive from annual rate (fraction of unit_cost per year)
    if pd.isna(row.get("unit_cost")) or pd.isna(row.get("holding_cost_rate_annual")):
        raise ValueError("Need either holding_cost_per_unit_per_day OR (unit_cost and holding_cost_rate_annual).")
    unit_cost = float(row["unit_cost"])           # currency / unit
    rate_annual = float(row["holding_cost_rate_annual"])  # fraction per year
    return unit_cost * rate_annual / 365.0


def add_computed_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prot_days = []
    z_vals = []
    muP = []
    sigmaP = []
    hpd = []

    for _, r in out.iterrows():
        protection_days, z, mu_P, sigma_P = _compute_protection_stats(r)
        prot_days.append(protection_days)
        z_vals.append(z)
        muP.append(mu_P)
        sigmaP.append(sigma_P)
        hpd.append(_resolve_holding_cost_per_day(r))

    out["protection_days"] = prot_days
    out["z_value"] = z_vals
    out["mu_protection"] = muP
    out["sigma_protection"] = sigmaP
    out["holding_cost_per_unit_per_day"] = hpd
    return out


def solve_with_neos_pyomo(df: pd.DataFrame, round_to_lot: bool = False, verbose: bool = False) -> pd.DataFrame:
    if not PYOMO_AVAILABLE:
        raise RuntimeError("Pyomo not available. Install with: pip install pyomo")

    # Ensure NEOS email is configured (required by NEOS server)
    neos_email = os.getenv("NEOS_EMAIL")
    if not neos_email:
        raise RuntimeError("NEOS_EMAIL environment variable not set. Example: export NEOS_EMAIL=you@example.com")

    # Build a small LP: minimize sum h_i * SS_i; s.t. SS_i >= z_i*sigma_i and ROP_i = mu_i + SS_i
    model = ConcreteModel()
    idx = list(df.index)
    model.I = Set(initialize=idx, ordered=True)

    # Parameters
    model.mu = Param(model.I, initialize=df["mu_protection"].to_dict())
    model.sigma = Param(model.I, initialize=df["sigma_protection"].to_dict())
    model.z = Param(model.I, initialize=df["z_value"].to_dict())
    model.h = Param(model.I, initialize=df["holding_cost_per_unit_per_day"].to_dict())

    # Decision vars
    model.SS = Var(model.I, domain=NonNegativeReals)
    model.ROP = Var(model.I, domain=NonNegativeReals)

    # Constraints
    def c_service_rule(m, i):
        return m.SS[i] >= m.z[i] * m.sigma[i]
    model.C_service = Constraint(model.I, rule=c_service_rule)

    def c_rop_rule(m, i):
        return m.ROP[i] == m.mu[i] + m.SS[i]
    model.C_rop = Constraint(model.I, rule=c_rop_rule)

    # Objective: daily holding cost of safety stock
    def obj_rule(m):
        return sum(m.h[i] * m.SS[i] for i in m.I)
    model.OBJ = Objective(rule=obj_rule)

    # Solve remotely on NEOS using CBC (free MILP solver). This is an LP, but CBC is fine.
    solver_manager = SolverManagerFactory("neos")
    results = solver_manager.solve(model, opt="cbc", tee=verbose)

    term = getattr(results.solver, "termination_condition", None)
    if term is None or term not in (TerminationCondition.optimal, TerminationCondition.locallyOptimal):
        raise RuntimeError(f"NEOS/CBC did not return optimal solution. Termination: {term}")

    # Collect solution
    SS = [float(value(model.SS[i])) for i in model.I]
    ROP = [float(value(model.ROP[i])) for i in model.I]

    out = df.copy()
    out["safety_stock_units"] = SS
    out["rop_units"] = ROP

    # Optional rounding to lot sizes
    if round_to_lot and "lot_size_units" in out.columns:
        def _round_up_to_lot(x, lot):
            try:
                lot = max(1, int(lot))
            except Exception:
                lot = 1
            return int(math.ceil(float(x) / lot) * lot)
        out["safety_stock_units_rounded"] = [ _round_up_to_lot(ss, lot) for ss, lot in zip(out["safety_stock_units"], out["lot_size_units"].fillna(1)) ]
        out["rop_units_rounded"] = [ _round_up_to_lot(rp, lot) for rp, lot in zip(out["rop_units"], out["lot_size_units"].fillna(1)) ]
    else:
        out["safety_stock_units_rounded"] = out["safety_stock_units"].round(0).astype(int)
        out["rop_units_rounded"] = out["rop_units"].round(0).astype(int)

    out["daily_holding_cost_safety_stock"] = out["holding_cost_per_unit_per_day"] * out["safety_stock_units"]
    return out


def closed_form_solution(df: pd.DataFrame, round_to_lot: bool = False) -> pd.DataFrame:
    """Use the direct closed-form (SS = z*sigmaP, ROP = muP + SS) if NEOS is unavailable.
    Still returns the same output columns for convenience.
    """
    out = df.copy()
    out["safety_stock_units"] = out["z_value"] * out["sigma_protection"]
    out["rop_units"] = out["mu_protection"] + out["safety_stock_units"]

    if round_to_lot and "lot_size_units" in out.columns:
        def _round_up_to_lot(x, lot):
            try:
                lot = max(1, int(lot))
            except Exception:
                lot = 1
            return int(math.ceil(float(x) / lot) * lot)
        out["safety_stock_units_rounded"] = [ _round_up_to_lot(ss, lot) for ss, lot in zip(out["safety_stock_units"], out["lot_size_units"].fillna(1)) ]
        out["rop_units_rounded"] = [ _round_up_to_lot(rp, lot) for rp, lot in zip(out["rop_units"], out["lot_size_units"].fillna(1)) ]
    else:
        out["safety_stock_units_rounded"] = out["safety_stock_units"].round(0).astype(int)
        out["rop_units_rounded"] = out["rop_units"].round(0).astype(int)

    out["daily_holding_cost_safety_stock"] = out["holding_cost_per_unit_per_day"] * out["safety_stock_units"]
    return out


def main():
    ap = argparse.ArgumentParser(description="Lean ROP & Safety Stock via Pyomo + NEOS")
    # Defaults to output_data/synthetic_ops_data.csv if not provided
    default_in = Path(__file__).parent / "output_data" / "synthetic_ops_data_monthly.csv"
    default_out = Path(__file__).parent / "output_data" / "opti_results.csv"
    ap.add_argument("input", nargs="?", default=str(default_in), help="Path to input CSV with required columns")
    ap.add_argument("--out", default=str(default_out), help="Output CSV path")
    ap.add_argument("--round-to-lot", action="store_true", help="Round SS/ROP up to lot_size_units (ceil)")
    ap.add_argument("--no-neos", action="store_true", help="Skip NEOS and use closed-form (for testing)")
    ap.add_argument("--verbose", action="store_true", help="Solver output (tee)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # Sanity: fill missing review period with 0 (continuous review)
    if "review_period_days" not in df.columns:
        df["review_period_days"] = 0.0
    df["review_period_days"] = df["review_period_days"].fillna(0.0)

    # Compute derived quantities
    augmented = add_computed_columns(df)

    try:
        if args.no_neos:
            raise RuntimeError("Skipping NEOS as requested")
        result = solve_with_neos_pyomo(augmented, round_to_lot=args.round_to_lot, verbose=args.verbose)
        used_neos = True
    except Exception as e:
        sys.stderr.write(f"[WARN] Falling back to closed-form due to: {e}\n")
        result = closed_form_solution(augmented, round_to_lot=args.round_to_lot)
        used_neos = False

    # Persist (ensure dir exists)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    # CLI summary
    print(("Solved with NEOS/CBC" if used_neos else "Closed-form fallback"))
    print(f"Wrote: {args.out}")

    # Pretty preview (first 8 rows)
    display_cols = [
        "sku_id", "location_id", "unit_of_measure",
        "avg_daily_demand", "std_daily_demand", "avg_lead_time_days", "std_lead_time_days",
        "service_level_target", "review_period_days",
        "mu_protection", "sigma_protection", "z_value",
        "safety_stock_units", "rop_units",
        "safety_stock_units_rounded", "rop_units_rounded",
        "holding_cost_per_unit_per_day", "daily_holding_cost_safety_stock"
    ]
    existing = [c for c in display_cols if c in result.columns]
    print(result[existing].head(8).to_string(index=False))


if __name__ == "__main__":
    main()