#!/usr/bin/env python3
"""
Synthetic Ops Data Generator — MONTHLY (3 pairs)
------------------------------------------------
Generates realistic, *monthly-level* synthetic data for exactly THREE
(material, location) pairs and APPENDS newer months to a single CSV.

- One row per (sku_id, location_id, month). `date` is the first day of month.
- Stable per-SKU & per-location parameters via global seed.
- Seasonality + trend + inflation (evaluated at mid-month).
- Idempotent: if file exists, generation starts at (last_month + 1).

Output columns (unchanged schema for downstream compatibility)
--------------------------------------------------------------
date, sku_id, location_id, unit_of_measure, avg_daily_demand, std_daily_demand,
avg_lead_time_days, std_lead_time_days, service_level_target, unit_cost,
holding_cost_rate_annual, order_cost_fixed, moq_units, lot_size_units,
review_period_days, holding_cost_per_unit_per_day
"""

from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import math
import os
from datetime import datetime, timedelta, date
from typing import List, Tuple, Dict, Optional

# ---- Robust imports with environment diagnostics ---------------------------------
def _import_np_pd_or_die():
    try:
        import numpy as _np  # noqa
    except Exception as e:
        raise SystemExit(
            "Failed to import numpy. Install deps first: pip install -r requirements.txt\n"
            f"Original error: {e}"
        )
    try:
        import pandas as _pd  # noqa
    except Exception as e:
        raise SystemExit(
            "Failed to import pandas. Install deps first: pip install -r requirements.txt\n"
            f"Original error: {e}"
        )

    # Detect shadowing: if 'pandas' module lacks DataFrame, or path looks local
    import importlib
    pd = importlib.import_module("pandas")
    pd_file = getattr(pd, "__file__", "") or ""
    has_df = hasattr(pd, "DataFrame")

    suspicious = False
    msg = []
    if not has_df:
        suspicious = True
        msg.append("pandas module has no attribute 'DataFrame' (likely shadowed).")
    if pd_file:
        # If importing from current working directory or project path, warn
        cwd = os.path.abspath(os.getcwd())
        if os.path.abspath(pd_file).startswith(cwd):
            suspicious = True
            msg.append(f"pandas is being imported from: {pd_file}")

    if suspicious:
        raise SystemExit(
            "Environment issue detected: Python is importing a local 'pandas' instead of the library.\n"
            + "\n".join(f"- {m}" for m in msg) +
            "\n\nFix:\n"
            "1) Search your repo for files/folders named 'pandas' and rename/delete them (e.g., 'pandas.py', 'pandas/').\n"
            "2) Remove any leftover __pycache__ next to them.\n"
            "3) Reinstall the real library: pip install --upgrade --force-reinstall pandas\n"
            "   (tip) Verify with: python -c \"import pandas,inspect; print(pandas.__file__)\"\n"
        )
    return _np, pd

np, pd = _import_np_pd_or_die()
# ----------------------------------------------------------------------------------

# ---------------------------
# Utility: stable RNG per key
# ---------------------------
def _stable_uint64(key: str, seed: int) -> int:
    s = f"{seed}:{key}".encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:8], "big", signed=False)

def _rng_for(key: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(_stable_uint64(key, seed))

# ---------------------------
# ID helpers
# ---------------------------
def make_ids(prefix: str, n: int, width: int = 4) -> List[str]:
    return [f"{prefix}-{i:0{width}d}" for i in range(1, n + 1)]

# ---------------------------
# Parameter generation (stable per SKU / location)
# ---------------------------
def generate_static_params_for_sku(sku_id: str, seed: int) -> Dict:
    rng = _rng_for(f"SKU_PARAMS::{sku_id}", seed)

    uom = "EA" if rng.random() < 0.7 else "case"
    case_pack = int(rng.choice([6, 8, 10, 12, 18, 24])) if uom == "case" else 1

    if uom == "EA":
        base_mu, base_sigma = math.log(20), 0.9
    else:
        base_mu, base_sigma = math.log(1.5), 0.8
    base_demand_level = float(rng.lognormal(mean=base_mu, sigma=base_sigma))

    cv = float(rng.uniform(0.2, 0.6))
    annual_amp = float(rng.uniform(0.10, 0.45))
    annual_phase = float(rng.uniform(0, 2 * math.pi))
    annual_trend = float(rng.uniform(-0.08, 0.20))

    if uom == "EA":
        unit_cost_base = float(rng.lognormal(mean=math.log(8.0), sigma=0.7)) + 0.5
    else:
        unit_cost_base = float(rng.lognormal(mean=math.log(80.0), sigma=0.7)) + 2.0

    holding_rate_annual = float(rng.uniform(0.18, 0.35))
    order_cost_fixed = float(rng.uniform(25, 250))

    if uom == "EA":
        lot_size = int(rng.choice([1, 1, 1, 2, 5]))
        moq = int(lot_size * rng.choice([1, 1, 2, 3, 5]))
    else:
        lot_size = int(rng.choice([case_pack, case_pack, 2 * case_pack]))
        moq = int(lot_size * rng.choice([1, 2, 3]))

    review_period_days = int(rng.choice([1, 7, 7, 14, 28]))
    lt_mean = float(rng.uniform(4, 35))
    lt_std = float(max(0.5, lt_mean * rng.uniform(0.2, 0.5)))
    service_level = float(rng.choice([0.90, 0.95, 0.98, 0.99], p=[0.2, 0.5, 0.25, 0.05]))

    return dict(
        unit_of_measure=uom,
        case_pack=case_pack,
        base_demand_level=base_demand_level,
        cv=cv,
        annual_amp=annual_amp,
        annual_phase=annual_phase,
        annual_trend=annual_trend,
        unit_cost_base=unit_cost_base,
        holding_cost_rate_annual=holding_rate_annual,
        order_cost_fixed=order_cost_fixed,
        moq_units=moq,
        lot_size_units=lot_size,
        review_period_days=review_period_days,
        lt_mean=lt_mean,
        lt_std=lt_std,
        service_level_target=service_level,
    )

def generate_location_factor(location_id: str, seed: int) -> float:
    rng = _rng_for(f"LOC_FACTOR::{location_id}", seed)
    return float(rng.uniform(0.6, 1.6))

# ---------------------------
# Dynamics helpers (monthly)
# ---------------------------
ANCHOR_DATE = date(2020, 1, 1)

def years_since_anchor(d: date) -> float:
    return (d - ANCHOR_DATE).days / 365.0

def annual_seasonal_multiplier(d: date, amp: float, phase: float) -> float:
    doy = d.timetuple().tm_yday
    theta = 2 * math.pi * (doy / 365.0) + phase
    return 1.0 + amp * math.sin(theta)

def inflation_multiplier(d: date, annual_inflation: float = 0.02) -> float:
    y = years_since_anchor(d)
    return float((1.0 + annual_inflation) ** y)

# --- Month helpers ---
def first_of_month(d: date) -> date:
    return date(d.year, d.month, 1)

def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    return date(y, m, 1)

def month_range(start_month: date, end_month: date) -> List[date]:
    """Inclusive list of month start dates."""
    cur = first_of_month(start_month)
    out = []
    while cur <= end_month:
        out.append(cur)
        cur = add_months(cur, 1)
    return out

# ---------------------------
# Pairs selection (exactly 3)
# ---------------------------
Pair = Tuple[str, str]  # (sku_id, location_id)

def determine_three_pairs(out_file: str, explicit_pairs: Optional[List[Pair]] = None) -> List[Pair]:
    """If file exists, reuse the first 3 pairs found there. Else use explicit or default new."""
    if os.path.exists(out_file):
        try:
            df_small = pd.read_csv(out_file, usecols=["sku_id", "location_id"]).drop_duplicates()
            pairs = [(r.sku_id, r.location_id) for r in df_small.itertuples(index=False)]
            if pairs:
                return pairs[:3]
        except Exception:
            pass
    if explicit_pairs:
        return explicit_pairs[:3]
    # default new
    skus = make_ids("SKU", 3)
    locs = make_ids("LOC", 3)
    return list(zip(skus, locs))  # [('SKU-0001','LOC-0001'), ('SKU-0002','LOC-0002'), ('SKU-0003','LOC-0003')]

# ---------------------------
# Row assembly (MONTHLY, 3 pairs)
# ---------------------------
def monthly_row_for(
    sku_id: str,
    location_id: str,
    month_start: date,
    sku_params: Dict,
    loc_factor: float,
    seed: int,
    annual_inflation_rate: float,
) -> Dict:
    mid_month = month_start + timedelta(days=14)  # approx midpoint for seasonality/inflation/trend
    seasonal = annual_seasonal_multiplier(mid_month, sku_params["annual_amp"], sku_params["annual_phase"])
    trend_mult = (1.0 + sku_params["annual_trend"]) ** years_since_anchor(mid_month)

    # Mean/Std are "average daily" rates for the month (consistent with original schema)
    mean_daily = sku_params["base_demand_level"] * loc_factor * seasonal * trend_mult
    std_daily = max(0.01, mean_daily * sku_params["cv"])

    nominal_cost = sku_params["unit_cost_base"] * inflation_multiplier(mid_month, annual_inflation_rate)
    unit_cost = nominal_cost  # promos/weekday removed at monthly level

    holding_rate = sku_params["holding_cost_rate_annual"]
    holding_per_day = unit_cost * holding_rate / 365.0

    return dict(
        date=month_start.isoformat(),  # first day of month
        sku_id=sku_id,
        location_id=location_id,
        unit_of_measure=sku_params["unit_of_measure"],
        avg_daily_demand=round(mean_daily, 4),
        std_daily_demand=round(std_daily, 4),
        avg_lead_time_days=round(sku_params["lt_mean"], 3),
        std_lead_time_days=round(sku_params["lt_std"], 3),
        service_level_target=sku_params["service_level_target"],
        unit_cost=round(unit_cost, 4),
        holding_cost_rate_annual=holding_rate,
        order_cost_fixed=round(sku_params["order_cost_fixed"], 2),
        moq_units=int(sku_params["moq_units"]),
        lot_size_units=int(sku_params["lot_size_units"]),
        review_period_days=int(sku_params["review_period_days"]),
        holding_cost_per_unit_per_day=round(holding_per_day, 6),
    )

def generate_monthly_frame(
    months: List[date],
    pairs: List[Pair],
    seed: int,
    annual_inflation_rate: float,
) -> pd.DataFrame:
    skus = sorted({sku for sku, _ in pairs})
    locs = sorted({loc for _, loc in pairs})

    sku_params = {sku: generate_static_params_for_sku(sku, seed) for sku in skus}
    loc_factors = {loc: generate_location_factor(loc, seed) for loc in locs}

    rows: List[Dict] = []
    for m in months:
        for sku, loc in pairs:
            rows.append(
                monthly_row_for(
                    sku, loc, m, sku_params[sku], loc_factors[loc],
                    seed=seed, annual_inflation_rate=annual_inflation_rate
                )
            )
    return pd.DataFrame.from_records(rows)

# ---------------------------
# Append helpers (monthly)
# ---------------------------
def _fast_last_date_from_csv(path: str) -> Optional[date]:
    """Read last non-empty 'date' quickly; expects ISO format YYYY-MM-DD."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            buf = b""
            while pos > 0:
                step = min(2048, pos)
                pos -= step
                f.seek(pos)
                buf = f.read(step) + buf
                if buf.count(b"\n") >= 5 or pos == 0:
                    break
        for line in reversed(buf.decode("utf-8", errors="ignore").splitlines()):
            if not line.strip() or line.startswith("date,"):
                continue
            first = line.split(",", 1)[0]
            try:
                return datetime.strptime(first, "%Y-%m-%d").date()
            except Exception:
                continue
    except Exception:
        pass
    # Fallback
    try:
        s = pd.read_csv(path, usecols=["date"])["date"]
        return pd.to_datetime(s).dt.date.max()
    except Exception:
        return None

def get_append_start_month(out_file: str, fallback_start_month: date) -> date:
    last = _fast_last_date_from_csv(out_file)
    if last is None:
        return first_of_month(fallback_start_month)
    return add_months(first_of_month(last), 1)

# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate MONTHLY synthetic ops data for THREE (sku,location) pairs.")
    default_out = Path(__file__).parent / "output_data" / "synthetic_ops_data_monthly.csv"
    p.add_argument("--out-file", default=str(default_out), help="Output CSV path.")
    p.add_argument("--start-month", default="2025-01", help="Start month YYYY-MM (used if out-file does not exist).")
    p.add_argument("--months", type=int, default=12, help="Number of months to generate (ignored if --end-month is given).")
    p.add_argument("--end-month", default=None, help="Optional inclusive end month YYYY-MM.")
    p.add_argument("--seed", type=int, default=42, help="Global seed for stable parameters.")
    p.add_argument("--annual-inflation-rate", type=float, default=0.02, help="Annual inflation applied to unit_cost.")
    p.add_argument(
        "--pairs",
        default=None,
        help="Optional explicit three pairs: 'SKU-0001@LOC-0001,SKU-0002@LOC-0002,SKU-0003@LOC-0003'"
    )
    return p.parse_args()

def _parse_month(s: str) -> date:
    try:
        dt = datetime.strptime(s, "%Y-%m")
        return date(dt.year, dt.month, 1)
    except ValueError:
        raise SystemExit(f"Invalid month '{s}'. Expected YYYY-MM.")

def _parse_pairs(s: Optional[str]) -> Optional[List[Pair]]:
    if not s:
        return None
    pairs: List[Pair] = []
    for token in s.split(","):
        token = token.strip()
        if not token: continue
        if "@" not in token:
            raise SystemExit(f"Invalid pair '{token}'. Use format SKU@LOC.")
        sku, loc = token.split("@", 1)
        pairs.append((sku.strip(), loc.strip()))
    if len(pairs) < 1:
        return None
    return pairs[:3]

def main():
    args = parse_args()
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)

    fallback_start = _parse_month(args.start_month)
    start_month = get_append_start_month(args.out_file, fallback_start)

    if args.end_month:
        end_month = _parse_month(args.end_month)
    else:
        end_month = add_months(start_month, args.months - 1)

    if end_month < start_month:
        print("Nothing to do: end_month precedes start_month.")
        return

    months = month_range(start_month, end_month)
    explicit_pairs = _parse_pairs(args.pairs)
    pairs = determine_three_pairs(args.out_file, explicit_pairs=explicit_pairs)

    df = generate_monthly_frame(
        months=months,
        pairs=pairs,
        seed=args.seed,
        annual_inflation_rate=args.annual_inflation_rate,
    )

    header = not os.path.exists(args.out_file)
    df.to_csv(args.out_file, mode="a", index=False, header=header, encoding="utf-8")
    print(f"Wrote {len(df):,} rows ({len(months)} months × {len(pairs)} pairs) → {args.out_file}")
    print("Pairs used:", ", ".join(f"{s}@{l}" for s,l in pairs))

if __name__ == "__main__":
    main()