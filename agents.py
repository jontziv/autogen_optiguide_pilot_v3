"""
Modern Multi-Agent System for ROP Optimization - Fixed Version
--------------------------------------------------------------
Complete implementation of the OptiGuide agent system with proper
error handling and all required methods.
"""

import json
import subprocess
import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import tempfile
import time,sys
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

try:
    # Modern AutoGen imports (v0.4+)
    from autogen_agentchat.agents import BaseChatAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.messages import TextMessage
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
    AUTOGEN_VERSION = "modern"
except ImportError:
    try:
        # Fallback to older AutoGen (v0.2-0.3)
        from autogen import BaseChatAgent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
        AUTOGEN_AVAILABLE = True
        AUTOGEN_VERSION = "legacy"
        logger.warning("Using legacy AutoGen API")
    except ImportError:
        AUTOGEN_AVAILABLE = False
        AUTOGEN_VERSION = "none"
        logger.error("AutoGen not available. Install with: pip install autogen-agentchat")

from config import safe_config_access

class OptimizationTools:
    """Tools for running data generation and optimization with enhanced error handling"""
    
    @staticmethod
    def load_data() -> pd.DataFrame:
        cfg = safe_config_access()
        data_file = cfg.paths.synthetic_data_file
        if not data_file.exists():
            raise FileNotFoundError(f"No data file found at {data_file}")
        df = pd.read_csv(data_file)
        return df
    
    @staticmethod
    def run_data_generation(months: int = 3, end_month: str | None = None) -> Dict[str, Any]:
        """Generate additional synthetic data with proper error handling"""
        import subprocess, pandas as pd
        cfg = safe_config_access()
        cmd = [
            sys.executable, str(cfg.paths.scripts_dir / "data_synth.py"),
            "--out-file", str(cfg.paths.synthetic_data_file),
        ]
        if end_month:
            cmd += ["--end-month", end_month]   # YYYY-MM
        else:
            cmd += ["--months", str(months)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        df = pd.read_csv(cfg.paths.synthetic_data_file)
        return {
            "success": True,
            "message": f"Generated {months} month(s) of data" if not end_month else f"Generated up to {end_month}",
            "total_rows": len(df),
            "unique_skus": df['sku_id'].nunique(),
            "unique_locations": df['location_id'].nunique(),
            "date_range": f"{df['date'].min()} to {df['date'].max()}",
            "stdout": result.stdout
        }
    @staticmethod
    def apply_scenario_delta(scenario: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply scenario changes to base data with comprehensive validation.

        Supported changes:
        - service_level_target: {default: float (0,1), overrides: {SKU|@LOC|SKU@LOC: float (0,1)}}
        - demand_multiplier: float>0 | {default: float>0, overrides: {SKU|@LOC|SKU@LOC: float>0}}
            * Multipliers are applied against the ORIGINAL baseline values (no compounding).
        - lead_time_days: {multiplier: float>0}  (applies to avg + std lead time)
        - moq_units: {overrides: {SKU|@LOC|SKU@LOC: number>0}}
        - review_period_days: int >= 0
        """
        try:
            scenario = scenario or {}
            df = OptimizationTools.load_data()

            # Work only on the latest month
            latest_date = df['date'].max()
            df_latest = df.loc[df['date'] == latest_date].copy()
            if df_latest.empty:
                raise ValueError("No data found for latest date")

            # Defensive: required identity columns
            for col in ("sku_id", "location_id"):
                if col not in df_latest.columns:
                    raise KeyError(f"Required column '{col}' missing from data")

            changes = scenario.get("changes", {}) or {}

            # Ensure optional columns exist
            for col in ("service_level_target", "moq_units", "review_period_days"):
                if col not in df_latest.columns:
                    df_latest[col] = np.nan

            # -------- helpers --------
            def _mask_for_key(key: str) -> pd.Series:
                """Build a boolean mask for 'SKU', '@LOC', or 'SKU@LOC' keys."""
                sku = loc = None
                if "@" in key:
                    sku, loc = key.split("@", 1)
                elif key.startswith("@"):
                    loc = key[1:]
                else:
                    sku = key
                m = pd.Series(True, index=df_latest.index)
                if sku: m &= (df_latest["sku_id"] == sku)
                if loc: m &= (df_latest["location_id"] == loc)
                return m

            def _count(mask: pd.Series) -> int:
                return int(mask.sum())

            # Keep ORIGINAL baselines for idempotent application (no compounding)
            base_avg_demand = df_latest["avg_daily_demand"].astype(float) if "avg_daily_demand" in df_latest.columns else None
            base_std_demand = df_latest["std_daily_demand"].astype(float) if "std_daily_demand" in df_latest.columns else None
            base_avg_lt = df_latest["avg_lead_time_days"].astype(float) if "avg_lead_time_days" in df_latest.columns else None
            base_std_lt = df_latest["std_lead_time_days"].astype(float) if "std_lead_time_days" in df_latest.columns else None

            # ---------------- 1) Service level changes ----------------
            if "service_level_target" in changes:
                sl_changes = changes["service_level_target"] or {}
                default_sl = sl_changes.get("default", None)
                if isinstance(default_sl, (int, float)):
                    val = float(default_sl)
                    if 0 < val < 1:
                        df_latest["service_level_target"] = val

                overrides = sl_changes.get("overrides") or {}
                sl_rows = 0
                for key, value in overrides.items():
                    if not isinstance(value, (int, float)):
                        continue
                    val = float(value)
                    if not (0 < val < 1):
                        continue
                    mask = _mask_for_key(key)
                    df_latest.loc[mask, "service_level_target"] = val
                    sl_rows += _count(mask)
                logger.info(f"Service level overrides applied to {sl_rows} rows")

            # ---------------- 2) Demand multiplier (global or overrides) ----------------
            if "demand_multiplier" in changes:
                if base_avg_demand is None:
                    raise KeyError("avg_daily_demand column is required for demand multiplier")
                # Build a factor vector against baseline; start with ones
                factor = pd.Series(1.0, index=df_latest.index)

                dm = changes["demand_multiplier"]
                if isinstance(dm, (int, float)):
                    f = float(dm)
                    if f <= 0:
                        raise ValueError("demand_multiplier must be > 0")
                    factor[:] = f
                elif isinstance(dm, dict):
                    default_f = float(dm.get("default", 1.0))
                    if default_f <= 0:
                        raise ValueError("demand_multiplier.default must be > 0")
                    factor[:] = default_f
                    for key, f in (dm.get("overrides") or {}).items():
                        if not isinstance(f, (int, float)) or float(f) <= 0:
                            continue
                        mask = _mask_for_key(key)
                        factor.loc[mask] = float(f)
                else:
                    raise TypeError("demand_multiplier must be a number or a dict with default/overrides")

                # Apply factors to ORIGINAL baseline (idempotent)
                df_latest["avg_daily_demand"] = base_avg_demand * factor
                if base_std_demand is not None:
                    df_latest["std_daily_demand"] = base_std_demand * factor

                # Non-negativity guard (shouldn't be needed, but safe)
                for col in ("avg_daily_demand", "std_daily_demand"):
                    if col in df_latest.columns:
                        df_latest[col] = df_latest[col].clip(lower=0)

                logger.info(f"Demand multiplier applied; distinct factors used: {factor.nunique()}")

            # ---------------- 3) Lead time multiplier ----------------
            if "lead_time_days" in changes:
                if base_avg_lt is None and base_std_lt is None:
                    raise KeyError("Lead time columns not found for lead_time_days change")
                multiplier = float((changes["lead_time_days"] or {}).get("multiplier", 1.0))
                if multiplier <= 0:
                    raise ValueError("lead_time_days.multiplier must be > 0")

                if base_avg_lt is not None:
                    df_latest["avg_lead_time_days"] = base_avg_lt * multiplier
                if base_std_lt is not None:
                    df_latest["std_lead_time_days"] = base_std_lt * multiplier

                for col in ("avg_lead_time_days", "std_lead_time_days"):
                    if col in df_latest.columns:
                        df_latest[col] = df_latest[col].clip(lower=0)

                logger.info("Lead time multiplier applied")

            # ---------------- 4) MOQ overrides ----------------
            if "moq_units" in changes:
                overrides = (changes["moq_units"] or {}).get("overrides") or {}
                moq_rows = 0
                for key, value in overrides.items():
                    try:
                        val = float(value)
                    except Exception:
                        continue
                    if val <= 0:
                        continue
                    mask = _mask_for_key(key)
                    df_latest.loc[mask, "moq_units"] = val
                    moq_rows += _count(mask)
                logger.info(f"MOQ overrides applied to {moq_rows} rows")

            # ---------------- 5) Review period ----------------
            if "review_period_days" in changes:
                try:
                    rp = int(changes["review_period_days"])
                except Exception as e:
                    raise ValueError("review_period_days must be an integer") from e
                df_latest["review_period_days"] = rp if rp > 0 else 0
                logger.info("Review period set")

            logger.info(f"Applied scenario changes to {len(df_latest)} records")
            return df_latest

        except Exception as e:
            logger.error(f"Failed to apply scenario: {e}")
            raise


    @staticmethod
    def run_optimizer(scenario_data: pd.DataFrame, persist: bool = True, timeout: int = 600) -> Dict[str, Any]:
            """
            Run optimizer on scenario data with comprehensive error handling.
            Assumes optimizer.py handles NEOS/closed-form so no local solver dependency.
            """
            try:
                cfg = safe_config_access()

                if scenario_data.empty:
                    raise ValueError("No scenario data provided")

                # input temp
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_in:
                    scenario_data.to_csv(f_in.name, index=False)
                    temp_input = f_in.name

                # output path
                if persist:
                    ts = time.strftime("%Y%m%d-%H%M%S")
                    out_path = cfg.paths.optimization_results_dir / f"opt_results_{ts}.csv"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    temp_output = str(out_path)
                else:
                    f_out = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                    f_out.close()
                    temp_output = f_out.name

                try:
                    cmd = [
                        sys.executable, str(cfg.paths.scripts_dir / "optimizer.py"),
                        temp_input,
                        "--out", temp_output,
                        "--round-to-lot"
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)

                    results_df = pd.read_csv(temp_output)
                    kpis = OptimizationTools._compute_kpis(results_df)

                    payload = {
                        "success": True,
                        "policy": results_df.to_dict('records'),
                        "kpis": kpis,
                        "trace": [result.stdout.strip()] if result.stdout and result.stdout.strip() else []
                    }
                    if persist:
                        payload["results_csv"] = temp_output
                    return payload

                finally:
                    # Clean up only the input temp; keep output if persisted
                    if os.path.exists(temp_input):
                        os.unlink(temp_input)
                    if not persist and os.path.exists(temp_output):
                        os.unlink(temp_output)

            except subprocess.CalledProcessError as e:
                return {
                    "success": False,
                    "message": f"Optimization failed: {e}",
                    "stderr": e.stderr if e.stderr else str(e)
                }
            except Exception as e:
                logger.error(f"Optimizer error: {e}")
                return {"success": False, "message": f"Unexpected error: {e}"}

    @staticmethod
    def _compute_kpis(results_df: pd.DataFrame) -> Dict[str, float]:
        """Compute KPIs from optimization results with error handling"""
        try:
            kpis = {}
            
            if "daily_holding_cost_safety_stock" in results_df.columns:
                kpis["total_cost"] = results_df["daily_holding_cost_safety_stock"].sum() * 365
            else:
                kpis["total_cost"] = 0.0
            
            if "service_level_target" in results_df.columns:
                kpis["avg_service_level"] = results_df["service_level_target"].mean()
            else:
                kpis["avg_service_level"] = 0.0
            
            if "safety_stock_units" in results_df.columns:
                kpis["total_safety_stock"] = results_df["safety_stock_units"].sum()
            else:
                kpis["total_safety_stock"] = 0.0
            
            if "rop_units" in results_df.columns:
                kpis["avg_rop"] = results_df["rop_units"].mean()
            else:
                kpis["avg_rop"] = 0.0
                
            kpis["items_count"] = len(results_df)
            
            return kpis
        except Exception as e:
            logger.error(f"KPI computation failed: {e}")
            return {"error": f"KPI computation failed: {e}"}
# agents.py (add this class)
class MonitoringAgent:
    """
    Checks MoM changes for the last two months per SKU-location and flags big shifts.
    KPIs: avg_daily_demand, avg_lead_time_days
    """
    def __init__(self, demand_thresh: float = 0.25, lt_thresh: float = 0.25):
        self.demand_thresh = demand_thresh
        self.lt_thresh = lt_thresh

    def check_latest(self) -> Dict[str, Any]:
        import pandas as pd
        df = OptimizationTools.load_data()
        df['date'] = pd.to_datetime(df['date'])
        last_month = df['date'].max()
        prev_month = df['date'].sort_values().unique()[-2] if df['date'].nunique() >= 2 else None
        if prev_month is None:
            return {"success": False, "message": "Not enough months to compare."}

        now = df[df['date'] == last_month]
        prev = df[df['date'] == prev_month]
        key = ['sku_id','location_id']
        merged = now.merge(prev, on=key, suffixes=('_now','_prev'))

        merged['demand_change'] = (merged['avg_daily_demand_now'] - merged['avg_daily_demand_prev'])/merged['avg_daily_demand_prev']
        merged['lt_change'] = (merged['avg_lead_time_days_now'] - merged['avg_lead_time_days_prev'])/merged['avg_lead_time_days_prev']

        alerts = []
        for r in merged.itertuples(index=False):
            flags = []
            if abs(r.demand_change) >= self.demand_thresh:
                flags.append(f"demand {r.demand_change:+.0%}")
            if abs(r.lt_change) >= self.lt_thresh:
                flags.append(f"lead time {r.lt_change:+.0%}")
            if flags:
                alerts.append({"sku_id": r.sku_id, "location_id": r.location_id, "flags": flags})

        return {
            "success": True,
            "last_month": str(last_month.date()),
            "prev_month": str(pd.to_datetime(prev_month).date()),
            "alerts": alerts
        }

class AsyncAgentOrchestrator:
    """Simplified async orchestrator that uses direct optimization tools"""
    
    def __init__(self):
        self.config = safe_config_access()
        self.tools = OptimizationTools()
        self.agents = {}
        logger.info("Async agent orchestrator initialized with direct tools")
    
    async def run_analysis(self, question: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using direct optimization tools"""
        try:
            # Run direct analysis without complex agent interactions
            return await self._run_direct_analysis(question, scenario)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "insights": ["Analysis failed due to system error"],
                "recommendations": ["Check system configuration and try again"]
            }
    
    async def _run_direct_analysis(self, question: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using direct tool calls"""
        try:
            # Stage 1: Apply scenario
            logger.info("Applying scenario transformations")
            scenario_data = await self._run_in_executor(
                self.tools.apply_scenario_delta, scenario
            )
            
            # Stage 2: Run optimization  
            logger.info("Running optimization")
            opt_results = await self._run_in_executor(
                self.tools.run_optimizer, scenario_data
            )
            
            # Stage 3: Generate insights
            logger.info("Generating insights")
            insights = self._generate_direct_insights(question, opt_results, scenario)
            
            return {
                "success": opt_results.get("success", False),
                "question": question,
                "scenario_data": len(scenario_data) if scenario_data is not None else 0,
                "optimization_results": opt_results,
                "insights": insights["insights"],
                "recommendations": insights["recommendations"],
                "execution_mode": "direct_tools"
            }
            
        except Exception as e:
            logger.error(f"Direct analysis failed: {e}")
            raise
    
    async def _run_in_executor(self, func, *args):
        """Run function in executor for async compatibility"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)
    def simple_parse(self, question: str, schema: dict) -> dict | None:
        if not AUTOGEN_AVAILABLE:
            return None
        try:
            # Minimal JSON-only parse with a system guardrail
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            api_key = getattr(self.config, "groq_api_key", None) or os.getenv("GROQ_API_KEY")
            llm = OpenAIChatCompletionClient(model=self.config.model_config.model,
                                 api_key=api_key,
                                 base_url="https://api.groq.com/openai/v1")
            system = ("Return ONLY compact JSON matching this schema (no prose): "
                    f"{json.dumps(schema)}")
            user = f"Question: {question}\nReturn JSON only."
            resp = llm.create(system_messages=[{"role":"system","content":system}],
                            messages=[{"role":"user","content":user}])
            text = resp.content if hasattr(resp, "content") else resp["choices"][0]["message"]["content"]
            # tolerant JSON extraction
            start = text.find("{"); end = text.rfind("}")
            parsed = json.loads(text[start:end+1]) if start>=0 and end>=0 else None
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _generate_direct_insights(self, question: str, opt_results: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights directly from optimization results"""
        insights = []
        recommendations = []
        
        try:
            # Extract scenario information
            changes = scenario.get("changes", {})
            scenario_name = scenario.get("run_options", {}).get("scenario_name", "Analysis")
            
            insights.append(f"Analysis completed for: {question[:100]}")
            insights.append(f"Scenario: {scenario_name}")
            
            if opt_results.get("success"):
                kpis = opt_results.get("kpis", {})
                
                # Cost insights
                if "total_cost" in kpis:
                    total_cost = kpis["total_cost"]
                    insights.append(f"Total annual inventory holding cost: ${total_cost:,.2f}")
                
                # Service level insights  
                if "avg_service_level" in kpis:
                    service_level = kpis["avg_service_level"]
                    insights.append(f"Average service level achieved: {service_level:.1%}")
                    
                    if service_level > 0.98:
                        recommendations.append("Service levels above 98% may have diminishing returns - evaluate cost-effectiveness")
                    elif service_level < 0.90:
                        recommendations.append("Service levels below 90% may indicate stockout risk")
                
                # Inventory insights
                if "total_safety_stock" in kpis:
                    safety_stock = kpis["total_safety_stock"] 
                    insights.append(f"Total safety stock required: {safety_stock:,.0f} units")
                    
                    if safety_stock > 15000:
                        recommendations.append("High safety stock levels detected - review lead time variability and demand patterns")
                
                # Operational insights
                if "items_count" in kpis:
                    items = kpis["items_count"]
                    insights.append(f"Inventory policies optimized for {items} SKU-location combinations")
                
                # Scenario-specific insights
                if changes.get("service_level_target", {}).get("default"):
                    new_sl = changes["service_level_target"]["default"] 
                    insights.append(f"Service level target set to {new_sl:.1%}")
                
                if changes.get("lead_time_days", {}).get("multiplier", 1.0) != 1.0:
                    multiplier = changes["lead_time_days"]["multiplier"]
                    change_pct = (multiplier - 1.0) * 100
                    insights.append(f"Lead times adjusted by {change_pct:+.1f}%")
                
            else:
                error_msg = opt_results.get("message", "Unknown optimization error")
                insights.append(f"Optimization failed: {error_msg}")
                recommendations.append("Review input data quality and parameter ranges")
                recommendations.append("Check solver configuration and network connectivity")
            
            # General recommendations
            if opt_results.get("success"):
                recommendations.extend([
                    "Monitor implementation of new inventory policies closely",
                    "Validate results with historical demand patterns", 
                    "Consider sensitivity analysis on key parameters"
                ])
            
            return {"insights": insights, "recommendations": recommendations}
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return {
                "insights": [f"Analysis completed with limited insights: {str(e)}"],
                "recommendations": ["Review system logs for detailed error information"]
            }

# Fallback for synchronous operation
def create_agent_group() -> Tuple[Any, Any]:
    """Create agent group for backward compatibility"""
    try:
        orchestrator = AsyncAgentOrchestrator()
        return orchestrator, orchestrator  # Return same object as both group and manager
    except Exception as e:
        logger.error(f"Failed to create agent group: {e}")
        raise

if __name__ == "__main__":
    # Test agent creation
    try:
        if not AUTOGEN_AVAILABLE:
            print("AutoGen not available. Install with: pip install autogen-agentchat")
        else:
            orchestrator = AsyncAgentOrchestrator()
            print(f"Created async orchestrator with {len(orchestrator.agents)} agents")
            
            # Test async operation
            async def test_analysis():
                test_scenario = {
                    "changes": {
                        "service_level_target": {"default": 0.98},
                        "lead_time_days": {"multiplier": 1.2}
                    }
                }
                
                result = await orchestrator.run_analysis(
                    "Test question: What if service level increases to 98%?",
                    test_scenario
                )
                print(f"Test result: {result.get('success', False)}")
            
            # Run test if data is available
            try:
                asyncio.run(test_analysis())
            except Exception as e:
                print(f"Test analysis failed (expected if no data): {e}")
                
    except Exception as e:
        print(f"Error creating agents: {e}")
        import traceback
        traceback.print_exc()