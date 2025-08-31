"""
Complete Multi-Agent OptiGuide Orchestration System
--------------------------------------------------
Implements the full OptiGuide pattern with structured workflows:
- Orchestrator (Planner-in-the-loop)
- Data/Scenario Engineer
- Optimizer Agent
- OptiGuide-style Explainer
- Policy & Risk Checker

Uses async patterns with proper fallback handling.
"""

import json,os
from textwrap import dedent
from typing import Dict, List, Any
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

try:
    from agents import AsyncAgentOrchestrator, OptimizationTools, AUTOGEN_AVAILABLE
except ImportError:
    AUTOGEN_AVAILABLE = False
    AsyncAgentOrchestrator = None
    OptimizationTools = None

from config import safe_config_access, get_system_prompt_base

class WorkflowStage(Enum):
    """OptiGuide workflow stages"""
    INITIALIZATION = "initialization"
    SCENARIO_PARSING = "scenario_parsing" 
    DATA_PREPARATION = "data_preparation"
    OPTIMIZATION = "optimization"
    POLICY_VALIDATION = "policy_validation"
    RESULT_EXPLANATION = "result_explanation"
    RECOMMENDATIONS = "recommendations"
    COMPLETE = "complete"

@dataclass
class StageResult:
    """Result from a workflow stage"""
    success: bool
    stage: WorkflowStage
    data: Dict[str, Any]
    insights: List[str]
    errors: List[str] = None
    next_stage: Optional[WorkflowStage] = None

class OptiGuideOrchestrator:
    """Complete OptiGuide orchestrator implementing the full pattern"""
    
    def __init__(self):
        self.config = safe_config_access()
        self.tools = OptimizationTools() if OptimizationTools else None
        self.agent_orchestrator = None
        self.analysis_context = {}
        self.baseline_results = None
        self.setup_orchestrator()
    
    def setup_orchestrator(self):
        """Setup the agent orchestrator with error handling"""
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available - using fallback mode")
            return
        
        try:
            self.agent_orchestrator = AsyncAgentOrchestrator()
            logger.info("OptiGuide orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agent orchestrator: {e}")
    
    async def run_structured_analysis(self, question: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete OptiGuide analysis workflow"""
        
        results = {
            "question": question,
            "scenario": scenario,
            "stages": {},
            "final_insights": [],
            "recommendations": [],
            "success": False,
            "execution_time": 0,
            "workflow_type": "optiguide"
        }
        base_template = {"changes": {}, "run_options": {"scenario_name": "baseline"}}
        if not scenario or "changes" not in scenario or not (scenario.get("changes") or {}):
            scenario = self._nl_to_scenario(question, scenario or base_template)    
        start_time = time.time()
        
        try:
            # Stage 1: Orchestrator - Parse and validate scenario
            results['stages']['scenario_parsing'] = await self._orchestrator_stage(question, scenario)
            
            # Stage 2: Data/Scenario Engineer - Prepare data and apply changes
            if results['stages']['scenario_parsing'].get('success'):
                results['stages']['data_preparation'] = await self._data_engineer_stage(scenario)
            
            # Stage 3: Optimizer Agent - Run optimization
            if results['stages'].get('data_preparation', {}).get('success'):
                results['stages']['optimization'] = await self._optimizer_agent_stage()
            
            # Stage 4: Policy & Risk Checker - Validate results
            if results['stages'].get('optimization', {}).get('success'):
                results['stages']['policy_validation'] = await self._policy_checker_stage()
            
            # Stage 5: OptiGuide Explainer - Generate insights
            if results['stages'].get('optimization', {}).get('success'):
                results['stages']['explanation'] = await self._optiguide_explainer_stage(question)
            
            # Determine overall success
            critical_stages = ['scenario_parsing', 'optimization', 'explanation']
            successful_critical = sum(1 for stage in critical_stages 
                                    if results['stages'].get(stage, {}).get('success', False))
            results['success'] = successful_critical >= 2

            if results['success']:
                results['final_insights'] = self._compile_insights()
                results['recommendations'] = self._compile_recommendations()
                results['conversation_file'] = self._persist_conversation(question, scenario, results.get('stages', {}))        
            
        except Exception as e:
            logger.error(f"OptiGuide analysis failed: {e}")
            results['error'] = str(e)
        finally:
            results['execution_time'] = time.time() - start_time
        
        return results
    
    async def _orchestrator_stage(self, question: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Stage A: Orchestrator (Planner-in-the-loop)"""
        try:
            # Validate question
            if not isinstance(question, str) or not question.strip():
                return {"success": False, "error": "Empty question provided"}
            
            # Parse question for intent
            question_analysis = self._analyze_question_intent(question)
            
            # Validate scenario structure
            if not isinstance(scenario, dict) or 'changes' not in scenario:
                return {"success": False, "error": "Invalid scenario format"}
            
            changes = scenario.get("changes", {})
            parsed_info = {
                "question_type": question_analysis["type"],
                "key_parameters": question_analysis["parameters"],
                "service_level_changes": bool(changes.get("service_level_target")),
                "lead_time_changes": bool(changes.get("lead_time_days")),
                "moq_changes": bool(changes.get("moq_units")),
                "review_period_changes": bool(changes.get("review_period_days")),
                "demand_changes": bool(changes.get("demand_multiplier")),  # NEW
            }
            
            # Store in context
            self.analysis_context['question_analysis'] = question_analysis
            self.analysis_context['validated_scenario'] = scenario

            # NEW: count only booleans; + number of detected parameters in the question
            flag_count = sum(1 for k, v in parsed_info.items() if isinstance(v, bool) and v)
            param_count = len(parsed_info.get("key_parameters") or [])

            return {
                "success": True,
                "stage": "orchestrator",
                "parsed_info": parsed_info,
                "response": (
                    f"Orchestrator parsed '{question[:100]}' and identified "
                    f"{flag_count} scenario change(s) and {param_count} focus parameter(s)."
                )
            }               
            
        except Exception as e:
            logger.error(f"Orchestrator stage failed: {e}")
            return {"success": False, "error": str(e)}
    def _nl_to_scenario(self, question: str, base: Dict[str, Any]) -> Dict[str, Any]:
        text = question.lower()
        changes = {}
        filters = {}
        scenario = dict(base or {})
        scenario.setdefault("changes", {})
        ch = scenario["changes"]

        if self.agent_orchestrator and AUTOGEN_AVAILABLE:
            # Ask a small agent to emit ONLY JSON with this schema
            schema = {
                "changes": {
                    "service_level_target": {"default": "float in (0,1)", "overrides": "map 'SKU@LOC'->float"},
                    "lead_time_days": {"multiplier": "float > 0"},
                    "review_period_days": "one of [1,7,14,28]"
                },
                "run_options": {"scenario_name": "str"}
            }
            # (Pseudo-call: your AsyncAgentOrchestrator already holds LLM clients)
            parsed = self.agent_orchestrator.simple_parse(question, schema)  # implement to return dict or None
            if isinstance(parsed, dict):
                # deep-merge parsed into scenario
                scenario["changes"].update(parsed.get("changes", {}))
                scenario["run_options"] = parsed.get("run_options", scenario.get("run_options", {}))
        else:
            # Heuristics (fallback): extract percentages & service levels
            import re
            text = question.lower()
            scenario.setdefault("changes", {})
            ch = scenario["changes"]

            overrides = {}
            for m in re.finditer(r"(sku[-_0-9a-z]+).*?(increase|decrease|drop)s?.*?(\d+)\s*%", text):
                sku = m.group(1).upper()
                sign = m.group(2)
                pct  = int(m.group(3))
                factor = 1 + (pct/100.0) * (+1 if "increase" in sign else -1)
                overrides[sku] = factor

            # global demand % (e.g., “demand increases 90%”)
            m = re.search(r"(demand|avg_daily_demand|volume)[^%]*?(\d+)\s*%", text)
            if m and "demand_multiplier" not in ch:
                ch["demand_multiplier"] = {"default": 1 + int(m.group(2))/100.0, "overrides": overrides or {}}
            elif overrides:
                ch["demand_multiplier"] = {"default": 1.0, "overrides": overrides}

            # generic percent without keyword -> keep your existing LT multiplier fallback
            m = re.search(r"(\d+)\s*%\s*(increase|decrease|change)?", text)
            if m and "demand_multiplier" not in ch and "lead_time_days" not in ch:
                pct = 1 + int(m.group(1))/100.0
                ch.setdefault("lead_time_days", {})["multiplier"] = pct

            # service level like 95% / 98% / 99%
            m = re.search(r"(9[0-9]|8[5-9])\s*%", text)
            if m:
                sl = int(m.group(1))/100.0
                ch.setdefault("service_level_target", {})["default"] = sl

        return scenario

    def _persist_conversation(self, question: str, scenario: Dict[str, Any], stages: Dict[str, Any]) -> str:
        import time, json
        cfg = safe_config_access()
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = cfg.paths.conversations_dir / f"conversation_{ts}.json"
        payload = {"question": question, "scenario": scenario, "stages": stages}
        out.write_text(json.dumps(payload, indent=2))
        return str(out)    
    def _analyze_question_intent(self, question: str) -> Dict[str, Any]:
        """Analyze the what-if question to understand intent"""
        question_lower = question.lower()
        
        intent_analysis = {
            "type": "general",
            "parameters": [],
            "comparison_needed": False,
            "impact_focus": []
        }
        
        # Detect parameter focus
        if any(term in question_lower for term in ["lead time", "supplier", "delivery"]):
            intent_analysis["parameters"].append("lead_time")
            intent_analysis["type"] = "lead_time_scenario"
        
        if any(term in question_lower for term in ["service level", "fill rate", "availability"]):
            intent_analysis["parameters"].append("service_level")
            intent_analysis["type"] = "service_level_scenario"
        
        if any(term in question_lower for term in ["cost", "inventory", "holding"]):
            intent_analysis["impact_focus"].append("cost")
        
        if any(term in question_lower for term in ["stock", "availability", "shortage"]):
            intent_analysis["impact_focus"].append("availability")
        
        # Detect comparison intent
        if any(term in question_lower for term in ["compare", "vs", "versus", "difference"]):
            intent_analysis["comparison_needed"] = True
        
        return intent_analysis
    
    async def _data_engineer_stage(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Stage B: Data/Scenario Engineer"""
        try:
            if not self.tools:
                return {"success": False, "error": "Optimization tools not available"}
            
            # Load base data
            base_data = await self._run_in_executor(self.tools.load_data)
            if base_data.empty:
                return {"success": False, "error": "No base data available"}
            
            # Apply scenario changes
            scenario_data = await self._run_in_executor(self.tools.apply_scenario_delta, scenario)
            
            # Compute baseline KPIs for comparison
            baseline_kpis = self._compute_baseline_kpis(base_data)
            
            # Store in context
            self.analysis_context['base_data'] = base_data
            self.analysis_context['scenario_data'] = scenario_data
            self.analysis_context['baseline_kpis'] = baseline_kpis
            
            return {
                "success": True,
                "stage": "data_engineer",
                "data_stats": {
                    "base_records": len(base_data),
                    "scenario_records": len(scenario_data),
                    "unique_skus": scenario_data['sku_id'].nunique(),
                    "unique_locations": scenario_data['location_id'].nunique()
                },
                "response": f"Data Engineer prepared {len(scenario_data)} records with scenario transformations applied"
            }
            
        except Exception as e:
            logger.error(f"Data Engineer stage failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _optimizer_agent_stage(self) -> Dict[str, Any]:
        """Stage C: Optimizer Agent"""
        try:
            scenario_data = self.analysis_context.get('scenario_data')
            if scenario_data is None:
                return {"success": False, "error": "No scenario data available"}
            
            # Run optimization
            opt_results = await self._run_in_executor(self.tools.run_optimizer, scenario_data)
            
            if not opt_results.get('success'):
                return {"success": False, "error": opt_results.get('message', 'Optimization failed')}
            
            # Store results in context
            self.analysis_context['optimization_results'] = opt_results
            
            # Generate optimization trace
            trace = [
                f"Optimization completed successfully",
                f"Policy generated for {len(opt_results.get('policy', []))} items",
                f"Total annual cost: ${opt_results.get('kpis', {}).get('total_cost', 0):,.2f}"
            ]
            
            return {
                "success": True,
                "stage": "optimizer_agent",
                "optimization_results": opt_results,
                "trace": trace,
                "response": f"Optimizer Agent completed successfully with total cost ${opt_results.get('kpis', {}).get('total_cost', 0):,.2f}"
            }
            
        except Exception as e:
            logger.error(f"Optimizer Agent stage failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _policy_checker_stage(self) -> Dict[str, Any]:
        """Stage E: Policy & Risk Checker"""
        try:
            opt_results = self.analysis_context.get('optimization_results', {})
            policy_data = opt_results.get('policy', [])
            
            if not policy_data:
                return {"success": False, "error": "No policy data to validate"}
            
            validation_results = {
                "total_items": len(policy_data),
                "feasible_items": 0,
                "risk_flags": [],
                "constraint_violations": []
            }
            
            for item in policy_data:
                feasible = True
                
                # Check for negative safety stock
                if item.get('safety_stock_units', 0) < 0:
                    validation_results["risk_flags"].append(f"Negative safety stock for {item.get('sku_id')}")
                    feasible = False
                
                # Check ROP vs demand relationship
                rop = item.get('rop_units', 0)
                daily_demand = item.get('avg_daily_demand', 1)
                if rop < daily_demand * 7:  # Less than 1 week of demand
                    validation_results["risk_flags"].append(f"Low ROP detected for {item.get('sku_id')}")
                
                # Check service level reasonableness
                service_level = item.get('service_level_target', 0.95)
                if service_level > 0.995:
                    validation_results["risk_flags"].append(f"Very high service level ({service_level:.1%}) for {item.get('sku_id')}")
                
                if feasible:
                    validation_results["feasible_items"] += 1
            
            # Store in context
            self.analysis_context['policy_validation'] = validation_results
            
            risk_count = len(validation_results["risk_flags"])
            response = f"Policy Checker validated {validation_results['total_items']} items, found {risk_count} risk flags"
            
            return {
                "success": True,
                "stage": "policy_checker",
                "validation_results": validation_results,
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Policy Checker stage failed: {e}")
            return {"success": False, "error": str(e)}
    # ----- LLM Explainer helpers -------------------------------------------------
    def _collapse_policy_for_llm(self, policy_list, kpis, top_n=30):
        """Reduce policy list to a compact payload for the LLM."""
        if not isinstance(policy_list, list):
            return {"items": [], "kpis": kpis or {}}

        # Keep only the most useful columns (robust to missing keys)
        keep = [
            "sku_id", "location_id",
            "service_level", "service_level_target",
            "safety_stock", "safety_stock_units", "safety_stock_units_rounded",
            "rop", "rop_units", "rop_units_rounded",
            "order_qty", "order_quantity",
            "expected_stockouts",
            "holding_cost", "order_cost", "total_cost",
            "avg_daily_demand", "std_daily_demand",
            "avg_lead_time_days", "std_lead_time_days",
        ]

        def shrink(d):
            return {k: d.get(k) for k in keep if k in d}

        slim = [shrink(x) for x in policy_list]
        # Heuristic: sort by total_cost or safety_stock where present
        def score(x):
            return (
                (x.get("total_cost") if x.get("total_cost") is not None else 0.0),
                (x.get("safety_stock") or x.get("safety_stock_units") or 0.0)
            )
        slim.sort(key=score, reverse=True)
        return {"items": slim[:top_n], "kpis": kpis or {}}

    def _llm_explain(self, question, scenario, baseline_kpis, optimized_kpis, policy, trace):
        """Call GROQ to interpret solver output (OPTIGUIDE: LLM interprets results)."""
        try:
            cfg = self.config  # Config instance
            api_key = (
                getattr(getattr(cfg, "model_config", None), "groq_api_key", None)
                or getattr(cfg, "groq_api_key", None)
                or os.getenv("GROQ_API_KEY")
            )
            if not api_key:
                raise RuntimeError("Missing GROQ_API_KEY")

            # Import locally to avoid import errors when key is absent
            from groq import Groq  # requires 'groq' package

            client = Groq(api_key=api_key)

            system_prompt = dedent(get_system_prompt_base()) + dedent("""
            You are the *Explainer Agent* in an OPTIGUIDE loop.
            - Do NOT invent numbers: use only the optimizer outputs provided.
            - Optimizer is the source of truth; you only interpret.
            Return a compact JSON with keys:
              summary: str
              insights: [str]
              tradeoffs: [str]
              risks: [str]
              next_tests: [str]
            """)

            payload = {
                "question": question,
                "scenario": scenario or {},
                "baseline_kpis": baseline_kpis or {},
                "optimized_kpis": optimized_kpis or {},
                "policy_top": self._collapse_policy_for_llm(policy, optimized_kpis, top_n=30),
                "trace": trace or [],
            }

            # Ask for JSON; if model emits text-with-json, we will robustly parse.
# Ask for JSON; if model emits text-with-json, we'll robustly parse.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "You will receive a JSON package with optimizer outputs. Interpret it."},
                {"role": "assistant", "content": "Ready. Please provide the JSON inputs."},
                {"role": "user", "content": json.dumps(payload)},
            ]
            resp = client.chat.completions.create(
                model=getattr(getattr(cfg, "model_config", None), "model", "llama-3.1-70b-versatile"),
                messages=messages,
                temperature=getattr(getattr(cfg, "model_config", None), "temperature", 0.1),
                max_tokens=getattr(getattr(cfg, "model_config", None), "max_tokens", 4000),
                timeout=getattr(getattr(cfg, "model_config", None), "timeout", 120),
            )

            content = resp.choices[0].message.content if resp and resp.choices else "{}"
            # Try to parse JSON; fallback to wrapping as free-text
            try:
                data = json.loads(content)
            except Exception:
                data = {
                    "summary": content.strip()[:1200],
                    "insights": [],
                    "tradeoffs": [],
                    "risks": [],
                    "next_tests": [],
                }

            # Normalize keys for downstream code (_compile_* expects these)
            explanation = {
                "summary": data.get("summary", ""),
                "insights": data.get("insights", []) or [],
                "trade_offs": data.get("tradeoffs", []) or [],
                "key_drivers": [],  # optional; can be inferred by LLM in "insights"
                "recommendations": data.get("next_tests", []) or [],
                "next_scenarios": data.get("next_tests", []) or [],
                "risks": data.get("risks", []) or [],
            }
            return explanation, True  # used_llm=True

        except Exception as e:
            # Fall back to deterministic explainer on any error
            logger.warning(f"LLM explain failed, falling back: {e}")
            return None, False
    def _llm_explain_groq(self, question, scenario, baseline_kpis, optimized_kpis, policy, trace):
            return self._llm_explain(question, scenario, baseline_kpis, optimized_kpis, policy, trace)

    def _autogen_llm_explain(self, question, scenario, baseline_kpis, optimized_kpis, policy, trace):
        """Optional AutoGen wrapper. For now, safely delegates to direct GROQ call.
        You can swap this to a GroupChat flow later without changing callers.
        """
        explanation, used_llm = self._llm_explain(
            question, scenario, baseline_kpis, optimized_kpis, policy, trace
        )
        return explanation, used_llm
    
    
    async def _optiguide_explainer_stage(self, question: str) -> dict:
        """
        Stage: Explainer.
        First try the LLM (Groq) to interpret results; on any issue, fall back to deterministic explainer you already have.
        """
        try:
            opt_results = self.analysis_context.get("optimization_results", {}) or {}
            baseline_kpis = self.analysis_context.get("baseline_kpis", {}) or {}
            scenario = self.analysis_context.get("validated_scenario", {}) or {}
            q_analysis = self.analysis_context.get("question_analysis", {}) or {}

            if not opt_results.get("success"):
                return {"success": False, "error": "No optimization results to explain."}

            kpis = opt_results.get("kpis", {}) or {}
            policy = opt_results.get("policy", []) or []
            trace = opt_results.get("trace", []) or []

            # Try Groq LLM explainer
            explanation, used_llm = self._llm_explain(
                question=question,
                scenario=scenario,
                baseline_kpis=baseline_kpis,
                optimized_kpis=kpis,
                policy=policy,
                trace=trace,
            )
            if explanation is None:
                # Deterministic fallback (your existing function)
                explanation = self._generate_optiguide_explanation(
                    question, opt_results, baseline_kpis, q_analysis
                )
                used_llm = False

            # Save and return
            self.analysis_context["optiguide_explanation"] = explanation
            return {
                "success": True,
                "stage": "optiguide_explainer",
                "used_llm": used_llm,
                "explanation": explanation,
                "response": (
                    f"Explainer ({'LLM' if used_llm else 'deterministic'}) "
                    f"returned {len(explanation.get('insights', []))} insights and "
                    f"{len(explanation.get('recommendations', []))} next tests."
                ),
            }

        except Exception as e:
            logger.error(f"Explainer stage failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_optiguide_explanation(self, question: str, opt_results: Dict[str, Any], 
                                      baseline_kpis: Dict[str, Any], question_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate OptiGuide-style explanation - LLM interprets solver output"""
        
        current_kpis = opt_results.get('kpis', {})
        policy_data = opt_results.get('policy', [])
        
        explanation = {
            "summary": "",
            "insights": [],
            "trade_offs": [],
            "key_drivers": [],
            "recommendations": [],
            "next_scenarios": []
        }
        
        # Summary
        total_cost = current_kpis.get('total_cost', 0)
        service_level = current_kpis.get('avg_service_level', 0)
        explanation["summary"] = f"Analysis shows total annual cost of ${total_cost:,.2f} with {service_level:.1%} average service level"
        
        # Key insights based on question intent
        if question_analysis.get("type") == "service_level_scenario":
            if service_level > 0.98:
                explanation["insights"].append(f"High service level target of {service_level:.1%} achieved but at premium cost")
            elif service_level < 0.90:
                explanation["insights"].append(f"Service level of {service_level:.1%} may indicate stockout risk")
            else:
                explanation["insights"].append(f"Balanced service level of {service_level:.1%} achieved")
        
        if question_analysis.get("type") == "lead_time_scenario":
            total_safety_stock = current_kpis.get('total_safety_stock', 0)
            if total_safety_stock > 10000:
                explanation["insights"].append(f"Lead time changes resulted in high safety stock requirement of {total_safety_stock:,.0f} units")
            explanation["key_drivers"].append("Lead time variability is primary driver of safety stock requirements")
        
        # Cost analysis
        if baseline_kpis and 'total_cost' in baseline_kpis:
            cost_change = total_cost - baseline_kpis['total_cost']
            change_pct = (cost_change / baseline_kpis['total_cost']) * 100 if baseline_kpis['total_cost'] > 0 else 0
            if abs(change_pct) > 5:
                explanation["trade_offs"].append(f"Total cost changed by {change_pct:+.1f}% vs baseline (${cost_change:+,.2f})")
        
        # Risk assessment
        high_cost_items = [item for item in policy_data 
                          if item.get('daily_holding_cost_safety_stock', 0) > total_cost / len(policy_data) * 2]
        if high_cost_items:
            explanation["insights"].append(f"{len(high_cost_items)} items account for disproportionate holding costs")
        
        # Recommendations based on OptiGuide pattern
        if service_level > 0.98:
            explanation["recommendations"].append("Consider if service levels above 98% provide sufficient ROI")
            explanation["next_scenarios"].append("Test impact of reducing service level to 95% for B/C class items")
        
        if current_kpis.get('total_safety_stock', 0) > 15000:
            explanation["recommendations"].append("High safety stock levels detected - review supplier lead time reliability")
            explanation["next_scenarios"].append("Analyze impact of 10% lead time reduction through supplier partnerships")
        
        explanation["recommendations"].extend([
            "Implement new ROP policies gradually with close monitoring",
            "Validate results against historical demand patterns",
            "Consider ABC analysis for differentiated service levels"
        ])
        
        return explanation
    
    def _compute_baseline_kpis(self, base_data) -> Dict[str, float]:
        """Compute baseline KPIs for comparison"""
        try:
            if base_data.empty:
                return {}
            
            # Get latest date data
            latest_date = base_data['date'].max()
            latest_data = base_data[base_data['date'] == latest_date]
            
            kpis = {}
            
            # Estimate baseline cost (simplified)
            if 'unit_cost' in latest_data.columns and 'avg_daily_demand' in latest_data.columns:
                # Rough estimate: assume 30 days of inventory on hand
                inventory_value = (latest_data['unit_cost'] * latest_data['avg_daily_demand'] * 30).sum()
                holding_rate = latest_data['holding_cost_rate_annual'].mean() if 'holding_cost_rate_annual' in latest_data.columns else 0.25
                kpis['total_cost'] = inventory_value * holding_rate
            
            if 'service_level_target' in latest_data.columns:
                kpis['avg_service_level'] = latest_data['service_level_target'].mean()
            
            return kpis
            
        except Exception as e:
            logger.error(f"Baseline KPI computation failed: {e}")
            return {}
    
    async def _run_in_executor(self, func, *args):
        """Run synchronous function in executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)
    
    def _compile_insights(self) -> List[str]:
        """Compile insights from OptiGuide explanation"""
        explanation = self.analysis_context.get('optiguide_explanation', {})
        
        insights = []
        
        # Add summary
        if explanation.get('summary'):
            insights.append(explanation['summary'])
        
        # Add key insights
        insights.extend(explanation.get('insights', []))
        
        # Add trade-offs
        insights.extend(explanation.get('trade_offs', []))
        
        # Add key drivers
        for driver in explanation.get('key_drivers', []):
            insights.append(f"Key driver: {driver}")
        
        # Fallback insights if explanation failed
        if not insights:
            opt_results = self.analysis_context.get('optimization_results', {})
            if opt_results.get('success'):
                kpis = opt_results.get('kpis', {})
                insights.append(f"Optimization completed with ${kpis.get('total_cost', 0):,.2f} total annual cost")
                insights.append(f"Average service level: {kpis.get('avg_service_level', 0):.1%}")
        
        return insights
    
    def _compile_recommendations(self) -> List[str]:
        """Compile recommendations from OptiGuide explanation"""
        explanation = self.analysis_context.get('optiguide_explanation', {})
        
        recommendations = explanation.get('recommendations', [])
        
        # Add next scenario suggestions
        for scenario in explanation.get('next_scenarios', []):
            recommendations.append(f"Next test: {scenario}")
        
        # Fallback recommendations
        if not recommendations:
            recommendations.extend([
                "Monitor implementation of new inventory policies closely",
                "Validate results with historical demand patterns",
                "Consider sensitivity analysis on key parameters"
            ])
        
        return recommendations


class StreamlitOptiGuideOrchestrator:
    """Streamlit wrapper for OptiGuide orchestrator"""
    
    def __init__(self):
        self.core_orchestrator = OptiGuideOrchestrator()
        self.analysis_history = []
    
    def run_analysis(self, question: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run OptiGuide analysis and format for Streamlit"""
        try:
            # Run async analysis in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    self.core_orchestrator.run_structured_analysis(question, scenario)
                )
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"OptiGuide analysis execution failed: {e}")
            results = {
                "success": False,
                "error": str(e),
                "question": question,
                "scenario": scenario,
                "workflow_type": "optiguide_failed"
            }
        
        # Format for Streamlit display
        formatted_results = self._format_for_streamlit(results, question, scenario)
        
        # Add to history
        self.analysis_history.append(formatted_results)
        
        return formatted_results
    
    def _format_for_streamlit(self, results: Dict[str, Any], question: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Format OptiGuide results for Streamlit display"""
        scenario = (
                    scenario
                    or results.get("validated_scenario")
                    or results.get("scenario")
                    or {}
                )
        run_opts = scenario.get("run_options") or {}
        scenario_name = run_opts.get("scenario_name", "OptiGuide Analysis")

        formatted_results = {
            "success": results.get('success', False),
            "question": question,
            "scenario_name": scenario.get('run_options', {}).get('scenario_name', 'OptiGuide Analysis'),
            "timestamp": time.time(),
            "stages": [],
            "insights": results.get('final_insights', []),
            "recommendations": results.get('recommendations', []),
            "error": results.get('error'),
            "execution_time": results.get('execution_time', 0),
            "workflow_type": results.get('workflow_type', 'optiguide')
        }
        
        # Format stage results for display
        stage_display_names = {
            "scenario_parsing": "Orchestrator (Planning)",
            "data_preparation": "Data/Scenario Engineer", 
            "optimization": "Optimizer Agent",
            "policy_validation": "Policy & Risk Checker",
            "explanation": "OptiGuide Explainer"
        }
        
        kpis = (results.get("kpis")
            or (results.get("stages", {}).get("optimization", {})
                .get("optimization_results", {})
                .get("kpis")))
        
        if kpis:
            formatted_results["kpis"] = kpis

        # Bubble up policy (list of per-material rows)
        policy = (results.get("policy")
            or (results.get("stages", {}).get("optimization", {})
                .get("optimization_results", {})
                .get("policy")))
        if policy:
            formatted_results["policy"] = policy

        for stage_key, stage_result in results.get('stages', {}).items():
            formatted_stage = {
                "name": stage_display_names.get(stage_key, stage_key.replace('_', ' ').title()),
                "success": stage_result.get('success', False),
                "response": stage_result.get('response', self._get_stage_summary(stage_result)),
                "error": stage_result.get('error'),
                "trace": stage_result.get('trace') or (stage_result.get('optimization_results') or {}).get('trace', [])
            }
            formatted_results['stages'].append(formatted_stage)
        
        return formatted_results
    
    def _get_stage_summary(self, stage_result: Dict[str, Any]) -> str:
        """Generate summary for stage results"""
        if 'optimization_results' in stage_result:
            results = stage_result['optimization_results']
            return f"Optimization completed with ${results.get('kpis', {}).get('total_cost', 0):,.2f} total cost"
        elif 'data_stats' in stage_result:
            stats = stage_result['data_stats']
            return f"Processed {stats['scenario_records']} records for {stats['unique_skus']} SKUs"
        elif 'validation_results' in stage_result:
            validation = stage_result['validation_results']
            return f"Validated {validation['total_items']} items, {len(validation['risk_flags'])} risk flags"
        elif 'explanation' in stage_result:
            explanation = stage_result['explanation']
            return f"Generated {len(explanation['insights'])} insights with trade-off analysis"
        else:
            return "Stage completed successfully"
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get formatted analysis history"""
        return self.analysis_history
    
    def clear_history(self):
        """Clear analysis history"""
        self.analysis_history = []
        if hasattr(self.core_orchestrator, 'analysis_context'):
            self.core_orchestrator.analysis_context = {}


# Global orchestrator instance for Streamlit - now using complete OptiGuide pattern
streamlit_orchestrator = StreamlitOptiGuideOrchestrator()

# Utility functions
def run_quick_analysis(question: str, service_level: float = 0.95, lead_time_multiplier: float = 1.0) -> Dict[str, Any]:
    """Quick OptiGuide analysis function"""
    scenario = {
        "changes": {
            "service_level_target": {"default": service_level},
            "lead_time_days": {"multiplier": lead_time_multiplier}
        },
        "run_options": {"scenario_name": "quick_optiguide_analysis"}
    }
    
    return streamlit_orchestrator.run_analysis(question, scenario)

def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health status"""
    try:
        from config import check_system_health
        config_health = check_system_health()
    except Exception as e:
        config_health = {"status": "error", "error": str(e)}
    
    return {
        "config_health": config_health,
        "optiguide_available": streamlit_orchestrator.core_orchestrator.tools is not None,
        "autogen_available": AUTOGEN_AVAILABLE,
        "overall_status": "healthy" if config_health.get("status") == "healthy" else "degraded"
    }

# Backward compatibility
def create_agent_group():
    """Create agent group for backward compatibility"""
    return streamlit_orchestrator.core_orchestrator, streamlit_orchestrator.core_orchestrator

if __name__ == "__main__":
    # Test the complete OptiGuide system
    print("Testing OptiGuide Multi-Agent System...")
    
    try:
        # Test system health
        health = get_system_health()
        print(f"System health: {health['overall_status']}")
        print(f"OptiGuide available: {health['optiguide_available']}")
        print(f"AutoGen available: {health['autogen_available']}")
        
        # Test OptiGuide orchestrator
        orchestrator = StreamlitOptiGuideOrchestrator()
        print("OptiGuide orchestrator created successfully")
        
        # Test quick analysis
        print("\nRunning OptiGuide analysis test...")
        result = run_quick_analysis(
            "What if lead time increases 20% and we target 98% service level?",
            service_level=0.98,
            lead_time_multiplier=1.2
        )
        
        print(f"OptiGuide analysis result: Success={result['success']}")
        print(f"Workflow type: {result.get('workflow_type', 'unknown')}")
        print(f"Stages completed: {len(result.get('stages', []))}")
        print(f"Insights generated: {len(result.get('insights', []))}")
        print(f"Recommendations: {len(result.get('recommendations', []))}")
        
        if result.get('error'):
            print(f"Error encountered: {result['error']}")
        
        if result['success']:
            print("\nOptiGuide system test completed successfully!")
            print("The system implements the complete OptiGuide pattern:")
            print("- Orchestrator (Planner-in-the-loop)")
            print("- Data/Scenario Engineer") 
            print("- Optimizer Agent")
            print("- OptiGuide Explainer (LLM interprets optimization results)")
            print("- Policy & Risk Checker")
        else:
            print("\nOptiGuide system test completed with issues - check configuration")
            
    except Exception as e:
        print(f"OptiGuide test failed: {e}")
        import traceback
        traceback.print_exc()