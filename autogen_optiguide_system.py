#!/usr/bin/env python3
"""
Fixed AutoGen-Powered OptiGuide Implementation for Groq
======================================================
Properly configures AutoGen with Groq LLM to avoid fallback mode.
"""

import json
import logging
import re
import time
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
import asyncio
import os
import sys
import tempfile
import subprocess
import warnings

# Suppress AutoGen serialization warnings
logging.getLogger("autogen_core.events").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Message could not be serialized")

logger = logging.getLogger(__name__)

# AutoGen imports with proper error handling
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat  
    from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
    from autogen_agentchat.messages import TextMessage
    from autogen_agentchat.base import TaskResult
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
    logger.info("AutoGen modern API available") 
except ImportError as e:
    AUTOGEN_AVAILABLE = False
    logger.warning(f"AutoGen not available: {e}")

# Import config
try:
    from config import safe_config_access
    config = safe_config_access()
except ImportError:
    logger.error("Could not import config")
    config = None

@dataclass
class AnalysisResult:
    """Structured result from OptiGuide analysis"""
    success: bool
    conversation_summary: str
    insights: List[str]
    recommendations: List[str]
    kpis: Optional[Dict[str, Any]] = None
    policy_data: Optional[List[Dict]] = None
    execution_time: float = 0
    agent_conversations: List[str] = None

class AutoGenOptiGuideOrchestrator:
    """Fixed AutoGen-powered OptiGuide orchestration"""
    
    def __init__(self):
        self.config = config
        self.agents = {}
        self.team = None
        self._last_opt_result: dict = None
        
        if AUTOGEN_AVAILABLE and self.config:
            self._setup_agents()
        else:
            logger.warning("AutoGen not available - OptiGuide system will use fallback mode")
    def _parse_question_to_scenario(self, question: str) -> Dict[str, Any]:
        """Parse natural language question into scenario with fixed regex"""
        question_lower = question.lower()
        
        scenario = {
            "changes": {},
            "run_options": {"scenario_name": "analysis"}
        }
        
        # Fixed lead time parsing
        lead_time_patterns = [
            r'lead\s*times?\s+(?:increase|rise|go\s+up)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%',
            r'lead\s*times?\s+.*?(\d+(?:\.\d+)?)\s*%.*?(?:increase|rise|higher)',
            r'(\d+(?:\.\d+)?)\s*%.*?lead\s*time.*?(?:increase|rise)'
        ]
        
        for pattern in lead_time_patterns:
            match = re.search(pattern, question_lower)
            if match:
                percentage = float(match.group(1))
                multiplier = 1.0 + (percentage / 100.0)
                scenario["changes"]["lead_time_days"] = {"multiplier": multiplier}
                logger.info(f"Parsed lead time increase: {percentage}% -> multiplier {multiplier}")
                break
        
        # Extract SKU-specific demand changes (existing code)
        demand_overrides = {}
        sku_patterns = [
            r'(sku[-\s]*\w+).*?(?:increase?|rise?).*?(\d+)\s*%',
            r'(sku[-\s]*\w+).*?(?:decrease?|drop?).*?(\d+)\s*%',
            r'(?:demand.*?for\s+)?(sku[-\s]*\w+).*?(?:increases?|rises?).*?(\d+)\s*%',
            r'(?:demand.*?for\s+)?(sku[-\s]*\w+).*?(?:decreases?|drops?).*?(\d+)\s*%'
        ]
        
        for pattern in sku_patterns:
            for match in re.finditer(pattern, question_lower):
                sku = match.group(1).upper().replace(' ', '-')
                if not sku.startswith('SKU-'):
                    sku = f"SKU-{sku}"
                
                percentage = float(match.group(2))
                is_decrease = any(word in match.group(0) for word in ['decrease', 'drop', 'fall'])
                multiplier = 1 + (percentage / 100 * (-1 if is_decrease else 1))
                demand_overrides[sku] = multiplier

        if demand_overrides:
            scenario["changes"]["demand_multiplier"] = {
                "default": 1.0,
                "overrides": demand_overrides
            }

        # Extract service level changes (existing code)
        service_match = re.search(r'service\s+level.*?(\d+)\s*%', question_lower)
        if service_match:
            sl_percentage = float(service_match.group(1)) / 100.0
            scenario["changes"]["service_level_target"] = {"default": sl_percentage}

        return scenario
    def _setup_agents(self):
        """Setup AutoGen agents with proper Groq configuration"""
        try:
            # Get API key - try multiple sources
            api_key = (
                os.getenv("GROQ_API_KEY") or 
                getattr(self.config, 'groq_api_key', None) or
                os.getenv("OPENAI_API_KEY")  # Fallback
            )
            
            if not api_key:
                raise ValueError("No API key found. Set GROQ_API_KEY in environment or .env file")
            
            # Get model name
            model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            
            # Create the OpenAI-compatible client for Groq
            client = OpenAIChatCompletionClient(
                model=model_name,
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                timeout=120,
                max_retries=3,
                model_info={
                    "vision": False,
                    "function_calling": True,
                    "json_output": True,
                    "structured_output": True,
                    "family": "llama",
                },
            )

            # Tool function for optimization
            def _run_optimization_tool(scenario_json: str) -> dict:
                """Run optimization and return results with robust JSON parsing"""
                try:
                    # Handle both string and dict inputs
                    if isinstance(scenario_json, dict):
                        scenario = scenario_json
                    else:
                        # Clean and parse JSON string
                        cleaned = scenario_json.strip()
                        if cleaned.startswith('"') and cleaned.endswith('"'):
                            cleaned = cleaned[1:-1]  # Remove outer quotes
                        
                        # Replace escaped quotes
                        cleaned = cleaned.replace('\\"', '"')
                        
                        try:
                            scenario = json.loads(cleaned)
                        except json.JSONDecodeError:
                            # Fallback: try to extract JSON from text
                            import re
                            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                            if json_match:
                                scenario = json.loads(json_match.group())
                            else:
                                raise ValueError(f"Could not parse JSON: {cleaned}")

                    logger.info(f"Parsed scenario: {scenario}")
                    
                    df_latest = self._apply_scenario_changes(scenario)
                    result = self._run_optimizer(df_latest)
                    self._last_opt_result = result

                    return result

                except Exception as e:
                    logger.error(f"Optimization tool failed: {e}")
                    return {"success": False, "error": str(e)}

            # Create agents with proper system messages
            self.agents["data_engineer"] = AssistantAgent(
                "DataEngineer",
                model_client=client,
                system_message=(
                    "You are a Data Engineer specializing in inventory optimization. "
                    "Your job is to validate scenario JSON inputs, describe data transformations, "
                    "and ensure the scenario is properly formatted for optimization. "
                    "When you receive a scenario, explain what changes will be applied and "
                    "then pass control to the Optimizer."
                ),
            )

            self.agents["optimizer"] = AssistantAgent(
                "Optimizer",
                model_client=client,
                tools=[_run_optimization_tool],
                system_message=(
                    "You are an Inventory Optimizer. Use the `_run_optimization_tool(scenario_json)` "
                    "to compute optimal reorder points and safety stock levels. "
                    "IMPORTANT: When calling the tool, pass the scenario_json as a clean JSON string "
                    "without extra escaping. For example: "
                    '_run_optimization_tool(\'{"changes":{"lead_time_days":{"multiplier":1.2}}}\') '
                    "After calling the tool, summarize the key KPIs: total_cost, avg_service_level, "
                    "total_safety_stock, and items_count. "
                    "Then pass control to the Explainer for business interpretation."
                ),
            )

            self.agents["explainer"] = AssistantAgent(
                "Explainer",
                model_client=client,
                system_message=(
                    "You are a Supply Chain Business Analyst. Interpret optimization results "
                    "in business terms, highlighting cost-service level trade-offs, risks, "
                    "and actionable recommendations. "
                    "End your final message with exactly: COMPLETE"
                ),
            )

            # Create team with termination condition
            self.team = RoundRobinGroupChat(
                [self.agents["data_engineer"], self.agents["optimizer"], self.agents["explainer"]],
                termination_condition=TextMentionTermination("COMPLETE"),
                max_turns=10
            )
            
            logger.info("AutoGen team initialized successfully with Groq")
            
        except Exception as e:
            logger.error(f"Failed to setup AutoGen agents: {e}")
            self.agents = {}
            self.team = None
    
    def _apply_scenario_changes(self, scenario: Dict[str, Any]) -> pd.DataFrame:
        """Apply scenario changes to base data"""
        try:
            # Load base data
            data_file = self.config.paths.synthetic_data_file
            if not data_file.exists():
                raise FileNotFoundError(f"No data file found at {data_file}")
            
            df = pd.read_csv(data_file)
            latest_date = df['date'].max()
            df_latest = df[df['date'] == latest_date].copy()
            
            if df_latest.empty:
                raise ValueError("No data found for latest date")

            # Apply changes from scenario
            changes = scenario.get("changes", {})
            
            # Apply demand multiplier
            if "demand_multiplier" in changes:
                dm = changes["demand_multiplier"]
                if isinstance(dm, dict):
                    default_f = float(dm.get("default", 1.0))
                    overrides = dm.get("overrides", {})
                    
                    # Apply default
                    df_latest["avg_daily_demand"] *= default_f
                    df_latest["std_daily_demand"] *= default_f
                    
                    # Apply SKU-specific overrides
                    for sku, multiplier in overrides.items():
                        mask = df_latest["sku_id"] == sku
                        df_latest.loc[mask, "avg_daily_demand"] *= multiplier
                        df_latest.loc[mask, "std_daily_demand"] *= multiplier
                        logger.info(f"Applied {multiplier}x multiplier to {sku}")

            # Apply service level changes
            if "service_level_target" in changes:
                sl_changes = changes["service_level_target"]
                if isinstance(sl_changes, dict):
                    default_sl = sl_changes.get("default")
                    if default_sl and 0 < default_sl <= 1:
                        df_latest["service_level_target"] = default_sl
                        logger.info(f"Set service level to {default_sl}")

            # Apply lead time changes
            if "lead_time_days" in changes:
                lt_changes = changes["lead_time_days"]
                if isinstance(lt_changes, dict):
                    multiplier = lt_changes.get("multiplier", 1.0)
                    df_latest["avg_lead_time_days"] *= multiplier
                    df_latest["std_lead_time_days"] *= multiplier
                    logger.info(f"Applied {multiplier}x lead time multiplier")

            return df_latest
            
        except Exception as e:
            logger.error(f"Failed to apply scenario: {e}")
            raise
    
    def _run_optimizer(self, scenario_data: pd.DataFrame) -> Dict[str, Any]:
        """Run the optimizer on scenario data"""
        try:
            # Create temporary input file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                scenario_data.to_csv(f.name, index=False)
                temp_input = f.name

            # Create output file
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_path = self.config.paths.optimization_results_dir / f"opt_results_{ts}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Run optimizer
                cmd = [
                    sys.executable, 
                    str(self.config.paths.scripts_dir / "optimizer.py"),
                    temp_input,
                    "--out", str(out_path),
                    "--round-to-lot"
                ]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=True, 
                    timeout=600
                )

                # Read results
                results_df = pd.read_csv(out_path)
                kpis = self._compute_kpis(results_df)

                return {
                    "success": True,
                    "policy": results_df.to_dict('records'),
                    "kpis": kpis,
                    "trace": [result.stdout.strip()] if result.stdout else []
                }

            finally:
                # Cleanup
                if os.path.exists(temp_input):
                    os.unlink(temp_input)

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "message": f"Optimization failed: {e}",
                "stderr": e.stderr if e.stderr else str(e)
            }
        except Exception as e:
            logger.error(f"Optimizer error: {e}")
            return {"success": False, "message": f"Unexpected error: {e}"}
    
    def _compute_kpis(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Compute KPIs from optimization results"""
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

    async def analyze_question(self, question: str) -> AnalysisResult:
        
        """Enhanced analyze_question with detailed conversation tracking"""
        if not self.team:
            return AnalysisResult(
                success=False,
                conversation_summary="AutoGen team not available - check API configuration",
                insights=["System configuration error"],
                recommendations=["Verify GROQ_API_KEY is set and AutoGen is installed"],
                execution_time=0
            )

        start_time = time.time()
        
        try:
            # Parse question into scenario
            scenario = self._parse_question_to_scenario(question)
            
            if not scenario.get("changes"):
                return AnalysisResult(
                    success=False,
                    conversation_summary="Could not parse the question into actionable scenario",
                    insights=[f"Unable to understand: '{question}'"],
                    recommendations=["Try rephrasing with specific SKU changes or parameter targets"],
                    execution_time=time.time() - start_time
                )

            # Reset team and prepare task
            await self.team.reset()
            
            # Enhanced conversation tracking with better error handling
            conversation_log = []
            message_count = 0
            
            scenario_json = json.dumps(scenario, separators=(",", ":"))
            task = TextMessage(
                content=(
                    f"ANALYZE THIS QUESTION: {question}\n\n"
                    f"SCENARIO: {scenario_json}\n\n"
                    "DataEngineer: Validate the scenario and describe transformations.\n"
                    "Optimizer: Use _run_optimization_tool() with the scenario JSON to compute optimal policies.\n"
                    "Explainer: Interpret results and provide business recommendations."
                ),
                source="user"
            )

            # Run the team with enhanced logging and error handling
            self._last_opt_result = None
            
            try:
                async for message in self.team.run_stream(task=task):
                    message_count += 1
                    try:
                        if hasattr(message, 'source') and hasattr(message, 'content'):
                            # Create formatted conversation entry with safe content extraction
                            timestamp = time.strftime("%H:%M:%S")
                            content = str(message.content)[:2000] if message.content else ""
                            formatted_message = f"**[{timestamp}] {message.source}**: {content}{'...' if len(str(message.content or '')) > 2000 else ''}"
                            conversation_log.append(formatted_message)
                            
                            # Log to console for debugging (safely)
                            logger.info(f"Agent {message.source}: {content[:200]}...")
                            
                    except Exception as log_error:
                        # Don't fail the entire process for logging errors
                        logger.warning(f"Failed to log message: {log_error}")
                        continue
                        
            except Exception as stream_error:
                logger.error(f"Stream processing error: {stream_error}")
                # Continue with whatever we have so far
            
            # Generate comprehensive conversation summary
            conversation_summary = self._generate_conversation_summary(conversation_log, scenario, question)
            
            # Get optimization results
            optimization_result = self._last_opt_result or {"success": False, "message": "No optimization result"}

            # Generate final analysis
            summary = self._generate_summary(question, optimization_result)
            insights, recommendations = self._generate_insights(optimization_result, scenario)
            
            return AnalysisResult(
                success=optimization_result.get('success', False),
                conversation_summary=summary,
                insights=insights,
                recommendations=recommendations,
                kpis=optimization_result.get('kpis'),
                policy_data=optimization_result.get('policy'),
                execution_time=time.time() - start_time,
                agent_conversations=conversation_log  # Enhanced conversation log
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return AnalysisResult(
                success=False,
                conversation_summary=f"Analysis failed: {str(e)}",
                insights=["Technical error occurred"],
                recommendations=["Check system configuration and try again"],
                execution_time=time.time() - start_time,
                agent_conversations=[f"**ERROR**: {str(e)}"]
            )
    def _generate_conversation_summary(self, conversation_log: List[str], scenario: Dict[str, Any], question: str) -> str:
        """Generate a summary of the agent conversation"""
        if not conversation_log:
            return f"No agent conversation recorded for: {question}"
        
        # Count messages per agent
        agent_counts = {}
        for msg in conversation_log:
            if "**" in msg and "**:" in msg:
                try:
                    agent_name = msg.split("**")[1].split("**:")[0].split("]")[-1].strip()
                    agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
                except:
                    continue
        
        # Create summary
        summary_parts = [
            f"AutoGen team analyzed: '{question[:80]}{'...' if len(question) > 80 else ''}'",
            f"Agent interactions: {len(conversation_log)} messages"
        ]
        
        if agent_counts:
            agent_summary = ", ".join([f"{agent}: {count}" for agent, count in agent_counts.items()])
            summary_parts.append(f"Participation: {agent_summary}")
        
        # Add scenario info
        changes = scenario.get("changes", {})
        if changes:
            change_types = list(changes.keys())
            summary_parts.append(f"Scenario changes: {', '.join(change_types)}")
        
        return " | ".join(summary_parts)    

    def _generate_summary(self, question: str, opt_result: Dict[str, Any]) -> str:
        """Generate analysis summary"""
        if opt_result.get('success'):
            kpis = opt_result.get('kpis', {})
            return (
                f"Analysis of '{question}' completed successfully. "
                f"Optimization found total annual cost of ${kpis.get('total_cost', 0):,.2f} "
                f"with {kpis.get('avg_service_level', 0):.1%} average service level "
                f"across {kpis.get('items_count', 0)} items."
            )
        else:
            return f"Analysis of '{question}' encountered issues: {opt_result.get('message', 'Unknown error')}"

    def _generate_insights(self, opt_result: Dict[str, Any], scenario: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Generate insights and recommendations"""
        insights = []
        recommendations = []
        
        if opt_result.get('success'):
            kpis = opt_result.get('kpis', {})
            
            insights.extend([
                f"Total annual inventory cost: ${kpis.get('total_cost', 0):,.2f}",
                f"Average service level achieved: {kpis.get('avg_service_level', 0):.1%}",
                f"Total safety stock required: {kpis.get('total_safety_stock', 0):,.0f} units"
            ])
            
            # Scenario-specific insights
            changes = scenario.get('changes', {})
            if changes.get('demand_multiplier', {}).get('overrides'):
                overrides = changes['demand_multiplier']['overrides']
                insights.append(f"Applied demand changes to {len(overrides)} specific SKUs")
            
            recommendations.extend([
                "Monitor implementation of new policies closely",
                "Validate with historical demand patterns",
                "Consider sensitivity analysis on key parameters"
            ])
        else:
            insights.append("Optimization failed to find feasible solution")
            recommendations.extend([
                "Check input data quality and constraints",
                "Review scenario parameters for feasibility"
            ])
        
        return insights, recommendations


# Streamlit Integration Class
class StreamlitAutoGenOrchestrator:
    """Streamlit wrapper for AutoGen OptiGuide system"""
    
    def __init__(self):
        try:
            self.core_orchestrator = AutoGenOptiGuideOrchestrator()
            self.analysis_history = []
            self.available = bool(self.core_orchestrator.team)
        except Exception as e:
            logger.error(f"Failed to initialize AutoGen orchestrator: {e}")
            self.available = False
            self.analysis_history = []

    def run_analysis(self, question: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback method for compatibility with original orchestrator"""
        return self.analyze_question(question)

    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze question and return Streamlit-friendly response"""
        if not self.available:
            return {
                'success': False,
                'question': question,
                'summary': "AutoGen system not properly configured - check GROQ_API_KEY",
                'insights': ["AutoGen agents not available"],
                'recommendations': ["Verify GROQ_API_KEY is set in environment", "Check AutoGen installation"],
                'execution_time': 0,
                'agent_conversations': [],
                'autogen_available': False
            }
        
        try:
            # Check if we're already in an async context (like Streamlit)
            try:
                # This will raise RuntimeError if no event loop is running
                loop = asyncio.get_running_loop()
                # We're in an async context, but this is a sync method
                # Use asyncio.run_coroutine_threadsafe to run in thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.core_orchestrator.analyze_question(question))
                    result = future.result()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run
                result = asyncio.run(self.core_orchestrator.analyze_question(question))
            
            # Format for Streamlit
            formatted_result = {
                'success': result.success,
                'question': question,
                'summary': result.conversation_summary,
                'insights': result.insights,
                'recommendations': result.recommendations,
                'kpis': result.kpis,
                'policy': result.policy_data,
                'execution_time': result.execution_time,
                'agent_conversations': result.agent_conversations or [],
                'autogen_available': True
            }
            
            # Add to history
            self.analysis_history.append(formatted_result)
            return formatted_result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                'success': False,
                'question': question,
                'summary': f"Analysis failed: {str(e)}",
                'insights': ["Technical error occurred"],
                'recommendations': ["Check system logs and configuration"],
                'execution_time': 0,
                'agent_conversations': [],
                'autogen_available': True
            }

# Test function
if __name__ == "__main__":
    print("Testing Fixed AutoGen OptiGuide System...")
    
    if not AUTOGEN_AVAILABLE:
        print("âŒ AutoGen not available - install with: pip install autogen-agentchat autogen-ext[openai]")
        exit(1)
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("âŒ GROQ_API_KEY not set in environment")
        print("Set it with: export GROQ_API_KEY=your_key_here")
        exit(1)
    
    print("âœ… API key found")
    
    try:
        orchestrator = StreamlitAutoGenOrchestrator()
        if orchestrator.available:
            print("âœ… AutoGen orchestrator created successfully")
            print("âœ… System ready for what-if analysis!")
            
            # Test analysis
            test_question = "What if demand for SKU-0001 increases by 50%?"
            print(f"\nðŸ§ª Testing with: {test_question}")
            
            result = orchestrator.analyze_question(test_question)
            print(f"Result: {'âœ… Success' if result['success'] else 'âŒ Failed'}")
            if result.get('summary'):
                print(f"Summary: {result['summary']}")
        else:
            print("âŒ AutoGen orchestrator not available - check configuration")
            
    except Exception as e:
        print(f"âŒ Test failed: {e}")
        import traceback
        traceback.print_exc()