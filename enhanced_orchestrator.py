"""
Enhanced Real-time OptiGuide Orchestrator for Streamlit
======================================================
Provides real-time updates, better natural language processing,
and seamless integration with the modern Streamlit interface.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import streamlit as st

logger = logging.getLogger(__name__)

try:
    from agents import AsyncAgentOrchestrator, OptimizationTools, AUTOGEN_AVAILABLE, MonitoringAgent
    from orchestrator import OptiGuideOrchestrator, WorkflowStage, StageResult
except ImportError:
    AUTOGEN_AVAILABLE = False
    AsyncAgentOrchestrator = None
    OptiGuideOrchestrator = None

from config import safe_config_access

class AnalysisStatus(Enum):
    """Analysis execution status"""
    IDLE = "idle"
    PARSING = "parsing"
    DATA_PREP = "data_preparation"
    OPTIMIZING = "optimizing"
    VALIDATING = "validating"
    EXPLAINING = "explaining"
    COMPLETE = "complete"
    ERROR = "error"

@dataclass
class RealTimeUpdate:
    """Real-time update for Streamlit interface"""
    stage: str
    status: AnalysisStatus
    message: str
    progress: float
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class EnhancedNLProcessor:
    """Enhanced natural language processing for what-if questions"""
    
    def __init__(self):
        self.parameter_patterns = {
            'service_level': r'(?:service\s+level|fill\s+rate|availability).*?(\d+(?:\.\d+)?)\s*%',
            'lead_time': r'(?:lead\s+time|delivery\s+time|supplier).*?(\d+(?:\.\d+)?)\s*%?\s*(?:increase|decrease|change)',
            'demand': r'(?:demand|volume|sales).*?(\d+(?:\.\d+)?)\s*%?\s*(?:increase|decrease|change)',
            'cost': r'(?:cost|price).*?(\d+(?:\.\d+)?)\s*%?\s*(?:increase|decrease|change)',
            'moq': r'(?:moq|minimum\s+order|lot\s+size).*?(\d+)',
        }
        
        self.entity_patterns = {
            'sku': r'(?:sku|product|item)[-\s]*([a-zA-Z0-9-]+)',
            'location': r'(?:location|warehouse|site)[-\s]*([a-zA-Z0-9-]+)',
        }
    
    def parse_question(self, question: str) -> Dict[str, Any]:
        """Parse natural language question into structured scenario"""
        question_lower = question.lower()
        
        scenario = {
            "changes": {},
            "run_options": {
                "scenario_name": self._generate_scenario_name(question),
                "question_type": self._classify_question(question_lower)
            }
        }
        
        # Extract parameters and values
        for param, pattern in self.parameter_patterns.items():
            matches = re.finditer(pattern, question_lower)
            for match in matches:
                value = float(match.group(1))
                direction = self._extract_direction(match.group(0))
                
                if param == 'service_level':
                    if value > 1:  # Assume percentage
                        value = value / 100
                    scenario["changes"]["service_level_target"] = {"default": value}
                
                elif param == 'lead_time':
                    multiplier = 1 + (value / 100 * direction)
                    scenario["changes"]["lead_time_days"] = {"multiplier": multiplier}
                
                elif param == 'demand':
                    multiplier = 1 + (value / 100 * direction)
                    scenario["changes"]["demand_multiplier"] = {"default": multiplier, "overrides": {}}
        
        # Extract SKU/Location specific changes
        sku_demand_changes = self._extract_sku_specific_changes(question)
        if sku_demand_changes:
            if "demand_multiplier" not in scenario["changes"]:
                scenario["changes"]["demand_multiplier"] = {"default": 1.0, "overrides": {}}
            scenario["changes"]["demand_multiplier"]["overrides"].update(sku_demand_changes)
        
        return scenario
    
    def _generate_scenario_name(self, question: str) -> str:
        """Generate descriptive scenario name"""
        words = question.split()[:5]
        return "_".join(word.lower().strip("?.,!") for word in words if len(word) > 2)
    
    def _classify_question(self, question: str) -> str:
        """Classify the type of what-if question"""
        if any(term in question for term in ['service level', 'fill rate', 'availability']):
            return "service_level_analysis"
        elif any(term in question for term in ['lead time', 'supplier', 'delivery']):
            return "lead_time_analysis"
        elif any(term in question for term in ['cost', 'price', 'holding']):
            return "cost_analysis"
        elif any(term in question for term in ['demand', 'volume', 'sales']):
            return "demand_analysis"
        else:
            return "general_analysis"
    
    def _extract_direction(self, text: str) -> int:
        """Extract direction of change (positive or negative)"""
        if any(word in text for word in ['increase', 'rise', 'up', 'higher', 'more']):
            return 1
        elif any(word in text for word in ['decrease', 'drop', 'fall', 'lower', 'less', 'reduce']):
            return -1
        return 1  # Default to increase
    
    def _extract_sku_specific_changes(self, question: str) -> Dict[str, float]:
        """Extract SKU-specific demand changes"""
        changes = {}
        
        # Pattern: "SKU-0001 increases 90%" or "demand for SKU-0001 increases by 90%"
        sku_patterns = [
            r'(?:sku[-\s]*(\w+)).*?(?:increase|rise)s?.*?(\d+)\s*%',
            r'(?:demand\s+for\s+sku[-\s]*(\w+)).*?(?:increase|rise)s?.*?(\d+)\s*%',
            r'(?:sku[-\s]*(\w+)).*?(?:decrease|drop|fall)s?.*?(\d+)\s*%',
            r'(?:demand\s+for\s+sku[-\s]*(\w+)).*?(?:decrease|drop|fall)s?.*?(\d+)\s*%'
        ]
        
        for pattern in sku_patterns:
            matches = re.finditer(pattern, question.lower())
            for match in matches:
                sku = f"SKU-{match.group(1)}" if not match.group(1).startswith('SKU') else match.group(1).upper()
                percentage = float(match.group(2))
                
                # Determine direction
                is_decrease = any(word in match.group(0) for word in ['decrease', 'drop', 'fall'])
                multiplier = 1 + (percentage / 100 * (-1 if is_decrease else 1))
                changes[sku] = multiplier
        
        return changes

class StreamlitEnhancedOrchestrator:
    """Enhanced orchestrator with real-time Streamlit integration"""
    
    def __init__(self):
        self.core_orchestrator = OptiGuideOrchestrator() if OptiGuideOrchestrator else None
        self.nl_processor = EnhancedNLProcessor()
        self.tools = OptimizationTools() if OptimizationTools else None
        self.monitoring_agent = MonitoringAgent() if 'MonitoringAgent' in globals() else None
        
        # Callback for real-time updates
        self.update_callback: Optional[Callable[[RealTimeUpdate], None]] = None
        
    def set_update_callback(self, callback: Callable[[RealTimeUpdate], None]):
        """Set callback function for real-time updates"""
        self.update_callback = callback
    
    def _send_update(self, stage: str, status: AnalysisStatus, message: str, 
                     progress: float, data: Optional[Dict] = None, error: Optional[str] = None):
        """Send real-time update"""
        if self.update_callback:
            update = RealTimeUpdate(stage, status, message, progress, data, error)
            self.update_callback(update)
    
    def run_analysis(self, question: str, scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run enhanced analysis with real-time updates"""
        start_time = time.time()
        
        # Phase 1: Natural Language Processing
        self._send_update("NL Processing", AnalysisStatus.PARSING, "Processing your question...", 0.1)
        
        try:
            if not scenario or not scenario.get("changes"):
                parsed_scenario = self.nl_processor.parse_question(question)
                scenario = parsed_scenario
            
            self._send_update("NL Processing", AnalysisStatus.PARSING, 
                            f"Identified scenario: {scenario.get('run_options', {}).get('scenario_name', 'analysis')}", 0.2)
            
        except Exception as e:
            self._send_update("NL Processing", AnalysisStatus.ERROR, 
                            f"Failed to parse question: {str(e)}", 0.2, error=str(e))
            return {"success": False, "error": f"Question parsing failed: {e}", "execution_time": time.time() - start_time}
        
        # Phase 2: Data Preparation
        self._send_update("Data Preparation", AnalysisStatus.DATA_PREP, "Loading and transforming data...", 0.3)
        
        try:
            if not self.tools:
                raise RuntimeError("Optimization tools not available")
            
            # Load base data
            base_data = self.tools.load_data()
            if base_data.empty:
                raise ValueError("No data available for analysis")
            
            self._send_update("Data Preparation", AnalysisStatus.DATA_PREP, 
                            f"Loaded {len(base_data)} records", 0.4)
            
            # Apply scenario transformations
            scenario_data = self.tools.apply_scenario_delta(scenario)
            
            self._send_update("Data Preparation", AnalysisStatus.DATA_PREP, 
                            f"Applied transformations to {len(scenario_data)} records", 0.5)
            
        except Exception as e:
            self._send_update("Data Preparation", AnalysisStatus.ERROR, 
                            f"Data preparation failed: {str(e)}", 0.5, error=str(e))
            return {"success": False, "error": f"Data preparation failed: {e}", "execution_time": time.time() - start_time}
        
        # Phase 3: Optimization
        self._send_update("Optimization", AnalysisStatus.OPTIMIZING, "Running optimization model...", 0.6)
        
        try:
            opt_results = self.tools.run_optimizer(scenario_data, persist=True)
            
            if not opt_results.get('success'):
                raise RuntimeError(opt_results.get('message', 'Optimization failed'))
            
            self._send_update("Optimization", AnalysisStatus.OPTIMIZING, 
                            f"Optimization complete - {len(opt_results.get('policy', []))} policies generated", 0.7)
            
        except Exception as e:
            self._send_update("Optimization", AnalysisStatus.ERROR, 
                            f"Optimization failed: {str(e)}", 0.7, error=str(e))
            return {"success": False, "error": f"Optimization failed: {e}", "execution_time": time.time() - start_time}
        
        # Phase 4: Policy Validation
        self._send_update("Validation", AnalysisStatus.VALIDATING, "Validating optimization results...", 0.8)
        
        try:
            validation_results = self._validate_policy(opt_results.get('policy', []))
            self._send_update("Validation", AnalysisStatus.VALIDATING, 
                            f"Validation complete - {validation_results.get('feasible_items', 0)} items validated", 0.85)
            
        except Exception as e:
            logger.warning(f"Policy validation failed: {e}")
            validation_results = {"total_items": len(opt_results.get('policy', [])), "risk_flags": []}
        
        # Phase 5: Explanation Generation
        self._send_update("Explanation", AnalysisStatus.EXPLAINING, "Generating insights and recommendations...", 0.9)
        
        try:
            insights, recommendations = self._generate_explanation(question, scenario, opt_results)
            
            self._send_update("Explanation", AnalysisStatus.EXPLAINING, 
                            f"Generated {len(insights)} insights and {len(recommendations)} recommendations", 0.95)
            
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            insights = ["Analysis completed successfully"]
            recommendations = ["Review results and validate with domain experts"]
        
        # Phase 6: Monitoring Alerts
        alerts = []
        if self.monitoring_agent:
            try:
                monitoring_result = self.monitoring_agent.check_latest()
                if monitoring_result.get('success') and monitoring_result.get('alerts'):
                    alerts = monitoring_result['alerts']
            except Exception as e:
                logger.warning(f"Monitoring check failed: {e}")
        
        # Compile final results
        execution_time = time.time() - start_time
        
        final_results = {
            "success": True,
            "question": question,
            "scenario": scenario,
            "optimization_results": opt_results,
            "policy_validation": validation_results,
            "insights": insights,
            "recommendations": recommendations,
            "monitoring_alerts": alerts,
            "execution_time": execution_time,
            "timestamp": time.time(),
            "workflow_type": "enhanced_optiguide"
        }
        
        # Add convenience fields for UI
        if opt_results.get('kpis'):
            final_results['kpis'] = opt_results['kpis']
        if opt_results.get('policy'):
            final_results['policy'] = opt_results['policy']
        
        self._send_update("Complete", AnalysisStatus.COMPLETE, 
                        f"Analysis complete in {execution_time:.2f}s", 1.0, data=final_results)
        
        return final_results
    
    def _validate_policy(self, policy_data: List[Dict]) -> Dict[str, Any]:
        """Enhanced policy validation"""
        validation_results = {
            "total_items": len(policy_data),
            "feasible_items": 0,
            "risk_flags": [],
            "constraint_violations": [],
            "performance_metrics": {}
        }
        
        if not policy_data:
            return validation_results
        
        total_cost = 0
        high_cost_items = []
        
        for item in policy_data:
            feasible = True
            
            # Basic feasibility checks
            safety_stock = item.get('safety_stock_units', 0)
            rop = item.get('rop_units', 0)
            daily_demand = item.get('avg_daily_demand', 1)
            service_level = item.get('service_level_target', 0.95)
            
            if safety_stock < 0:
                validation_results["constraint_violations"].append(
                    f"Negative safety stock for {item.get('sku_id')}"
                )
                feasible = False
            
            if rop < daily_demand * 3:  # Less than 3 days of demand
                validation_results["risk_flags"].append(
                    f"Very low ROP for {item.get('sku_id')} - potential stockout risk"
                )
            
            if service_level > 0.995:
                validation_results["risk_flags"].append(
                    f"Extremely high service level ({service_level:.1%}) for {item.get('sku_id')}"
                )
            
            # Cost analysis
            daily_holding_cost = item.get('daily_holding_cost_safety_stock', 0)
            total_cost += daily_holding_cost * 365
            
            if daily_holding_cost > 10:  # Threshold for high-cost items
                high_cost_items.append(item.get('sku_id'))
            
            if feasible:
                validation_results["feasible_items"] += 1
        
        # Performance metrics
        validation_results["performance_metrics"] = {
            "total_annual_holding_cost": total_cost,
            "average_service_level": sum(item.get('service_level_target', 0.95) for item in policy_data) / len(policy_data),
            "high_cost_items": high_cost_items,
            "feasibility_rate": validation_results["feasible_items"] / len(policy_data) if policy_data else 0
        }
        
        return validation_results
    
    def _generate_explanation(self, question: str, scenario: Dict, opt_results: Dict) -> tuple[List[str], List[str]]:
        """Generate insights and recommendations"""
        kpis = opt_results.get('kpis', {})
        policy_data = opt_results.get('policy', [])
        
        insights = []
        recommendations = []
        
        # Cost insights
        total_cost = kpis.get('total_cost', 0)
        if total_cost > 0:
            insights.append(f"Total annual inventory holding cost: ${total_cost:,.2f}")
        
        # Service level insights
        avg_service_level = kpis.get('avg_service_level', 0)
        if avg_service_level > 0:
            insights.append(f"Average service level achieved: {avg_service_level:.1%}")
            
            if avg_service_level > 0.98:
                recommendations.append("Consider cost-benefit analysis for service levels above 98%")
            elif avg_service_level < 0.90:
                recommendations.append("Service levels below 90% indicate potential stockout risk")
        
        # Safety stock insights
        total_safety_stock = kpis.get('total_safety_stock', 0)
        if total_safety_stock > 0:
            insights.append(f"Total safety stock required: {total_safety_stock:,.0f} units")
            
            if total_safety_stock > 1000:
                recommendations.append("High safety stock levels detected - review demand variability and lead time reliability")
        
        # Scenario-specific insights
        changes = scenario.get('changes', {})
        if changes.get('service_level_target'):
            target = changes['service_level_target'].get('default')
            if target:
                insights.append(f"Service level target set to {target:.1%}")
        
        if changes.get('lead_time_days'):
            multiplier = changes['lead_time_days'].get('multiplier', 1.0)
            if multiplier != 1.0:
                change_pct = (multiplier - 1.0) * 100
                insights.append(f"Lead times adjusted by {change_pct:+.1f}%")
        
        if changes.get('demand_multiplier'):
            demand_changes = changes['demand_multiplier']
            if isinstance(demand_changes, dict):
                overrides = demand_changes.get('overrides', {})
                if overrides:
                    insights.append(f"Demand changes applied to {len(overrides)} specific SKUs")
        
        # General recommendations
        recommendations.extend([
            "Monitor implementation of new inventory policies closely",
            "Validate results with historical demand patterns",
            "Consider sensitivity analysis on key parameters",
            "Review supplier performance and lead time reliability"
        ])
        
        return insights, recommendations

# Streamlit integration functions
def create_progress_display(current_stage: str, all_stages: List[str], progress: float):
    """Create progress display for Streamlit"""
    progress_bar = st.progress(progress)
    
    # Stage indicators
    cols = st.columns(len(all_stages))
    for i, (col, stage) in enumerate(zip(cols, all_stages)):
        with col:
            if stage == current_stage:
                st.markdown(f"🔄 **{stage}**")
            elif i < all_stages.index(current_stage):
                st.markdown(f"✅ {stage}")
            else:
                st.markdown(f"⏳ {stage}")
    
    return progress_bar

def display_real_time_analysis(orchestrator: StreamlitEnhancedOrchestrator, question: str):
    """Display real-time analysis with updates"""
    # Create containers for different parts of the display
    status_container = st.container()
    progress_container = st.container()
    results_container = st.container()
    
    # Track updates
    updates = []
    
    def update_callback(update: RealTimeUpdate):
        updates.append(update)
        
        with status_container:
            st.markdown(f"**Current Stage:** {update.stage}")
            st.markdown(f"**Status:** {update.message}")
            
            if update.error:
                st.error(f"Error: {update.error}")
        
        with progress_container:
            st.progress(update.progress)
    
    # Set callback and run analysis
    orchestrator.set_update_callback(update_callback)
    
    with st.spinner("Running analysis..."):
        results = orchestrator.run_analysis(question)
    
    # Display final results
    with results_container:
        if results.get('success'):
            st.success("Analysis completed successfully!")
            
            # Display KPIs
            if results.get('kpis'):
                st.markdown("### Key Performance Indicators")
                kpis = results['kpis']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Service Level", f"{kpis.get('avg_service_level', 0):.1%}")
                with col2:
                    st.metric("Total Cost", f"${kpis.get('total_cost', 0):,.2f}")
                with col3:
                    st.metric("Safety Stock", f"{kpis.get('total_safety_stock', 0):,.0f}")
                with col4:
                    st.metric("Items", f"{kpis.get('items_count', 0)}")
            
            # Display insights
            if results.get('insights'):
                st.markdown("### Insights")
                for insight in results['insights']:
                    st.markdown(f"• {insight}")
            
            # Display recommendations
            if results.get('recommendations'):
                st.markdown("### Recommendations")
                for rec in results['recommendations']:
                    st.markdown(f"• {rec}")
        
        else:
            st.error(f"Analysis failed: {results.get('error', 'Unknown error')}")
    
    return results

# Global instance
enhanced_orchestrator = StreamlitEnhancedOrchestrator()