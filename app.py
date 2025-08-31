#!/usr/bin/env python3
"""
Enhanced OptiGuide Streamlit Application with Fixed AutoGen Integration
======================================================================
Features conversational AI agents, natural language processing,
and real-time what-if analysis using properly configured AutoGen framework.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import hashlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from data_display_components import DataDisplayManager

# Configure page
st.set_page_config(
    page_title="OptiGuide AI - AutoGen Edition",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced modules with better error handling
@st.cache_resource
def load_modules():
    """Load required modules with comprehensive error handling"""
    modules = {}
    errors = []
    
    try:
        from config import config, validate_dependencies
        modules["config"] = config
        modules["validate_dependencies"] = validate_dependencies
    except Exception as e:
        errors.append(f"Config loading failed: {e}")
        
    try:
        from agents import OptimizationTools, MonitoringAgent
        modules["OptimizationTools"] = OptimizationTools
        modules["MonitoringAgent"] = MonitoringAgent
    except Exception as e:
        errors.append(f"Agent modules failed: {e}")
        
    # Try to import the FIXED AutoGen orchestrator
    try:
        from autogen_optiguide_system import StreamlitAutoGenOrchestrator
        modules["StreamlitAutoGenOrchestrator"] = StreamlitAutoGenOrchestrator
        
        # Test if AutoGen is actually working
        test_orchestrator = StreamlitAutoGenOrchestrator()
        modules["AUTOGEN_AVAILABLE"] = test_orchestrator.available
        
        if test_orchestrator.available:
            logger.info("✅… AutoGen orchestrator loaded and configured successfully")
        else:
            logger.warning("⚠️¸ AutoGen orchestrator loaded but not properly configured")
            errors.append("AutoGen not properly configured - check GROQ_API_KEY")
            
    except Exception as e:
        errors.append(f"AutoGen orchestrator failed: {e}")
        modules["AUTOGEN_AVAILABLE"] = False
        logger.warning(f"AutoGen orchestrator not available: {e}")
        
        # Fallback to original orchestrator
        try:
            from orchestrator import StreamlitOptiGuideOrchestrator
            modules["StreamlitAutoGenOrchestrator"] = StreamlitOptiGuideOrchestrator
            modules["AUTOGEN_AVAILABLE"] = False
            logger.info("Using fallback orchestrator")
        except Exception as e2:
            errors.append(f"Fallback orchestrator also failed: {e2}")
        
    return modules, errors

# Apply enhanced styling
def inject_enhanced_css():
    """Apply enhanced CSS styling for conversational AI interface"""
    st.markdown("""
    <style>
    /* Import futuristic fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
    
    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e6ed;
        font-family: 'Exo 2', sans-serif;
    }
    
    .kpi-small [data-testid="stMetricValue"] { 
        font-size: 0.9rem !important; 
        line-height: 1.15;
        white-space: normal;
    }
    .kpi-small [data-testid="stMetricLabel"] { 
        font-size: 0.75rem !important; 
        line-height: 1.1;
        white-space: normal;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #ffffff;
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 1rem;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Status indicators */
    .status-success { 
        color: #00ff88;
        font-weight: bold;
    }
    .status-warning { 
        color: #ffa500;
        font-weight: bold;
    }
    .status-error { 
        color: #ff4757;
        font-weight: bold;
    }
    
    /* Agent conversation styling */
    .agent-conversation {
        background: rgba(10, 25, 47, 0.6);
        border-left: 3px solid #00d4ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
    }
    
    /* Chat message styling */
    .user-message {
        background: linear-gradient(145deg, rgba(255, 140, 0, 0.15), rgba(255, 0, 128, 0.1));
        border-left: 4px solid #ff8c00;
        border-radius: 15px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    
    .agent-message {
        background: linear-gradient(145deg, rgba(0, 212, 255, 0.15), rgba(0, 149, 255, 0.1));
        border-left: 4px solid #00d4ff;
        border-radius: 15px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Session state management
def initialize_session_state():
    """Initialize session state variables"""
    default_state = {
        'conversation_history': [],
        'current_analysis': None,
        'show_agent_conversations': False,
        'analysis_counter': 0,
        'processing_question': False,
        'autogen_status_checked': False
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Data loading and caching
@st.cache_data(ttl=300)
def load_data():
    """Load and cache current dataset"""
    try:
        modules, _ = load_modules()
        if "OptimizationTools" in modules:
            return modules["OptimizationTools"].load_data()
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()
# Add this function after your existing load_data() function
@st.cache_data(ttl=300)
def load_and_display_current_data():
    """Load current data and return both DataFrame and summary stats"""
    try:
        modules, _ = load_modules()
        if "OptimizationTools" in modules:
            data = modules["OptimizationTools"].load_data()
            
            # Generate summary stats
            if not data.empty:
                stats = {
                    "total_records": len(data),
                    "unique_skus": data['sku_id'].nunique(),
                    "unique_locations": data['location_id'].nunique(),
                    "date_range": f"{data['date'].min()} to {data['date'].max()}",
                    "latest_date": data['date'].max(),
                    "avg_demand": data['avg_daily_demand'].mean() if 'avg_daily_demand' in data.columns else 0,
                    "total_inventory_value": (data['unit_cost'] * data['avg_daily_demand'] * 30).sum() if all(col in data.columns for col in ['unit_cost', 'avg_daily_demand']) else 0
                }
            else:
                stats = {}
            
            return data, stats
        return pd.DataFrame(), {}
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame(), {}

def check_autogen_status():
    """Check AutoGen configuration status"""
    import os
    
    status = {
        "autogen_installed": False,
        "groq_api_key": False,
        "client_configured": False,
        "error_message": None
    }
    
    try:
        # Check AutoGen installation
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        status["autogen_installed"] = True
    except ImportError as e:
        status["error_message"] = f"AutoGen not installed: {e}"
        return status
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key != "your_groq_api_key_here":
        status["groq_api_key"] = True
    else:
        status["error_message"] = "GROQ_API_KEY not set or using default value"
        return status
    
    # Test client creation
    try:
        client = OpenAIChatCompletionClient(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "family": "llama",
            }
        )
        status["client_configured"] = True
    except Exception as e:
        status["error_message"] = f"Client configuration failed: {e}"
    
    return status

def display_system_status():
    """Display enhanced system status with AutoGen diagnostics"""
    modules, errors = load_modules()
    autogen_status = check_autogen_status()
    
    st.markdown("### System Status")
    
    # AutoGen Status
    if autogen_status["client_configured"]:
        st.markdown('<span class="status-success">✅… AutoGen Framework: Ready</span>', unsafe_allow_html=True)
        st.markdown('<span class="status-success">✅… Groq Integration: Connected</span>', unsafe_allow_html=True)
    elif autogen_status["autogen_installed"] and autogen_status["groq_api_key"]:
        st.markdown('<span class="status-warning">⚠️¸ AutoGen Framework: Configuration Issue</span>', unsafe_allow_html=True)
        if autogen_status["error_message"]:
            st.warning(autogen_status["error_message"])
    else:
        st.markdown('<span class="status-error">❌ AutoGen Framework: Not Available</span>', unsafe_allow_html=True)
        if autogen_status["error_message"]:
            st.error(autogen_status["error_message"])
    
    # Configuration Status
    config_status = "Ready" if not errors else f"Issues ({len(errors)})"
    status_class = "status-success" if not errors else "status-warning"
    st.markdown(f'<span class="{status_class}">📊 Configuration: {config_status}</span>', unsafe_allow_html=True)
    
    # Conversations
    conversation_count = len(st.session_state.conversation_history)
    st.markdown(f"💬 Conversations: {conversation_count}")
    
    # Show setup instructions if needed
    if not autogen_status["client_configured"]:
        with st.expander("ðŸ”§ Setup Instructions", expanded=True):
            st.markdown("""
            **To enable full AutoGen capabilities:**
            
            1. **Install AutoGen packages:**
            ```bash
            pip install autogen-agentchat autogen-ext[openai]
            ```
            
            2. **Get Groq API Key:**
               - Visit [console.groq.com](https://console.groq.com/keys)
               - Create a free API key
            
            3. **Set environment variable:**
            ```bash
            export GROQ_API_KEY="your_actual_api_key_here"
            ```
            
            4. **Restart the Streamlit app**
            """)

def handle_user_question(question: str):
    """Process user question using AutoGen agents or fallback"""
    if not question.strip():
        return
    
    user_message = {
        'role': 'user',
        'content': question,
        'timestamp': datetime.now()
    }
    st.session_state.conversation_history.append(user_message)
    st.session_state.processing_question = True
    
    display_conversation_message("user", question)
    
    with st.spinner("OptiGuide agents are analyzing your question..."):
        try:
            modules, errors = load_modules()
            
            if modules.get("AUTOGEN_AVAILABLE", False):
                AutoGenClass = modules["StreamlitAutoGenOrchestrator"]
                orchestrator = AutoGenClass()
                
                # Synchronous call - let the orchestrator handle async internally
                result = orchestrator.analyze_question(question)
                result["execution_mode"] = "autogen_agents"
            else:
                # Fallback
                FallbackClass = modules["StreamlitAutoGenOrchestrator"]
                orchestrator = FallbackClass()
                result = orchestrator.run_analysis(question, {})
                result["execution_mode"] = "fallback_mode"
            
            st.session_state.current_analysis = result
            st.session_state.analysis_counter += 1
            
            assistant_message = {
                'role': 'assistant',
                'content': result.get('summary', 'Analysis completed'),
                'timestamp': datetime.now(),
                'success': result.get('success', False),
                'full_result': result
            }
            st.session_state.conversation_history.append(assistant_message)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            # Add error handling...
        finally:
            st.session_state.processing_question = False

def display_conversation_message(role: str, content: str, timestamp: datetime = None):
    """Display a conversation message with appropriate styling"""
    if timestamp is None:
        timestamp = datetime.now()
    
    time_str = timestamp.strftime("%H:%M:%S")
    
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>You ({time_str}):</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    
    elif role == "assistant":
        st.markdown(f"""
        <div class="agent-message">
            <strong>OptiGuide AI ({time_str}):</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)

def display_analysis_result(result: Dict[str, Any], key_prefix: str = ""):
    """Enhanced display for comprehensive analysis results"""
    success = result.get('success', False)
    execution_mode = result.get('execution_mode', 'unknown')
    
    # Status indicator with execution mode
    if success:
        if execution_mode == "autogen_agents":
            status_text = "✅… Analysis Complete (AutoGen Agents)"
            status_class = "status-success"
        elif execution_mode == "fallback_mode":
            status_text = "✅… Analysis Complete (Fallback Mode)"
            status_class = "status-warning"
        else:
            status_text = "✅… Analysis Complete"
            status_class = "status-success"
    else:
        status_text = "❌ Analysis Failed"
        status_class = "status-error"
    
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <span class="{status_class}">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Main results in tabs for better organization
    if success:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📋 Detailed Results", "🤖 Agent Insights", "📥 Export"])
        
        with tab1:
            # Main summary and KPIs
            if result.get('summary'):
                st.markdown(f"""
                <div style="background: rgba(16, 35, 70, 0.8); border: 1px solid rgba(0, 212, 255, 0.3); 
                            border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
                    <h4>Analysis Summary</h4>
                    {result['summary']}
                </div>
                """, unsafe_allow_html=True)
            
            # KPIs Section
            kpis = result.get('kpis', {})
            if kpis:
                st.markdown("### Key Performance Indicators")
                create_kpi_dashboard(kpis)
        
        with tab2:
            # Detailed optimization results
            policy_data = result.get('policy', [])
            if policy_data:
                st.markdown("### 📋 Optimization Policy Details")
                
                policy_df = pd.DataFrame(policy_data)
                
                # Create sub-tabs for different views
                subtab1, subtab2, subtab3 = st.tabs(["Key Metrics", "Full Policy", "Cost Analysis"])
                
                with subtab1:
                    # Show only key columns
                    key_cols = ['sku_id', 'location_id', 'service_level_target', 
                              'safety_stock_units_rounded', 'rop_units_rounded', 
                              'daily_holding_cost_safety_stock']
                    available_cols = [col for col in key_cols if col in policy_df.columns]
                    
                    if available_cols:
                        st.dataframe(
                            policy_df[available_cols],
                            width='stretch',
                            height=350
                        )
                
                with subtab2:
                    # Full policy data with filtering
                    col1, col2 = st.columns([2, 1])
                    with col2:
                        # Add simple filters
                        if 'sku_id' in policy_df.columns:
                            sku_filter = st.selectbox(
                                "Filter by SKU:",
                                ['All'] + list(policy_df['sku_id'].unique()),
                                key=f"{key_prefix}_sku_filter"
                            )
                            if sku_filter != 'All':
                                policy_df = policy_df[policy_df['sku_id'] == sku_filter]
                    
                    with col1:
                        st.dataframe(
                            policy_df,
                            width='stretch',
                            height=400
                        )
                
                with subtab3:
                    # Cost breakdown and analysis
                    if 'daily_holding_cost_safety_stock' in policy_df.columns:
                        total_daily = policy_df['daily_holding_cost_safety_stock'].sum()
                        annual_cost = total_daily * 365
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Daily Cost", f"${total_daily:,.2f}")
                            st.metric("Annual Cost", f"${annual_cost:,.2f}")
                        
                        with col2:
                            # Cost distribution
                            if len(policy_df) > 1:
                                fig = px.pie(
                                    policy_df,
                                    values='daily_holding_cost_safety_stock',
                                    names='sku_id',
                                    title="Daily Holding Cost Distribution"
                                )
                                fig.update_traces(textinfo="percent+label")
                                st.plotly_chart(fig, width='stretch')
        
        with tab3:
            # Insights and recommendations
            insights = result.get('insights', [])
            if insights:
                st.markdown("### 💡 Key Insights")
                for i, insight in enumerate(insights, 1):
                    st.markdown(f"**{i}.** {insight}")
            
            recommendations = result.get('recommendations', [])
            if recommendations:
                st.markdown("### ðŸŽ¯ Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"**{i}.** {rec}")
            
            # Show agent conversations in this tab if available
            if result.get('agent_conversations'):
                st.markdown("### 🤖 Agent Team Conversation")
                
                # Make conversations collapsible with better formatting
                with st.expander("View Detailed Agent Interactions", expanded=False):
                    for i, conversation in enumerate(result['agent_conversations']):
                        # Parse agent name for better formatting
                        if "**" in conversation and "**:" in conversation:
                            try:
                                parts = conversation.split("**:", 1)
                                agent_header = parts[0] + "**"
                                agent_content = parts[1] if len(parts) > 1 else ""
                                
                                st.markdown(f"""
                                <div style="background: rgba(0, 212, 255, 0.1); border-left: 4px solid #00d4ff; 
                                           border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                                    <strong>{agent_header}</strong><br/>
                                    {agent_content}
                                </div>
                                """, unsafe_allow_html=True)
                            except:
                                st.markdown(f"- {conversation}")
                        else:
                            st.markdown(f"- {conversation}")
        
        with tab4:
            # Export options
            st.markdown("### 📥 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result.get('policy'):
                    policy_df = pd.DataFrame(result['policy'])
                    csv_data = policy_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="ðŸ“„ Policy CSV",
                        data=csv_data,
                        file_name=f"optimization_policy_{int(time.time())}.csv",
                        mime="text/csv",
                        key=f"policy_download_{key_prefix}"
                    )
            
            with col2:
                if result.get('kpis'):
                    import json
                    kpis_json = json.dumps(result['kpis'], indent=2).encode('utf-8')
                    st.download_button(
                        label="📊 KPIs JSON",
                        data=kpis_json,
                        file_name=f"kpis_{int(time.time())}.json",
                        mime="application/json",
                        key=f"kpis_download_{key_prefix}"
                    )
            
            with col3:
                # Full analysis export
                import json
                full_result = {k: v for k, v in result.items() if k not in ['policy']}
                if result.get('policy'):
                    full_result['policy_summary'] = f"{len(result['policy'])} items optimized"
                
                full_json = json.dumps(full_result, indent=2, default=str).encode('utf-8')
                st.download_button(
                    label="📋 Full Analysis",
                    data=full_json,
                    file_name=f"full_analysis_{int(time.time())}.json",
                    mime="application/json",
                    key=f"full_download_{key_prefix}"
                )
    
    else:
        # Error display
        error_msg = result.get('error', 'Unknown error occurred')
        st.error(f"Analysis failed: {error_msg}")
        
        if result.get('recommendations'):
            st.markdown("### Suggested Actions")
            for rec in result['recommendations']:
                st.markdown(f"- {rec}")
    
    # Execution details
    exec_time = result.get('execution_time', 0)
    st.markdown(f"*Analysis completed in {exec_time:.2f} seconds using {execution_mode.replace('_', ' ')}*")

def create_kpi_dashboard(kpis: Dict[str, Any]):
    """Create interactive KPI dashboard"""
    if not kpis:
        return
        
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Service Level", f"{kpis.get('avg_service_level', 0):.1%}")
    
    with col2:
        cost = kpis.get('total_cost', 0)
        st.metric("Annual Cost", f"${cost:,.0f}")
    
    with col3:
        safety_stock = kpis.get('total_safety_stock', 0)
        st.metric("Safety Stock", f"{safety_stock:,.0f}")
    
    with col4:
        items = kpis.get('items_count', 0)
        st.metric("SKU-Locations", f"{items}")

def display_conversation_interface():
    """Enhanced conversation interface with better data display"""
    # Create tabs for better organization
    data_tab, main_tab, results_tab = st.tabs(["📊 Current Data", "🤖 AI Analysis", "📓 Results History"])
    
    with main_tab:
        st.markdown("### Conversational AI Analysis")
        
        # Show agent conversation toggle prominently
        col1, col2 = st.columns([3, 1])
        with col2:
            st.session_state.show_agent_conversations = st.toggle(
                "Show Agent Conversations", 
                value=st.session_state.show_agent_conversations,
                help="Display detailed AutoGen agent interactions"
            )
        
        # Conversation history with enhanced display
        for idx, message in enumerate(st.session_state.conversation_history):
            display_conversation_message(
                message['role'], 
                message['content'], 
                message['timestamp']
            )
            
            # Enhanced results display for assistant messages
            if message['role'] == 'assistant' and message.get('full_result'):
                result = message['full_result']
                
                # Show agent conversations if enabled
                if st.session_state.show_agent_conversations and result.get('agent_conversations'):
                    with st.expander(f"🤖 Agent Team Conversation #{idx+1}", expanded=False):
                        for conv in result['agent_conversations']:
                            st.markdown(f"""
                            <div style="background: rgba(0, 212, 255, 0.1); border-left: 3px solid #00d4ff; 
                                       border-radius: 5px; padding: 0.8rem; margin: 0.5rem 0; font-family: monospace;">
                                {conv}
                            </div>
                            """, unsafe_allow_html=True)
                
                display_analysis_result(result, key_prefix=f"msg_{idx}")
        
        # Input interface (keep your existing one)
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_question = st.text_input(
                "Ask your what-if question:",
                placeholder="What if demand for SKU-0001 increases by 50% and service level is 98%?",
                key="user_question_input",
                disabled=st.session_state.processing_question
            )
        
        with col2:
            ask_button = st.button(
                "Ask OptiGuide", 
                type="primary",
                disabled=st.session_state.processing_question or not user_question.strip()
            )
        
        if ask_button and user_question.strip():
            handle_user_question(user_question)
            st.rerun()
    
    with data_tab:
        st.markdown("### 📊 Current Dataset")
        
        # Load data with caching
        data, stats = load_and_display_current_data()
        
        if not data.empty:
            # Create data display manager instance
            display_manager = DataDisplayManager()
            
            # Show overview metrics
            display_manager.display_current_data_overview(data)
            
            # Show trends if we have time series data
            if 'date' in data.columns:
                display_manager.display_data_trends(data)
            
            # Detailed data table with filtering
            display_manager.display_detailed_data_table(data, key_suffix="current_data")

            st.markdown("#### 🎯 Optimization Results Table")

            opt_dir = Path("output_data")
            candidate_files = []

            # Prefer explicit opt_results.csv if present
            if (opt_dir / "opt_results.csv").exists():
                candidate_files = [opt_dir / "opt_results.csv"]
            else:
                # Fall back to most recent timestamped file
                timestamped = sorted(opt_dir.glob("opt_results_*.csv"), reverse=True)
                if timestamped:
                    candidate_files = [timestamped[0]]
                else:
                    # Final fallback: some repos keep 'opti_results.csv'
                    if (opt_dir / "opti_results.csv").exists():
                        candidate_files = [opt_dir / "opti_results.csv"]

            if candidate_files:
                latest_path = candidate_files[0]
                try:
                    opt_df = pd.read_csv(latest_path)
                    st.dataframe(opt_df, width='stretch', height=350)
                    st.caption(f"Showing: `{latest_path.name}`")
                    st.download_button(
                        "📥 Download optimization results (CSV)",
                        opt_df.to_csv(index=False).encode("utf-8"),
                        file_name=latest_path.name,
                        mime="text/csv",
                        key="download_opt_results_current_tab"
                    )
                except Exception as e:
                    st.warning(f"Couldn't read {latest_path.name}: {e}")
            else:
                st.info("No optimization results found yet in `output_data/`. Run an analysis to generate some.")
        else:
            st.warning("No data available")
            
            # Show data generation option
            if st.button("🛠️ Generate Sample Data", type="secondary"):
                try:
                    modules, _ = load_modules()
                    if "OptimizationTools" in modules:
                        result = modules["OptimizationTools"].run_data_generation(months=3)
                        st.success(result.get('message', 'Data generated'))
                        st.cache_data.clear()
                        st.rerun()
                except Exception as e:
                    st.error(f"Data generation failed: {e}")
    
    with results_tab:
        st.markdown("### 📋 Analysis Results History")
        
        if st.session_state.conversation_history:
            # Filter to get only assistant messages with results
            results_messages = [
                msg for msg in st.session_state.conversation_history 
                if msg['role'] == 'assistant' and msg.get('full_result', {}).get('success')
            ]
            
            if results_messages:
                # Create expandable sections for each result
                for idx, msg in enumerate(reversed(results_messages)):  # Show newest first
                    result = msg['full_result']
                    timestamp = msg['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                    
                    with st.expander(f"Analysis {len(results_messages)-idx} - {timestamp}", expanded=idx==0):
                        # Show KPIs
                        if result.get('kpis'):
                            st.markdown("**Key Performance Indicators**")
                            create_kpi_dashboard(result['kpis'])
                        
                        # Show optimization results table
                        if result.get('policy'):
                            st.markdown("**Optimization Results**")
                            policy_df = pd.DataFrame(result['policy'])
                            st.dataframe(
                                policy_df,
                                width='stretch',
                                height=300)
                            
                            # Download button
                            csv_data = policy_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv_data,
                                file_name=f"optimization_results_{timestamp.replace(':', '-').replace(' ', '_')}.csv",
                                mime="text/csv",
                                key=f"download_result_{idx}"
                            )
                        
                        # Show insights and recommendations
                        if result.get('insights'):
                            st.markdown("**Insights**")
                            for insight in result['insights']:
                                st.markdown(f"• {insight}")
                        
                        if result.get('recommendations'):
                            st.markdown("**Recommendations**")
                            for rec in result['recommendations']:
                                st.markdown(f"• {rec}")
            else:
                st.info("No successful analysis results yet. Run some what-if scenarios to see results here.")
        else:
            st.info("No analysis history yet. Start by asking a what-if question!")
def display_detailed_optimization_results(result: Dict[str, Any], key_prefix: str = ""):
    """Enhanced display for optimization results with detailed tables"""
    
    # Policy data table with enhanced display
    if result.get('policy'):
        policy_df = pd.DataFrame(result['policy'])
        
        st.markdown("#### 📋 Detailed Optimization Policy")
        
        # Create tabs for different views of the policy
        policy_tab1, policy_tab2, policy_tab3 = st.tabs(["Summary View", "Full Details", "Key Metrics"])
        
        with policy_tab1:
            # Show only key columns for summary
            summary_cols = ['sku_id', 'location_id', 'service_level_target', 
                          'safety_stock_units_rounded', 'rop_units_rounded', 
                          'daily_holding_cost_safety_stock']
            summary_cols = [col for col in summary_cols if col in policy_df.columns]
            
            if summary_cols:
                st.dataframe(
                    policy_df[summary_cols],
                    use_container_width=True,
                    height=300
                )
        
        with policy_tab2:
            # Show all columns with filtering capability
            st.dataframe(
                policy_df,
                use_container_width=True,
                height=400
            )
        
        with policy_tab3:
            # Show aggregated metrics
            if not policy_df.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_safety_stock = policy_df.get('safety_stock_units', pd.Series()).sum()
                    avg_service_level = policy_df.get('service_level_target', pd.Series()).mean()
                    st.metric("Total Safety Stock", f"{total_safety_stock:,.0f}")
                    st.metric("Avg Service Level", f"{avg_service_level:.1%}")
                
                with col2:
                    total_daily_cost = policy_df.get('daily_holding_cost_safety_stock', pd.Series()).sum()
                    annual_cost = total_daily_cost * 365
                    st.metric("Daily Holding Cost", f"${total_daily_cost:,.2f}")
                    st.metric("Annual Holding Cost", f"${annual_cost:,.2f}")
                
                with col3:
                    avg_rop = policy_df.get('rop_units', pd.Series()).mean()
                    max_rop = policy_df.get('rop_units', pd.Series()).max()
                    st.metric("Average ROP", f"{avg_rop:,.0f}")
                    st.metric("Maximum ROP", f"{max_rop:,.0f}")

def main():
    """Enhanced main application function with better data visibility"""
    # Apply styling
    inject_enhanced_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Header (keep your existing header)
    st.markdown("""
    <div class="main-header">OptiGuide AI</div>
    <div style="text-align: center; font-size: 1.2rem; color: #8892b0; margin-bottom: 2rem;">
        Conversational What-If Analysis with AutoGen Agents
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced layout with better data visibility
    col1, col2 = st.columns([3, 1])
    
    with col2:
        display_system_status()
        
        # Enhanced current data overview
        with st.expander("📊 Data Overview", expanded=True):
            data, stats = load_and_display_current_data()
            if not data.empty and stats:
                st.metric("Total Records", f"{stats['total_records']:,}")
                st.metric("Unique SKUs", f"{stats['unique_skus']:,}")
                st.metric("Unique Locations", f"{stats['unique_locations']:,}")
                
                if stats.get('latest_date'):
                    st.write(f"Latest: {stats['latest_date']}")
                
                if stats.get('avg_demand'):
                    st.write(f"Avg Daily Demand: {stats['avg_demand']:.1f}")
                    
                if stats.get('total_inventory_value'):
                    st.write(f"Est. Inventory Value: ${stats['total_inventory_value']:,.0f}")
            else:
                st.warning("No data available")
        
        # Keep your existing controls
        with st.expander("⚙️ï¸ Controls", expanded=False):
            if st.button("🛠️ Generate New Data"):
                try:
                    modules, _ = load_modules()
                    if "OptimizationTools" in modules:
                        result = modules["OptimizationTools"].run_data_generation(months=1)
                        st.success(result.get('message', 'Data generated'))
                        st.cache_data.clear()
                        st.rerun()
                except Exception as e:
                    st.error(f"Data generation failed: {e}")
        
        # Keep your existing example questions
        with st.expander("💡 Example Questions", expanded=False):
            example_questions = [
                "What if demand for SKU-0001 increases 50%?",
                "What if service level target is 98% for all items?",
                "What if lead times increase by 20%?",
                "What if we reduce safety stock by 25% for SKU-0002?",
                "Compare current vs 95% service level scenario"
            ]
            
            for question in example_questions:
                if st.button(question, key=f"example_{hash(question)}", use_container_width=True):
                    handle_user_question(question)
                    st.rerun()
    
    with col1:
        # Use the enhanced conversation interface
        display_conversation_interface()
        
        # Clear conversation button
        if st.session_state.conversation_history:
            if st.button("Clear Conversation", type="secondary"):
                st.session_state.conversation_history = []
                st.session_state.current_analysis = None
                st.rerun()

if __name__ == "__main__":
    main()