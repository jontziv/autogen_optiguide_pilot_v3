#!/usr/bin/env python3
"""
Enhanced Data Display Components for OptiGuide
==============================================
Provides comprehensive data viewing capabilities for the Streamlit app,
including current data tables, optimization results, and interactive visualizations.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import hashlib
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

def _next_chart_key(base: str, fig=None) -> str:
    """
    Returns a Streamlit-safe unique key for plotly charts.
    - base: a human-readable scope (e.g., 'service_level')
    - fig: optional plotly figure; if provided, key is stable for identical figure JSON within a run
    """
    counter_key = "_plotly_seq"
    st.session_state[counter_key] = st.session_state.get(counter_key, 0) + 1

    h = ""
    if fig is not None:
        try:
            spec = json.dumps(fig.to_plotly_json(), sort_keys=True, separators=(",", ":"))
            h = "_" + hashlib.md5(spec.encode()).hexdigest()[:8]
        except Exception:
            pass

    return f"{base}_{st.session_state[counter_key]}{h}"


class DataDisplayManager:
    """Manages all data display components for the OptiGuide app"""
    
    def __init__(self):
        self.chart_colors = {
            'primary': '#00d4ff',
            'secondary': '#ff8c00', 
            'success': '#00ff88',
            'warning': '#ffa500',
            'error': '#ff4757',
            'background': 'rgba(10, 25, 47, 0.1)'
        }

    def display_current_data_overview(self, data: pd.DataFrame):
        """Display comprehensive overview of current data"""
        if data.empty:
            st.warning("No data available")
            return
        
        st.markdown("### 📊 Current Dataset Overview")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(data):,}")
        
        with col2:
            unique_skus = data['sku_id'].nunique() if 'sku_id' in data.columns else 0
            st.metric("Unique SKUs", f"{unique_skus:,}")
        
        with col3:
            unique_locations = data['location_id'].nunique() if 'location_id' in data.columns else 0
            st.metric("Unique Locations", f"{unique_locations:,}")
        
        with col4:
            date_range = f"{data['date'].min()} to {data['date'].max()}" if 'date' in data.columns else "N/A"
            st.markdown('<div class="kpi-small">', unsafe_allow_html=True)
            st.metric("Date Range", date_range)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data quality indicators
        st.markdown("#### Data Quality")
        quality_col1, quality_col2 = st.columns(2)
        
        with quality_col1:
            missing_data = data.isnull().sum().sum()
            completeness = (1 - missing_data / (len(data) * len(data.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        with quality_col2:
            latest_date = data['date'].max() if 'date' in data.columns else None
            if latest_date:
                days_old = (pd.Timestamp.now().date() - pd.to_datetime(latest_date).date()).days
                st.metric("Data Age (days)", f"{days_old}")
    
    def _plotly(self, fig, *, base_key: str, **kwargs):
        """Centralized wrapper so all plotly charts get unique keys."""
        st.plotly_chart(fig, key=_next_chart_key(base_key, fig), **kwargs)

    def display_detailed_data_table(self, data: pd.DataFrame, key_suffix: str = ""):
        """Display detailed, scrollable data table with filtering"""
        if data.empty:
            st.info("No data to display")
            return
        
        st.markdown("#### 📋 Detailed Data Table")
        
        # Filtering options
        with st.expander("🔍 Filter Options", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            # SKU filter
            if 'sku_id' in data.columns:
                with filter_col1:
                    sku_options = ['All'] + sorted(data['sku_id'].unique().tolist())
                    selected_sku = st.selectbox(
                        "Filter by SKU", 
                        sku_options, 
                        key=f"sku_filter_{key_suffix}"
                    )
            else:
                selected_sku = 'All'
            
            # Location filter
            if 'location_id' in data.columns:
                with filter_col2:
                    location_options = ['All'] + sorted(data['location_id'].unique().tolist())
                    selected_location = st.selectbox(
                        "Filter by Location", 
                        location_options, 
                        key=f"location_filter_{key_suffix}"
                    )
            else:
                selected_location = 'All'
            
            # Date filter
            if 'date' in data.columns:
                with filter_col3:
                    date_options = ['All'] + sorted(data['date'].unique().tolist(), reverse=True)
                    selected_date = st.selectbox(
                        "Filter by Date", 
                        date_options, 
                        key=f"date_filter_{key_suffix}"
                    )
            else:
                selected_date = 'All'
        
        # Apply filters
        filtered_data = data.copy()
        
        if selected_sku != 'All' and 'sku_id' in data.columns:
            filtered_data = filtered_data[filtered_data['sku_id'] == selected_sku]
        
        if selected_location != 'All' and 'location_id' in data.columns:
            filtered_data = filtered_data[filtered_data['location_id'] == selected_location]
        
        if selected_date != 'All' and 'date' in data.columns:
            filtered_data = filtered_data[filtered_data['date'] == selected_date]
        
        # Display filtered results count
        if len(filtered_data) != len(data):
            st.info(f"Showing {len(filtered_data):,} of {len(data):,} records after filtering")
        
        # Scrollable data table
        st.dataframe(
            filtered_data,
            width='stretch',  # Changed from use_container_width=True
            height=400,
            key=f"data_table_{key_suffix}"
        )
        
        # Download option
        csv_data = filtered_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            key=f"download_filtered_{key_suffix}"
        )
    
    def display_optimization_results(self, opt_results: Dict[str, Any], key_suffix: str = ""):
        """Display comprehensive optimization results"""
        if not opt_results or not opt_results.get('success'):
            st.error("No valid optimization results to display")
            return
        
        st.markdown("### 🎯 Optimization Results")
        
        # KPIs Section
        kpis = opt_results.get('kpis', {})
        if kpis:
            st.markdown("#### Key Performance Indicators")
            self._display_kpi_metrics(kpis)
        
        # Policy Results Table
        policy_data = opt_results.get('policy', [])
        if policy_data:
            st.markdown("#### 📋 Optimized Inventory Policy")
            
            policy_df = pd.DataFrame(policy_data)
            
            # Summary stats for policy
            if not policy_df.empty:
                policy_col1, policy_col2, policy_col3 = st.columns(3)
                
                with policy_col1:
                    total_safety_stock = policy_df.get('safety_stock_units', pd.Series()).sum()
                    st.metric("Total Safety Stock", f"{total_safety_stock:,.0f}")
                
                with policy_col2:
                    avg_rop = policy_df.get('rop_units', pd.Series()).mean()
                    st.metric("Average ROP", f"{avg_rop:,.0f}")
                
                with policy_col3:
                    total_daily_cost = policy_df.get('daily_holding_cost_safety_stock', pd.Series()).sum()
                    st.metric("Daily Holding Cost", f"${total_daily_cost:,.2f}")
            
            # Interactive policy table
            self.display_detailed_data_table(policy_df, key_suffix=f"policy_{key_suffix}")
            
            # Policy visualization
            if len(policy_df) > 0:
                st.markdown("#### 📈 Policy Visualization")
                self._create_policy_charts(policy_df)
    
    def _display_kpi_metrics(self, kpis: Dict[str, Any]):
        """Display KPI metrics in a formatted layout"""
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            service_level = kpis.get('avg_service_level', 0)
            st.metric(
                "Service Level", 
                f"{service_level:.1%}",
                delta=None
            )
        
        with kpi_col2:
            total_cost = kpis.get('total_cost', 0)
            st.metric(
                "Annual Cost", 
                f"${total_cost:,.0f}",
                delta=None
            )
        
        with kpi_col3:
            safety_stock = kpis.get('total_safety_stock', 0)
            st.metric(
                "Total Safety Stock", 
                f"{safety_stock:,.0f}",
                delta=None
            )
        
        with kpi_col4:
            items_count = kpis.get('items_count', 0)
            st.metric(
                "Items Optimized", 
                f"{items_count:,}",
                delta=None
            )
    
    def _create_policy_charts(self, policy_df: pd.DataFrame):
        """Create interactive charts for policy visualization"""
        if policy_df.empty:
            return
        
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Safety Stock", "Service Levels", "Costs"])
        
        with chart_tab1:
            self._create_safety_stock_chart(policy_df)
        
        with chart_tab2:
            self._create_service_level_chart(policy_df)
        
        with chart_tab3:
            self._create_cost_chart(policy_df)
    
    def _create_safety_stock_chart(self, policy_df: pd.DataFrame):
        """Create safety stock visualization"""
        if 'safety_stock_units' not in policy_df.columns or 'sku_id' not in policy_df.columns:
            st.info("Safety stock data not available for visualization")
            return
        
        fig = px.bar(
            policy_df,
            x='sku_id',
            y='safety_stock_units',
            color='location_id' if 'location_id' in policy_df.columns else None,
            title="Safety Stock by SKU",
            labels={'safety_stock_units': 'Safety Stock (Units)', 'sku_id': 'SKU'},
            color_discrete_sequence=[self.chart_colors['primary'], self.chart_colors['secondary'], self.chart_colors['success']]
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=16),
            xaxis_title_font=dict(size=12),
            yaxis_title_font=dict(size=12)
        )
        
        self._plotly(fig, base_key="safety_stock", use_container_width=True)
    
    def _create_service_level_chart(self, policy_df: pd.DataFrame):
        """Create service level visualization"""
        if 'service_level_target' not in policy_df.columns:
            st.info("Service level data not available for visualization")
            return
        
        fig = px.scatter(
            policy_df,
            x='sku_id',
            y='service_level_target',
            size='safety_stock_units' if 'safety_stock_units' in policy_df.columns else None,
            color='location_id' if 'location_id' in policy_df.columns else None,
            title="Service Level Targets by SKU",
            labels={'service_level_target': 'Service Level Target', 'sku_id': 'SKU'},
            color_discrete_sequence=[self.chart_colors['primary'], self.chart_colors['secondary'], self.chart_colors['success']]
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font=dict(size=16),
            xaxis_title_font=dict(size=12),
            yaxis_title_font=dict(size=12)
        )
        
        fig.update_yaxis(tickformat='.0%')
        
        self._plotly(fig, base_key="service_level", use_container_width=True)
    
    def _create_cost_chart(self, policy_df: pd.DataFrame):
        """Create cost visualization"""
        if 'daily_holding_cost_safety_stock' not in policy_df.columns:
            st.info("Cost data not available for visualization")
            return
        
        # Create cost breakdown chart
        cost_data = policy_df.copy()
        if 'sku_id' in cost_data.columns:
            cost_data['annual_cost'] = cost_data['daily_holding_cost_safety_stock'] * 365
            
            fig = px.pie(
                cost_data,
                values='annual_cost',
                names='sku_id',
                title="Annual Holding Cost Distribution",
                color_discrete_sequence=[self.chart_colors['primary'], self.chart_colors['secondary'], self.chart_colors['success'], self.chart_colors['warning']]
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=16)
            )
            
            self._plotly(fig, base_key="cost_pie", use_container_width=True)
    
    def display_data_trends(self, data: pd.DataFrame):
        """Display trend analysis for time-series data"""
        if data.empty or 'date' not in data.columns:
            return
        
        st.markdown("#### Data Trends")
        
        # Convert date column
        data_trends = data.copy()
        data_trends['date'] = pd.to_datetime(data_trends['date'])
        
        # Group by date for trending
        if 'avg_daily_demand' in data_trends.columns:
            trend_data = data_trends.groupby('date')['avg_daily_demand'].mean().reset_index()
            
            fig = px.line(
                trend_data,
                x='date',
                y='avg_daily_demand',
                title="Average Daily Demand Trend",
                labels={'avg_daily_demand': 'Average Daily Demand', 'date': 'Date'}
            )
            
            fig.update_traces(line_color=self.chart_colors['primary'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                title_font=dict(size=16),
                xaxis_title_font=dict(size=12),
                yaxis_title_font=dict(size=12)
            )
            
            self._plotly(fig, base_key="line", use_container_width=True)
    
    def display_data_export_options(self, data: pd.DataFrame, results: dict = None):
        """Display data export options"""
        st.markdown("#### Export Options")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if not data.empty:
                csv_data = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Export Current Data",
                    data=csv_data,
                    file_name=f"current_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )
        
        with export_col2:
            if results and results.get('policy'):
                policy_df = pd.DataFrame(results['policy'])
                policy_csv = policy_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Export Optimization Results",
                    data=policy_csv,
                    file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )
        
        with export_col3:
            if results:
                import json
                results_json = json.dumps(results, indent=2, default=str).encode('utf-8')
                st.download_button(
                    label="Export Full Analysis",
                    data=results_json,
                    file_name=f"full_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime='application/json'
                )
