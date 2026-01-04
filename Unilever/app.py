import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Unilever Supply Chain Control Dashboard",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .highlight-box {
        background-color: #F0F9FF;
        border-left: 4px solid #1E3A8A;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌾 Unilever Supply Chain Control Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-header">Integrated Strategy: Farmer Partnerships + Strategic Storage</h3>', unsafe_allow_html=True)

# Initialize session state for user inputs
if 'annual_need' not in st.session_state:
    st.session_state.annual_need = 20000  # tons
if 'farmer_price' not in st.session_state:
    st.session_state.farmer_price = 62500  # PKR/MT (2500/maund)
if 'storage_model' not in st.session_state:
    st.session_state.storage_model = "rental"

# Data generation functions
@st.cache_data
def generate_market_price_data():
    """Generate seasonal wheat price data"""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Harvest months (Apr-May): low prices
    # Lean months (Sep-Feb): high prices
    prices = {
        'Harvest Season (Low)': [0, 0, 0, 90000, 85000, 0, 0, 0, 0, 0, 0, 0],
        'Current Market': [110000, 112000, 108000, 95000, 92000, 96000, 
                          100000, 105000, 115000, 118000, 116000, 112000],
        'Our Target (Farmers)': [0, 0, 0, 62500, 62500, 0, 0, 0, 0, 0, 0, 0]
    }
    
    df = pd.DataFrame(prices, index=months)
    df['Month'] = df.index
    return df

@st.cache_data
def calculate_storage_costs(model, capacity_tons):
    """Calculate storage costs for different models"""
    costs = {
        'rental': {
            'capex': 0,
            'opex_per_ton_month': 5600,
            'description': 'Third-party silo rental'
        },
        'ownership': {
            'capex': 153460000,  # PKR for 10,000 MT
            'opex_per_ton_month': 1500,
            'description': 'Own silo with in-house operations'
        },
        'hybrid': {
            'capex': 153460000,  # PKR for 10,000 MT
            'opex_per_ton_month': 2500,
            'description': 'Own silo with outsourced operations'
        }
    }
    
    model_data = costs[model]
    annual_opex = capacity_tons * model_data['opex_per_ton_month'] * 6  # 6 months storage
    
    return {
        'capex': model_data['capex'],
        'annual_opex': annual_opex,
        'description': model_data['description'],
        'cost_per_ton': model_data['opex_per_ton_month']
    }

@st.cache_data
def calculate_integrated_savings(farmer_price, annual_need, storage_model, storage_capacity_percent):
    """Calculate savings from integrated farmer + storage strategy"""
    
    # Market prices
    harvest_price = 90000  # PKR/MT in April
    lean_price = 115000    # PKR/MT in October
    avg_market_price = 105000  # PKR/MT annual average
    
    # Storage capacity
    storage_capacity = annual_need * (storage_capacity_percent / 100)
    
    # Storage costs
    storage_costs = calculate_storage_costs(storage_model, storage_capacity)
    
    # Farmer integration savings
    farmer_savings_per_ton = avg_market_price - farmer_price
    
    # Storage arbitrage savings
    storage_arbitrage_per_ton = lean_price - harvest_price
    storage_net_saving_per_ton = storage_arbitrage_per_ton - (storage_costs['cost_per_ton'] * 6)
    
    # Combined savings
    combined_savings_per_ton = farmer_savings_per_ton + storage_net_saving_per_ton
    
    # Annual totals
    total_farmer_savings = farmer_savings_per_ton * annual_need
    total_storage_savings = storage_net_saving_per_ton * storage_capacity
    total_combined_savings = total_farmer_savings + total_storage_savings
    
    # Synergy effect
    synergy = total_combined_savings - (total_farmer_savings + total_storage_savings)
    
    return {
        'farmer_price': farmer_price,
        'market_price': avg_market_price,
        'farmer_savings_per_ton': farmer_savings_per_ton,
        'storage_net_saving_per_ton': storage_net_saving_per_ton,
        'combined_savings_per_ton': combined_savings_per_ton,
        'total_farmer_savings': total_farmer_savings,
        'total_storage_savings': total_storage_savings,
        'total_combined_savings': total_combined_savings,
        'synergy': synergy,
        'storage_capacity': storage_capacity,
        'storage_costs': storage_costs
    }

# ========== MAIN DASHBOARD ==========

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Strategy Parameters")
    
    # Annual need
    st.session_state.annual_need = st.number_input(
        "Annual Wheat Need (Tons)",
        min_value=5000,
        max_value=50000,
        value=st.session_state.annual_need,
        step=1000
    )
    
    # Farmer price
    st.session_state.farmer_price = st.number_input(
        "Farm-Gate Wheat Price (PKR/MT)",
        min_value=50000,
        max_value=80000,
        value=st.session_state.farmer_price,
        step=1000,
        help="Target: PKR 62,500/MT (PKR 2,500/maund)"
    )
    
    # Storage model
    st.session_state.storage_model = st.selectbox(
        "Storage Model",
        ["rental", "ownership", "hybrid"],
        format_func=lambda x: {
            "rental": "Silo Rental",
            "ownership": "Silo Ownership",
            "hybrid": "Hybrid Model"
        }[x]
    )
    
    # Storage capacity
    storage_capacity_percent = st.slider(
        "Storage Capacity (% of Annual Need)",
        min_value=10,
        max_value=80,
        value=50,
        step=5,
        help="How much of annual need to store from harvest season"
    )
    
    # Calculate button
    calculate = st.button("Calculate Strategy Impact", type="primary")

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", 
    "💰 Financial Impact", 
    "📈 Strategy Comparison", 
    "🔄 Implementation Plan"
])

with tab1:
    st.markdown("### The Integrated Strategy: Why Both Components Are Essential")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🌾 Farmer Integration ALONE
        **Problem:** 
        - Buy cheap wheat in April (harvest)
        - Must use immediately (no storage)
        - Back to market prices in lean season
        - **Result: Integration fails**
        
        **Solution needs:** Storage capacity
        """)
        
        st.markdown("""
        #### 🏗️ Storage ALONE  
        **Problem:**
        - Store market wheat (expensive)
        - Limited savings (only arbitrage)
        - Still dependent on mandi
        - **Result: Marginal benefit**
        
        **Solution needs:** Cheap wheat source
        """)
    
    with col2:
        st.markdown("""
        #### 🎯 COMBINED STRATEGY
        **How it works:**
        1. **Buy from farmers** in April @ PKR 62,500/MT
        2. **Store in silos** for 6 months
        3. **Release gradually** to mill
        4. **Avoid market prices** @ PKR 115,000/MT
        
        **Result:**
        - Stable low cost year-round
        - Complete supply chain control
        - Maximum savings from BOTH strategies
        """)
    
    # Visual: The missing link
    st.markdown("---")
    st.markdown("### The Missing Link Diagram")
    
    fig = go.Figure()
    
    # Add nodes
    fig.add_trace(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[
                "Farmers\n(April Harvest)", 
                "Without Storage\n(Must sell)", 
                "With Storage\n(Hold wheat)",
                "Market Prices\n(Sep-Feb)",
                "Our Mill\n(Year-round)",
                "High Cost\n(PKR 115K/MT)",
                "Low Cost\n(PKR 62.5K/MT)"
            ],
            color=[
                "#4CAF50", "#FF9800", "#2196F3", 
                "#F44336", "#9C27B0", "#F44336", "#4CAF50"
            ]
        ),
        link=dict(
            source=[0, 0, 1, 2, 3, 4, 4],
            target=[1, 2, 3, 4, 4, 5, 6],
            value=[1, 1, 1, 1, 1, 0.5, 0.5],
            color=[
                "rgba(255, 152, 0, 0.3)", "rgba(33, 150, 243, 0.3)",
                "rgba(244, 67, 54, 0.3)", "rgba(156, 39, 176, 0.3)",
                "rgba(244, 67, 54, 0.3)", "rgba(244, 67, 54, 0.6)",
                "rgba(76, 175, 80, 0.6)"
            ]
        )
    ))
    
    fig.update_layout(
        title_text="How Storage Enables Farmer Integration",
        font_size=12,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### Financial Impact Analysis")
    
    # Calculate savings
    results = calculate_integrated_savings(
        st.session_state.farmer_price,
        st.session_state.annual_need,
        st.session_state.storage_model,
        storage_capacity_percent
    )
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Annual Savings (Combined)",
            f"PKR {results['total_combined_savings']/1_000_000:,.1f}M",
            f"PKR {results['synergy']/1_000_000:,.1f}M synergy"
        )
    
    with col2:
        st.metric(
            "Cost per Ton (Our Strategy)",
            f"PKR {results['farmer_price']:,.0f}",
            f"PKR {results['market_price'] - results['farmer_price']:,.0f} below market"
        )
    
    with col3:
        savings_percent = ((results['market_price'] - results['farmer_price']) / results['market_price']) * 100
        st.metric(
            "Cost Reduction",
            f"{savings_percent:.1f}%",
            "vs. market average"
        )
    
    # Storage costs breakdown
    st.markdown("---")
    st.markdown("#### Storage Cost Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Storage Capacity",
            f"{results['storage_capacity']:,.0f} Tons",
            f"{storage_capacity_percent}% of annual need"
        )
    
    with col2:
        model_name = {
            "rental": "Silo Rental",
            "ownership": "Silo Ownership",
            "hybrid": "Hybrid Model"
        }[st.session_state.storage_model]
        
        st.metric(
            "Storage Model",
            model_name,
            results['storage_costs']['description']
        )
    
    with col3:
        st.metric(
            "Annual Storage Cost",
            f"PKR {results['storage_costs']['annual_opex']/1_000_000:,.1f}M",
            f"PKR {results['storage_costs']['cost_per_ton']:,.0f}/ton/month"
        )
    
    # Savings breakdown chart
    st.markdown("---")
    st.markdown("#### Savings Breakdown")
    
    savings_data = pd.DataFrame({
        'Component': ['Farmer Integration', 'Storage Arbitrage', 'Synergy Effect'],
        'Value (PKR Million)': [
            results['total_farmer_savings']/1_000_000,
            results['total_storage_savings']/1_000_000,
            results['synergy']/1_000_000
        ],
        'Color': ['#4CAF50', '#2196F3', '#9C27B0']
    })
    
    fig = px.bar(
        savings_data,
        x='Component',
        y='Value (PKR Million)',
        color='Color',
        color_discrete_map="identity",
        title="Where Savings Come From"
    )
    
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Strategy Comparison")
    
    # Compare all three strategies
    strategies = ['Current Model', 'Farmer Only', 'Storage Only', 'Combined']
    
    # Calculate for each strategy
    comparison_data = []
    
    # Current Model (baseline)
    baseline_cost = 105000 * st.session_state.annual_need
    
    # Farmer Only (no storage)
    farmer_only_cost = (st.session_state.farmer_price * 0.5 * st.session_state.annual_need) + \
                       (105000 * 0.5 * st.session_state.annual_need)
    
    # Storage Only (market wheat)
    storage_cost = calculate_storage_costs("rental", st.session_state.annual_need * 0.5)
    storage_only_cost = (90000 * 0.5 * st.session_state.annual_need) + \
                        (105000 * 0.5 * st.session_state.annual_need) + \
                        storage_cost['annual_opex']
    
    # Combined
    combined_cost = (st.session_state.farmer_price * 0.5 * st.session_state.annual_need) + \
                    (st.session_state.farmer_price * 0.5 * st.session_state.annual_need) + \
                    results['storage_costs']['annual_opex']
    
    # Populate data
    for i, (strategy, cost) in enumerate([
        ('Current Model', baseline_cost),
        ('Farmer Only', farmer_only_cost),
        ('Storage Only', storage_only_cost),
        ('Combined', combined_cost)
    ]):
        savings = baseline_cost - cost
        savings_percent = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        comparison_data.append({
            'Strategy': strategy,
            'Annual Cost (PKR M)': cost / 1_000_000,
            'Savings (PKR M)': savings / 1_000_000,
            'Savings %': savings_percent,
            'Risk Level': ['High', 'High', 'Medium', 'Low'][i],
            'Control Level': ['Low', 'Medium', 'Medium', 'High'][i]
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.dataframe(
        df_comparison,
        column_config={
            "Annual Cost (PKR M)": st.column_config.NumberColumn(format="%.1f"),
            "Savings (PKR M)": st.column_config.NumberColumn(format="%.1f"),
            "Savings %": st.column_config.NumberColumn(format="%.1f%%"),
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Visual comparison
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Annual Cost',
        x=df_comparison['Strategy'],
        y=df_comparison['Annual Cost (PKR M)'],
        marker_color='#FF6B6B'
    ))
    
    fig.add_trace(go.Bar(
        name='Savings',
        x=df_comparison['Strategy'],
        y=df_comparison['Savings (PKR M)'],
        marker_color='#4ECDC4'
    ))
    
    fig.update_layout(
        barmode='group',
        title="Cost Comparison of Different Strategies",
        yaxis_title="PKR Million",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Decision matrix
    st.markdown("---")
    st.markdown("#### Strategic Decision Matrix")
    
    decision_data = {
        'Factor': ['Cost Advantage', 'Supply Security', 'Quality Control', 
                   'Strategic Power', 'Implementation Risk', 'Flexibility'],
        'Current Model': ['Low', 'Low', 'Low', 'Low', 'Low', 'High'],
        'Farmer Only': ['Medium', 'Medium', 'High', 'Medium', 'High', 'Medium'],
        'Storage Only': ['Medium', 'High', 'Medium', 'High', 'Medium', 'Medium'],
        'Combined': ['High', 'High', 'High', 'High', 'Low', 'Low']
    }
    
    st.dataframe(pd.DataFrame(decision_data), hide_index=True, use_container_width=True)

with tab4:
    st.markdown("### Implementation Roadmap")
    
    # Phase-based implementation
    phases = [
        {
            'phase': 'Phase 1: Pilot (Months 1-6)',
            'activities': [
                'Rent 3,000 MT silo capacity',
                'Onboard 100-acre farmer pilot',
                'Test storage operations',
                'Validate wheat quality preservation',
                'Calculate real savings'
            ],
            'investment': 'PKR 7.2M',
            'expected_outcome': 'Proof of concept, 20% cost reduction'
        },
        {
            'phase': 'Phase 2: Scale (Months 7-18)',
            'activities': [
                'Expand to 1,000 acres farmers',
                'Increase storage to 10,000 MT',
                'Implement quality monitoring',
                'Develop farmer cluster system',
                'Begin mill integration'
            ],
            'investment': 'PKR 50M',
            'expected_outcome': '40% cost reduction, supply security'
        },
        {
            'phase': 'Phase 3: Optimize (Months 19-36)',
            'activities': [
                'Decide: continue rental or build own silos',
                'Full farmer integration (5,000+ acres)',
                'Advanced quality control systems',
                'Complete supply chain digitization',
                'Strategic partnerships with mills'
            ],
            'investment': 'PKR 150M (if ownership)',
            'expected_outcome': '50%+ cost reduction, market leadership'
        }
    ]
    
    for phase in phases:
        with st.expander(f"📋 {phase['phase']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("**Key Activities:**")
                for activity in phase['activities']:
                    st.markdown(f"• {activity}")
            
            with col2:
                st.metric("Investment", phase['investment'])
                st.metric("Expected Outcome", phase['expected_outcome'])
    
    # Gantt chart visualization
    st.markdown("---")
    st.markdown("#### Implementation Timeline")
    
    gantt_data = pd.DataFrame([
        {'Task': 'Farmer Pilot', 'Start': '2024-04-01', 'Finish': '2024-09-30', 'Phase': 'Pilot'},
        {'Task': 'Storage Rental', 'Start': '2024-04-01', 'Finish': '2025-03-31', 'Phase': 'Pilot'},
        {'Task': 'Farmer Expansion', 'Start': '2024-10-01', 'Finish': '2025-09-30', 'Phase': 'Scale'},
        {'Task': 'Storage Decision', 'Start': '2025-01-01', 'Finish': '2025-03-31', 'Phase': 'Scale'},
        {'Task': 'Mill Integration', 'Start': '2025-04-01', 'Finish': '2026-03-31', 'Phase': 'Scale'},
        {'Task': 'Full Integration', 'Start': '2026-04-01', 'Finish': '2027-03-31', 'Phase': 'Optimize'}
    ])
    
    gantt_data['Start'] = pd.to_datetime(gantt_data['Start'])
    gantt_data['Finish'] = pd.to_datetime(gantt_data['Finish'])
    
    fig = px.timeline(
        gantt_data, 
        x_start="Start", 
        x_end="Finish", 
        y="Task",
        color="Phase",
        title="3-Year Implementation Roadmap"
    )
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=300)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Decision points
    st.markdown("---")
    st.markdown("#### Key Decision Points")
    
    decision_points = [
        {
            'time': 'Month 6',
            'decision': 'Continue or stop pilot',
            'criteria': 'Achieved 20% savings? Farmer cooperation?',
            'options': ['Stop: minimal loss', 'Continue to Phase 2']
        },
        {
            'time': 'Month 12',
            'decision': 'Storage model choice',
            'criteria': 'Storage operations successful? Cost savings proven?',
            'options': ['Continue rental', 'Move to ownership', 'Try hybrid model']
        },
        {
            'time': 'Month 24',
            'decision': 'Full commitment',
            'criteria': '50%+ savings achieved? Supply secure?',
            'options': ['Full scale-up', 'Maintain current level', 'Adjust strategy']
        }
    ]
    
    for dp in decision_points:
        with st.expander(f"⏰ {dp['time']}: {dp['decision']}"):
            st.markdown(f"**Decision Criteria:** {dp['criteria']}")
            st.markdown("**Options:**")
            for option in dp['options']:
                st.markdown(f"• {option}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Unilever Pakistan Supply Chain Transformation Initiative</strong></p>
    <p>Integrated Farmer & Storage Strategy Dashboard | Version 1.0</p>
</div>
""", unsafe_allow_html=True)
