import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- CLEAR EVERYTHING ---
st.cache_data.clear()

# --- SIMPLE FIXED STORAGE CALCULATOR ---

def enhanced_storage_calculator():
    """Fixed calculator using REAL values from Unilever report."""
    
    st.subheader("🏦 Strategic Storage Options Calculator")
    st.caption("Based on Unilever Report Data: Harvest (PKR 90-95/kg) vs Lean (PKR 110-115/kg)")
    
    # Show correct values upfront
    st.info("""
    **Data from Unilever Report (Page 5.1):**
    - Harvest Season (April-May): **PKR 90,000-95,000 per ton** (PKR 90-95 per kg)
    - Lean Season (Oct-Dec): **PKR 110,000-115,000 per ton** (PKR 110-115 per kg)
    - Price Spread: **PKR 18,000-22,000 per ton** (PKR 18-22 per kg)
    """)
    
    # Input columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📋 Strategy Inputs")
        annual_need = st.number_input(
            "Annual Flour Need (Tons)",
            min_value=5000, max_value=30000, value=10000, step=1000
        )
        
        storage_coverage = st.slider(
            "Silo Coverage (% of Annual Need)",
            min_value=10, max_value=80, value=30, step=5
        )
        
        analysis_years = st.slider(
            "Analysis Period (Years)",
            min_value=1, max_value=10, value=7, step=1
        )
    
    with col2:
        st.markdown("##### ⚙️ Cost Parameters")
        holding_cost = st.number_input(
            "Silo Rental Cost (PKR/Kg/Month)",
            min_value=0.10, max_value=1.50, value=0.50, format="%.2f", step=0.05
        )
        
        working_capital_rate = st.slider(
            "Working Capital Rate (%)",
            min_value=5, max_value=25, value=15, step=1
        )
    
    with col3:
        st.markdown("##### 🎯 Strategic Choice")
        strategy = st.radio(
            "Select Storage Strategy:",
            ["Rental (Option A)", "Ownership (Option B)", "Hybrid (Option C)"],
            index=2
        )
        
        strategy_code = "hybrid"
        if "Rental" in strategy:
            strategy_code = "rental"
        elif "Ownership" in strategy:
            strategy_code = "ownership"
        
        if strategy_code == "rental":
            st.info("**Option A - Silo Rental**: No CapEx, High recurring costs.")
        elif strategy_code == "ownership":
            st.info("**Option B - Full Ownership**: High CapEx (PKR 153M), Lower long-term costs.")
        else:
            st.info("**Option C - Hybrid Model**: CapEx + Fixed O&M. Balanced risk & control.")
    
    st.markdown("---")
    
    # --- HARDCODED VALUES FROM REPORT (Page 5.1) ---
    harvest_price = 92.5  # PKR/kg (middle of 90-95 range)
    lean_price = 112.5    # PKR/kg (middle of 110-115 range)
    price_spread = 20.0   # PKR/kg (middle of 18-22 range)
    
    # --- CALCULATIONS ---
    recommended_volume_tons = annual_need * (storage_coverage / 100)
    recommended_volume_kg = recommended_volume_tons * 1000
    
    # Gross savings
    gross_savings_pkr = price_spread * recommended_volume_kg
    
    # Storage costs (6 months)
    storage_cost_pkr = holding_cost * 6 * recommended_volume_kg
    
    # Working capital cost
    inventory_value = harvest_price * recommended_volume_kg
    wc_cost_pkr = inventory_value * (working_capital_rate / 100)
    
    # Strategy-specific costs
    if strategy_code == "ownership":
        capex = 153460000  # PKR 153.46M
        annual_om = holding_cost * 12 * recommended_volume_kg * 0.3  # 30% of rental
    elif strategy_code == "hybrid":
        capex = 153460000
        annual_om = 50000000  # PKR 50M/year fixed O&M
    else:  # rental
        capex = 0
        annual_om = holding_cost * 12 * recommended_volume_kg
    
    # Annual savings
    annual_net_savings = gross_savings_pkr - storage_cost_pkr - wc_cost_pkr - annual_om
    
    # Multi-year savings
    total_savings = annual_net_savings * analysis_years - capex
    
    # Display Results
    st.subheader("📊 Financial Impact Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Price Spread",
            f"PKR {price_spread:.2f}/kg",
            delta="April-May → Oct-Dec"
        )
    
    with col2:
        st.metric(
            "Storage Volume",
            f"{int(recommended_volume_tons):,} Tons",
            delta=f"{storage_coverage}% of Need"
        )
    
    with col3:
        color = "normal" if total_savings > 0 else "inverse"
        st.metric(
            f"Net Savings ({analysis_years} Years)",
            f"PKR {total_savings/1_000_000:,.1f}M",
            delta="Profitable" if total_savings > 0 else "Not Viable",
            delta_color=color
        )
    
    # Detailed breakdown
    st.markdown("##### 💰 Detailed Breakdown")
    
    st.write(f"**Revenue (Savings):**")
    st.write(f"- Harvest Price: PKR {harvest_price:.2f}/kg")
    st.write(f"- Lean Price: PKR {lean_price:.2f}/kg")
    st.write(f"- Price Spread: PKR {price_spread:.2f}/kg")
    st.write(f"- Annual Gross Savings: PKR {gross_savings_pkr/1_000_000:,.1f}M")
    
    st.write(f"**Costs:**")
    st.write(f"- Storage Cost (6 months): PKR {storage_cost_pkr/1_000_000:,.1f}M")
    st.write(f"- Working Capital Cost: PKR {wc_cost_pkr/1_000_000:,.1f}M/year")
    if strategy_code == "ownership":
        st.write(f"- Capital Expenditure: PKR 153.5M (one-time)")
        st.write(f"- Annual O&M: PKR {annual_om/1_000_000:,.1f}M")
    elif strategy_code == "hybrid":
        st.write(f"- Capital Expenditure: PKR 153.5M (one-time)")
        st.write(f"- Annual O&M Fee: PKR 50.0M")
    else:
        st.write(f"- Annual Rental: PKR {annual_om/1_000_000:,.1f}M")
    
    # Chart
    st.markdown("---")
    st.subheader("📈 Cumulative Savings Projection")
    
    years = list(range(1, analysis_years + 1))
    cumulative = []
    
    for year in years:
        year_savings = annual_net_savings * year
        if strategy_code == "ownership" and year == 1:
            year_savings -= 153460000
        elif strategy_code == "hybrid":
            year_savings -= 153460000
            year_savings -= 50000000 * year
        cumulative.append(year_savings / 1_000_000)
    
    chart_df = pd.DataFrame({
        'Year': years,
        'Cumulative Savings (PKR M)': cumulative
    })
    
    st.line_chart(chart_df.set_index('Year'))
    
    # Final recommendation
    st.markdown("---")
    st.subheader("🎯 Strategic Recommendation")
    
    if total_savings > 0:
        st.success(f"""
        **✅ RECOMMENDATION: PROCEED WITH {strategy.upper()}**
        
        The {strategy} model shows **positive net savings of PKR {total_savings/1_000_000:,.1f}M** 
        over {analysis_years} years, validating the storage investment.
        
        **Key Drivers:**
        - Price arbitrage: PKR {price_spread:.1f}/kg spread
        - Volume: {int(recommended_volume_tons):,} tons ({storage_coverage}% of annual need)
        - Strategy: {strategy} provides optimal risk-return balance
        """)
    else:
        st.error(f"""
        **❌ NOT RECOMMENDED: {strategy.upper()} is not viable**
        
        The analysis shows **negative savings of PKR {abs(total_savings)/1_000_000:,.1f}M**.
        Consider adjusting:
        1. Reduce storage coverage (% of annual need)
        2. Negotiate lower rental/holding costs
        3. Explore different strategy (Rental vs Ownership vs Hybrid)
        """)

# --- MAIN APP ---
st.set_page_config(layout="wide", page_title="Unilever Storage Calculator")

st.title("🌾 UniGrain Connect: Storage Strategy Calculator")
st.markdown("### Cost-Benefit Analysis for Wheat Procurement Storage")

enhanced_storage_calculator()

# Footer
st.markdown("---")
st.caption("""
**Data Source:** Unilever Pakistan Procurement Report (Page 5.1)
- Harvest Price: PKR 90,000-95,000 per ton
- Lean Price: PKR 110,000-115,000 per ton  
- Price Spread: PKR 18,000-22,000 per ton
- Assumptions: 6-month storage, 15% working capital cost, 1% storage loss
""")
