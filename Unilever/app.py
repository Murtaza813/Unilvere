import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
st.cache_data.clear()

# --- Configuration ---
st.set_page_config(layout="wide", page_title="UniGrain Connect Prototype")

# --- 1. DATA GENERATION FUNCTIONS ---


@st.cache_data
def generate_market_data(days=365 * 3):
    """Simulates fluctuating daily wheat prices for key regions."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    regions = ['Karachi (Sindh)', 'Multan (Punjab)', 'Faisalabad (Punjab)', 'Sukkur (Sindh)']
    
    data = []
    base_price = 100
    np.random.seed(42) 
    
    for date in dates:
        for region in regions:
            volatility_multiplier = 1.0
            if 'Sindh' in region:
                volatility_multiplier = 1.5 
            
            price_offset = {'Karachi (Sindh)': 3, 'Multan (Punjab)': -2, 'Faisalabad (Punjab)': 1, 'Sukkur (Sindh)': -1}[region]
            
            # FIXED: Proper seasonal pattern - LOW in harvest, HIGH in lean
            day_of_year = date.timetuple().tm_yday
            
            # Harvest = April-May (days 90-150) = LOW prices (~90-95 PKR)
            # Lean = Oct-Dec (days 270-365) = HIGH prices (~110-115 PKR)
            # Using cosine: -10 means lowest at day 135 (May), highest at day 315 (Nov)
            seasonal_factor = -10 * np.cos((day_of_year - 135) * 2 * np.pi / 365)
            
            noise = np.random.normal(0, 1) * volatility_multiplier
            
            # Slight upward trend over years
            trend = 0.05 * (date - start_date).days / 365
            
            price = base_price + price_offset + seasonal_factor + noise + trend
            
            # Ensure prices stay in realistic range
            price = max(80, min(130, price))  # Between 80-130 PKR/kg
            
            data.append({
                'Date': date,
                'Region': region,
                'Mandi_Price_PKR_per_Kg': round(price, 2),
                'Volatility_Index': round(abs(noise) + abs(seasonal_factor * 0.1), 2)
            })
            
    df = pd.DataFrame(data)
    return df
    
@st.cache_data
def generate_supplier_data(num_suppliers=50):
    """Simulates the expanded network of flour mills."""
    np.random.seed(43)
    data = {
        'Supplier_ID': [f'SUP{i:03d}' for i in range(1, num_suppliers + 1)],
        'Supplier_Name': [f'Millers Co. {i}' for i in range(1, num_suppliers + 1)],
        'Location': np.random.choice(['Karachi', 'Multan', 'Hyderabad', 'Lahore', 'Faisalabad', 'Sukkur'], num_suppliers),
        'Max_Capacity_Tons': np.random.randint(50, 500, num_suppliers),
        'Quality_Rating': np.random.uniform(2.5, 5.0, num_suppliers).round(1),
        'On_Contract': np.random.choice([True, False], num_suppliers, p=[0.05, 0.95])
    }
    df = pd.DataFrame(data)
    df.loc[0, 'Supplier_Name'] = "**Current Sole Supplier Inc.**"
    df.loc[0, 'On_Contract'] = True
    return df

@st.cache_data
def generate_tender_history(df_market, df_suppliers):
    """Simulates past procurement tenders and winning bids for regression training."""
    tender_dates = df_market['Date'].unique()[::30] 
    data = []
    for i, date in enumerate(tender_dates):
        base_price = df_market[df_market['Date'] == date]['Mandi_Price_PKR_per_Kg'].mean()
        required_tons = np.random.randint(200, 1000)
        winning_supplier = df_suppliers.sample(1).iloc[0]
        milling_cost = np.random.uniform(5, 8) 
        margin_factor = 0.5 * (5 - winning_supplier['Quality_Rating']) / 5 
        winning_price = base_price + milling_cost + margin_factor + np.random.normal(0, 0.5)
        
        data.append({
            'Tender_ID': f'TNDR{i:03d}',
            'Date_Closed': date,
            'Required_Tons': required_tons,
            'Delivery_Location': winning_supplier['Location'],
            'Winning_Supplier_ID': winning_supplier['Supplier_ID'],
            'Winning_Price_PKR_per_Kg': round(winning_price, 2),
            'Market_Base_Price': round(base_price, 2)
        })

    return pd.DataFrame(data)

# --- 2. PREDICTIVE & ANALYTICS FUNCTIONS ---

def calculate_regional_volatility(df_market):
    """Calculates the Coefficient of Variation (CV) for each region over the last 12 months."""
    
    one_year_ago = datetime.today() - timedelta(days=365)
    df_recent = df_market[df_market['Date'] >= one_year_ago].copy()
    
    volatility_data = df_recent.groupby('Region')['Mandi_Price_PKR_per_Kg'].agg(
        Mean_Price='mean',
        Std_Dev='std'
    ).reset_index()
    
    volatility_data['CV (%)'] = (volatility_data['Std_Dev'] / volatility_data['Mean_Price']) * 100
    
    volatility_data['Mean_Price'] = volatility_data['Mean_Price'].round(2)
    volatility_data['Std_Dev'] = volatility_data['Std_Dev'].round(2)
    volatility_data['CV (%)'] = volatility_data['CV (%)'].round(2)
    
    return volatility_data.sort_values(by='CV (%)', ascending=False)

def simple_price_forecasting(df_market, region, days=30):
    """Simulates a simple linear forecast for the next 'days'."""
    df_region = df_market[df_market['Region'] == region].copy()
    df_region['Days_Since_Start'] = (df_region['Date'] - df_region['Date'].min()).dt.days
    
    model = LinearRegression()
    X = df_region[['Days_Since_Start']].values
    y = df_region['Mandi_Price_PKR_per_Kg'].values
    model.fit(X, y)
    
    last_date = df_region['Date'].max()
    last_day_count = df_region['Days_Since_Start'].max()
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_days_count = [[last_day_count + i] for i in range(1, days + 1)]
    
    future_prices = model.predict(future_days_count)
    
    df_forecast = pd.DataFrame({
        'Date': future_dates,
        'Mandi_Price_PKR_per_Kg': future_prices.round(2)
    })
    
    return df_forecast

def regression_target_price(df_tenders, df_suppliers, current_market_price, required_tons, location):
    """Uses Regression to estimate the expected winning bid price."""
    df_reg = df_tenders.merge(df_suppliers[['Supplier_ID', 'Max_Capacity_Tons', 'Quality_Rating']], 
                              left_on='Winning_Supplier_ID', right_on='Supplier_ID')
    
    df_reg['Price_Premium'] = df_reg['Winning_Price_PKR_per_Kg'] - df_reg['Market_Base_Price']
    
    features = ['Market_Base_Price', 'Max_Capacity_Tons', 'Quality_Rating']
    df_reg = pd.get_dummies(df_reg, columns=['Delivery_Location'], drop_first=True)
    location_features = [col for col in df_reg.columns if 'Delivery_Location_' in col]
    features.extend(location_features)
    
    X = df_reg[features].fillna(0)
    y = df_reg['Price_Premium']
    model = LinearRegression()
    model.fit(X, y)
    
    input_data = pd.DataFrame([{
        'Market_Base_Price': current_market_price,
        'Max_Capacity_Tons': required_tons,
        'Quality_Rating': 4.0 
    }])
    
    for loc_col in location_features:
        input_data[loc_col] = 0
        if loc_col.endswith(location) and loc_col in X.columns:
             input_data[loc_col] = 1
             
    for feature in features:
        if feature not in input_data.columns:
            input_data[feature] = 0
            
    input_data = input_data[features]
    
    predicted_premium = model.predict(input_data)[0]
    target_price = current_market_price + predicted_premium
    
    return target_price

def simulate_storage_strategy(df_market, annual_need_tons, holding_cost_pk_month, strategy="rental", capex_years=7):
    """
    Enhanced to support all three strategic options from the report.
    """
    df_market['Year'] = df_market['Date'].dt.year
    df_market['Month'] = df_market['Date'].dt.month
    
    # Use data from the last 2 full years
    last_full_year = datetime.today().year - 1
    df_sim = df_market[df_market['Year'] >= last_full_year - 1].copy()
    
    # Harvest season: March, April, May (low prices)
    df_low = df_sim[df_sim['Month'].isin([3, 4, 5])]
    # Lean season: October, November, December (high prices)
    df_high = df_sim[df_sim['Month'].isin([10, 11, 12])]
    
    avg_low_price = df_low['Mandi_Price_PKR_per_Kg'].mean()
    avg_high_price = df_high['Mandi_Price_PKR_per_Kg'].mean()
    
    # Core arbitrage calculation
    gross_saving_pk_kg = avg_high_price - avg_low_price
    holding_months = 6  # May -> November storage
    holding_cost_total_pk_kg = holding_cost_pk_month * holding_months
    
    # Strategy-specific costs
    if strategy == "rental":
        # Option A: Rental model - high recurring costs
        annual_storage_cost_pk_kg = holding_cost_pk_month * 12  # Full year cost
        capex_cost_pk_kg = 0
    elif strategy == "ownership":
        # Option B: Full ownership - high CapEx, lower O&M
        # PKR 15,346 per MT installed capacity = PKR 15.35 per kg
        capex_per_kg = 15.35  # PKR per kg capacity
        annual_capex_cost_pk_kg = capex_per_kg / capex_years
        # Lower O&M cost vs rental (approx 30% of rental)
        annual_storage_cost_pk_kg = holding_cost_pk_month * 12 * 0.3
        capex_cost_pk_kg = annual_capex_cost_pk_kg
    elif strategy == "hybrid":
        # Option C: Hybrid - CapEx + fixed O&M
        capex_per_kg = 15.35  # Same as ownership
        annual_capex_cost_pk_kg = capex_per_kg / capex_years
        # Fixed O&M fee (approx PKR 5 per kg annually from PKR 50M for 10,000 MT)
        fixed_om_pk_kg = 5.0  # PKR per kg per year
        annual_storage_cost_pk_kg = fixed_om_pk_kg
        capex_cost_pk_kg = annual_capex_cost_pk_kg
    else:
        annual_storage_cost_pk_kg = holding_cost_pk_month * 12
        capex_cost_pk_kg = 0
    
    # Net saving per kg
    net_saving_pk_kg = gross_saving_pk_kg - holding_cost_total_pk_kg - annual_storage_cost_pk_kg - capex_cost_pk_kg
    
    return {
        'Net_Saving_PKR_per_Kg': net_saving_pk_kg,
        'Gross_Saving_PKR_per_Kg': gross_saving_pk_kg,
        'Avg_Low_Price': avg_low_price,
        'Avg_High_Price': avg_high_price,
        'Harvest_Season': "March-May",
        'Lean_Season': "October-December",
        'Price_Spread_PKR_per_Kg': gross_saving_pk_kg,
        'Strategy': strategy.upper()
    }

# --- ENHANCED STORAGE CALCULATOR UI ---

def enhanced_storage_calculator():
    """Enhanced calculator with all three strategic options."""
    st.subheader("🏦 Strategic Storage Options Calculator")
    st.caption("Compare Rental, Ownership, and Hybrid models for maximum procurement savings")
    
    # Three-column layout for strategy comparison
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📋 Strategy Inputs")
        annual_need = st.number_input(
            "Annual Flour Need (Tons)",
            min_value=5000, 
            max_value=30000, 
            value=10000, 
            step=1000,
            help="Unilever's total annual wheat requirement"
        )
        
        storage_coverage = st.slider(
            "Silo Coverage (% of Annual Need)",
            min_value=10, 
            max_value=80, 
            value=50, 
            step=5,
            help="What percentage to store during harvest season"
        )
        
        analysis_years = st.slider(
            "Analysis Period (Years)",
            min_value=1, 
            max_value=10, 
            value=7,
            help="Strategic planning horizon"
        )
    
    with col2:
        st.markdown("##### ⚙️ Cost Parameters")
        
        holding_cost = st.number_input(
            "Silo Rental Cost (PKR/Kg/Month)",
            min_value=0.10,
            max_value=1.50,
            value=0.50, 
            format="%.2f",
            step=0.05,
            help="Monthly holding cost for rented silos"
        )
        
        working_capital_rate = st.slider(
            "Working Capital Rate (%)",
            min_value=5,
            max_value=25,
            value=15,
            step=1,
            help="Cost of capital for inventory financing"
        )
        
        storage_loss_rate = st.slider(
            "Storage Loss Rate (%)",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Annual loss due to shrinkage, pests, etc."
        )
    
    with col3:
        st.markdown("##### 🎯 Strategic Choice")
        strategy = st.radio(
            "Select Storage Strategy:",
            ["Rental (Option A)", "Ownership (Option B)", "Hybrid (Option C)"],
            help="Choose your strategic approach"
        )
        
        # Map to strategy codes
        strategy_code = "rental"
        if "Ownership" in strategy:
            strategy_code = "ownership"
        elif "Hybrid" in strategy:
            strategy_code = "hybrid"
        
        # Strategy descriptions
        if strategy_code == "rental":
            st.info("**Option A - Silo Rental**: No CapEx, High recurring costs. Quick implementation.")
        elif strategy_code == "ownership":
            st.info("**Option B - Full Ownership**: High CapEx (PKR 153M), Lower long-term costs.")
        else:
            st.info("**Option C - Hybrid Model**: CapEx + Fixed O&M. Balanced risk & control.")
    
    st.markdown("---")
    
    # Calculate recommended volume
    recommended_volume_tons = annual_need * (storage_coverage / 100)
    
    # Run simulation for selected strategy
    storage_results = simulate_storage_strategy(
        df_market,
        recommended_volume_tons,
        holding_cost,
        strategy_code,
        analysis_years
    )
    
    # Calculate working capital impact
    inventory_value = recommended_volume_tons * 1000 * storage_results['Avg_Low_Price']
    working_capital_cost = inventory_value * (working_capital_rate / 100)
    
    # Calculate total savings over analysis period
    annual_savings = storage_results['Net_Saving_PKR_per_Kg'] * recommended_volume_tons * 1000
    total_savings = annual_savings * analysis_years
    total_wc_cost = working_capital_cost * analysis_years
    
    # Adjust for strategy-specific costs
    if strategy_code == "ownership":
        capex = 153460000  # PKR 153.46M for 10,000 MT
        total_savings -= capex
    elif strategy_code == "hybrid":
        capex = 153460000
        annual_om = 50000000  # PKR 50M/year O&M fee
        total_savings -= (capex + (annual_om * analysis_years))
    
    # Net savings after all costs
    net_savings = total_savings - total_wc_cost
    
    # Display Results
    st.subheader("📊 Financial Impact Analysis")
    
    # Key Metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            "Price Spread (Harvest vs Lean)",
            f"PKR {storage_results['Price_Spread_PKR_per_Kg']:.2f}/kg",
            delta=f"{storage_results['Harvest_Season']} → {storage_results['Lean_Season']}"
        )
    
    with metric_col2:
        st.metric(
            "Recommended Storage Volume",
            f"{int(recommended_volume_tons):,} Tons",
            delta=f"{storage_coverage}% of Annual Need"
        )
    
    with metric_col3:
        savings_color = "normal" if net_savings > 0 else "inverse"
        st.metric(
            f"Net Savings over {analysis_years} Years",
            f"PKR {abs(net_savings)/1_000_000:,.1f}M",
            delta="Profitable" if net_savings > 0 else "Not Viable",
            delta_color=savings_color
        )
    
    # Detailed Breakdown
    st.markdown("##### 💰 Cost-Benefit Breakdown")
    
    breakdown_col1, breakdown_col2 = st.columns(2)
    
    with breakdown_col1:
        st.markdown("**Revenue Side (Savings):**")
        st.write(f"- Gross Price Savings: PKR {storage_results['Gross_Saving_PKR_per_Kg']:.2f}/kg")
        st.write(f"- Harvest Price: PKR {storage_results['Avg_Low_Price']:.2f}/kg")
        st.write(f"- Lean Season Price: PKR {storage_results['Avg_High_Price']:.2f}/kg")
        st.write(f"- Annual Gross Savings: PKR {annual_savings/1_000_000:,.1f}M")
    
    with breakdown_col2:
        st.markdown("**Cost Side:**")
        if strategy_code == "rental":
            st.write(f"- Annual Rental Cost: PKR {(holding_cost * 12 * recommended_volume_tons * 1000)/1_000_000:,.1f}M")
        elif strategy_code == "ownership":
            st.write(f"- Capital Expenditure: PKR 153.5M (one-time)")
            st.write(f"- Annual O&M Cost: ~30% of rental")
        else:  # hybrid
            st.write(f"- Capital Expenditure: PKR 153.5M (one-time)")
            st.write(f"- Annual O&M Fee: PKR 50.0M")
        
        st.write(f"- Working Capital Cost: PKR {working_capital_cost/1_000_000:,.1f}M/year")
        st.write(f"- Storage Loss ({storage_loss_rate}%): PKR {(inventory_value * storage_loss_rate/100)/1_000_000:,.1f}M/year")
    
    # Strategic Insights
    st.markdown("---")
    st.subheader("🎯 Strategic Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        # Breakeven Analysis
        if strategy_code == "rental":
            breakeven = "Immediate (monthly)"
        elif strategy_code == "ownership":
            breakeven = f"~{analysis_years} years (depends on utilization)"
        else:  # hybrid
            breakeven = "4-5 years (operational breakeven)"
        
        st.markdown(f"**📅 Breakeven Timeline:** {breakeven}")
        
        # Risk Assessment
        if strategy_code == "rental":
            risk_level = "Low"
            risk_desc = "Flexible, low commitment"
        elif strategy_code == "ownership":
            risk_level = "High"
            risk_desc = "High capital risk, long-term commitment"
        else:
            risk_level = "Medium"
            risk_desc = "Balanced risk profile"
        
        st.markdown(f"**⚠️ Risk Level:** {risk_level} - {risk_desc}")
    
    with insight_col2:
        # Strategic Fit
        if strategy_code == "rental":
            fit = "Short-term (1-3 years)"
            recommendation = "Start with rental to test market"
        elif strategy_code == "ownership":
            fit = "Long-term (5+ years)"
            recommendation = "Commit if demand is stable and predictable"
        else:
            fit = "Medium-term (3-7 years)"
            recommendation = "Ideal balance of control and flexibility"
        
        st.markdown(f"**🎯 Strategic Fit:** {fit}")
        st.markdown(f"**💡 Recommendation:** {recommendation}")
    
    # Cumulative Savings Chart using Streamlit's built-in chart
    st.markdown("---")
    st.subheader("📈 Cumulative Savings Projection")
    
    # Generate projection data
    years = list(range(1, analysis_years + 1))
    cumulative_savings = []
    
    for year in years:
        year_savings = annual_savings * year
        
        # Deduct costs
        if strategy_code == "ownership" and year == 1:
            year_savings -= 153460000  # Capex in year 1
        elif strategy_code == "hybrid":
            year_savings -= 153460000  # Capex in year 1
            year_savings -= 50000000 * year  # Annual O&M
        
        year_savings -= working_capital_cost * year  # Working capital
        cumulative_savings.append(year_savings / 1_000_000)  # Convert to millions
    
    # Create chart using Streamlit's built-in line_chart (NO PLOTLY)
    chart_data = pd.DataFrame({
        'Year': years,
        'Cumulative Savings (PKR M)': cumulative_savings
    })
    
    st.line_chart(chart_data.set_index('Year'))
    
    # Policy Warning
    st.warning("""
    **Important:** This analysis assumes stable government procurement policies. 
    Policy changes (import bans, price controls, subsidy changes) may significantly impact actual savings.
    """)
    
    return {
        'strategy': strategy,
        'net_savings': net_savings,
        'annual_savings': annual_savings,
        'breakeven_year': 4 if strategy_code == "hybrid" else 1
    }

# --- Data Generation (Run once) ---
df_market = generate_market_data()
df_suppliers = generate_supplier_data()
df_tenders = generate_tender_history(df_market, df_suppliers)
current_price = df_market['Mandi_Price_PKR_per_Kg'].iloc[-1].round(2)
historical_avg = df_market['Mandi_Price_PKR_per_Kg'].mean().round(2)

# --- 3. STREAMLIT PAGES ---

def page_dashboard():
    """Page 1: The Main Dashboard with Analytics and Bidding."""
    
    st.title("🌾 UniGrain Connect: Strategic Procurement Dashboard")
    st.markdown("### Cost Volatility Mitigation Prototype for Unilever")

    # --- KPI Section ---
    st.subheader("Key Performance Indicators (KPIs)")
    col1, col2, col3 = st.columns(3)

    col1.metric("Current Wheat Mandi Price (PKR/Kg)", f"{current_price}")
    col2.metric("3-Year Avg. Price (PKR/Kg)", f"{historical_avg}", delta=f"{(current_price - historical_avg):.2f}", delta_color="inverse")
    col3.metric("Supplier Network Size", f"{len(df_suppliers)} Millers", delta=f"Open Network: +{len(df_suppliers) - 1}", delta_color="normal")
    st.markdown("---")
    
    # --- Volatility Analysis ---
    st.subheader("📊 Price Volatility & Risk Analysis")
    
    df_volatility = calculate_regional_volatility(df_market)
    
    col_cv_chart, col_cv_table = st.columns([2, 1])
    
    with col_cv_chart:
        st.bar_chart(df_volatility.set_index('Region')['CV (%)'])
    
    with col_cv_table:
        st.markdown("##### Regional Volatility Scorecard (Last 12 Months)")
        st.dataframe(
            df_volatility[['Region', 'CV (%)', 'Mean_Price']],
            use_container_width=True,
            hide_index=True
        )

    st.info(f"**Decision Insight:** The region with the highest CV ({df_volatility['Region'].iloc[0]}) should be prioritized for long-term strategies like **Silo Storage** to lock in costs and mitigate price risk.")
    st.markdown("---")
    
    # --- ENHANCED STORAGE CALCULATOR ---
    enhanced_storage_calculator()
    
    # --- ANALYTICS AND BIDDING ---
    col_intelligence, col_marketplace = st.columns([1, 1])

    # --- COLUMN 1: MARKET INTELLIGENCE & FORECASTING ---
    with col_intelligence:
        st.subheader("🧠 Market Intelligence & Predictive Sourcing")
        
        # --- Price Forecasting Widget ---
        st.markdown("#### **Price Risk Forecasting (30-Day Outlook)**")
        
        forecast_region = st.selectbox(
            'Select Region for Price Forecast:',
            df_market['Region'].unique()
        )
        
        df_forecast = simple_price_forecasting(df_market, forecast_region)
        df_plot = df_market[df_market['Region'] == forecast_region].rename(columns={'Mandi_Price_PKR_per_Kg': 'Historical Price'})
        df_forecast = df_forecast.rename(columns={'Mandi_Price_PKR_per_Kg': 'Forecast Price'})
        df_full = pd.merge(df_plot[['Date', 'Historical Price']], df_forecast, on='Date', how='outer')
        df_full = df_full.set_index('Date').sort_index()
        
        st.line_chart(df_full)
        
        # Actionable Insight based on forecast
        latest_forecast = df_forecast['Forecast Price'].iloc[-1]
        current_regional_price = df_plot['Historical Price'].iloc[-1]
        
        st.markdown("##### **Actionable Insight**")
        if latest_forecast > current_regional_price * 1.01:
            st.error(f"**High Risk:** Price is predicted to **increase by {(latest_forecast - current_regional_price):.2f} PKR**. Initiate **Reverse Auction NOW**.")
        elif latest_forecast < current_regional_price * 0.99:
            st.warning(f"**Potential Saving:** Price is predicted to **decrease by {(current_regional_price - latest_forecast):.2f} PKR**. Consider holding off on major purchase.")
        else:
            st.info("Price is stable. Proceed with standard procurement plan.")

        st.markdown("---")

        # --- Target Price Regression Widget ---
        st.markdown("#### **Target Price Estimator (Regression Model)**")
        
        market_price = st.number_input("Input Current Mandi Price (PKR/Kg)", value=current_price, format="%.2f")
        required_tons = st.slider("Required Volume (Tons)", min_value=100, max_value=2000, value=500)
        delivery_loc = st.selectbox("Delivery Location (for transport cost factor)", df_suppliers['Location'].unique())

        target_price = regression_target_price(df_tenders, df_suppliers, market_price, required_tons, delivery_loc)
        
        st.success(f"**Predicted Target Winning Bid (PKR/Kg):** **{target_price:.2f}**")
        st.caption("This price is used to benchmark quotes from the competing millers.")

    # --- COLUMN 2: BIDDING MARKETPLACE & TENDER MANAGEMENT ---
    with col_marketplace:
        st.subheader("💪 Digital Bidding Marketplace")
        
        st.markdown("#### **Create New Reverse Auction Tender**")
        
        with st.form("new_tender_form"):
            req_tons = st.number_input("Required Fine Flour Volume (Tons)", min_value=100, value=500)
            req_date = st.date_input("Required Delivery Date", datetime.today() + timedelta(days=14))
            mode = st.radio("Procurement Mode", ["Buy Product (Flour)", "Toll Manufacturing (Grinding Service)"])
            
            submitted = st.form_submit_button("Post Live Tender (Simulated)")
            
            if submitted:
                st.success(f"Tender for **{req_tons} Tons** posted successfully to **{len(df_suppliers)}** competing millers! Bidding has started.")

        st.markdown("---")
        
        st.markdown("#### **Live Bidding Window (Tender TNDR001)**")
        
        # Simulate live bids
        np.random.seed(100)
        bids = []
        
        for i in range(10): # Show 10 simulated bidders
            supplier = df_suppliers.sample(1).iloc[0]
            bid_price = target_price + np.random.uniform(-1.5, 2.5) 
            
            bids.append({
                'Supplier': supplier['Supplier_Name'],
                'Location': supplier['Location'],
                'Quality Rating': supplier['Quality_Rating'],
                'Bid Price (PKR/Kg)': round(bid_price, 2)
            })

        df_bids = pd.DataFrame(bids).sort_values(by='Bid Price (PKR/Kg)')
        
        # Highlight the lowest bid
        def highlight_min_price(s):
            is_min = s == s.min()
            return ['background-color: #a0f0a0' if v else '' for v in is_min]

        st.dataframe(df_bids.style.apply(highlight_min_price, subset=['Bid Price (PKR/Kg)']), 
                     use_container_width=True,
                     hide_index=True)
        
        lowest_bid = df_bids['Bid Price (PKR/Kg)'].min()
        saving = target_price - lowest_bid
        
        st.metric(label="Lowest Bid Price", value=f"{lowest_bid:.2f} PKR/Kg", 
                  delta=f"Savings vs. Target: {saving:.2f} PKR/Kg", delta_color="normal")


def page_supplier_network():
    """Page 2: Detailed view of the expanded supplier network."""
    
    st.title("🌐 Supplier Network & Vetting Module")
    st.markdown("### Expanding the Sourcing Pool for Cost Reduction")

    st.warning(f"Unilever previously relied on 1-5 main suppliers. **UniGrain Connect** opens the market to **{len(df_suppliers)}** competing millers.")
    st.markdown("---")

    col_filters, col_table = st.columns([1, 3])

    with col_filters:
        st.markdown("#### Filter Network")
        
        # Filter 1: Location
        selected_locations = st.multiselect(
            'Filter by Location:',
            options=df_suppliers['Location'].unique(),
            default=df_suppliers['Location'].unique()
        )
        
        # Filter 2: Quality Rating
        min_rating = st.slider(
            'Minimum Quality Rating:',
            min_value=2.5, max_value=5.0, value=3.0, step=0.1
        )
        
        # Filter 3: Capacity
        max_capacity = st.slider(
            'Maximum Required Capacity (Tons/Day):',
            min_value=50, max_value=500, value=250, step=10
        )

    # Apply filters
    df_filtered = df_suppliers[
        (df_suppliers['Location'].isin(selected_locations)) &
        (df_suppliers['Quality_Rating'] >= min_rating) &
        (df_suppliers['Max_Capacity_Tons'] <= max_capacity)
    ]

    with col_table:
        st.markdown(f"#### Verified Millers ({len(df_filtered)} Found)")
        st.dataframe(
            df_filtered[['Supplier_Name', 'Location', 'Max_Capacity_Tons', 'Quality_Rating', 'On_Contract']],
            use_container_width=True,
            column_config={
                "On_Contract": "Existing Supplier?",
                "Max_Capacity_Tons": "Capacity (Tons/Day)",
                "Quality_Rating": st.column_config.ProgressColumn(
                    "Quality Rating (1-5)",
                    format="%.1f",
                    min_value=2.5,
                    max_value=5.0,
                ),
            },
            hide_index=True
        )
        st.caption("The 'Existing Supplier' is easily replaced by the network of verified millers above.")

# --- MAIN APP LOGIC (Sidebar Navigation) ---
st.sidebar.title("UniGrain Connect")
app_mode = st.sidebar.radio("Navigation", ["Dashboard & Bidding", "Supplier Network & Vetting"])

if app_mode == "Dashboard & Bidding":
    page_dashboard()
elif app_mode == "Supplier Network & Vetting":
    page_supplier_network()


