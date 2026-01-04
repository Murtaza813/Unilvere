import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import json
import random

# --- Configuration ---
st.set_page_config(layout="wide", page_title="UniGrain Connect Prototype")

# --- CLEAR CACHE ---
st.cache_data.clear()
st.cache_resource.clear()

# --- 1. DATA GENERATION FUNCTIONS ---

@st.cache_data
def generate_market_data(days=365 * 3):
    """FIXED: Simulates wheat prices with correct seasonal pattern."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    regions = ['Karachi (Sindh)', 'Multan (Punjab)', 'Faisalabad (Punjab)', 'Sukkur (Sindh)']
    
    data = []
    np.random.seed(42)
    
    for date in dates:
        month = date.month
        day_of_year = date.timetuple().tm_yday
        
        for region in regions:
            # Regional base prices
            if region == 'Karachi (Sindh)':
                base_price = 103
                volatility = 1.5
            elif region == 'Multan (Punjab)':
                base_price = 98
                volatility = 1.0
            elif region == 'Faisalabad (Punjab)':
                base_price = 101
                volatility = 1.0
            else:  # Sukkur
                base_price = 99
                volatility = 1.5
            
            # FIXED: CORRECT SEASONAL PATTERN
            # Harvest (April-May): LOW prices ~90-95
            # Lean (Oct-Dec): HIGH prices ~110-115
            if month in [3, 4, 5]:  # Harvest season
                seasonal_adjustment = -8  # LOWER prices
            elif month in [10, 11, 12]:  # Lean season
                seasonal_adjustment = 8   # HIGHER prices
            else:
                seasonal_adjustment = 0
            
            # Add some noise
            noise = np.random.normal(0, 2) * volatility
            
            # Small upward trend over years
            days_since_start = (date - start_date).days
            trend = 0.03 * (days_since_start / 365)
            
            price = base_price + seasonal_adjustment + noise + trend
            
            # Keep in realistic range
            price = max(85, min(125, price))
            
            data.append({
                'Date': date,
                'Region': region,
                'Mandi_Price_PKR_per_Kg': round(price, 2),
                'Volatility_Index': round(abs(noise), 2)
            })
    
    return pd.DataFrame(data)

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
    """Simulates past procurement tenders."""
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

# --- Farmer Integration Data ---

@st.cache_data
def generate_farmer_data():
    """Generate sample farmer data."""
    np.random.seed(44)
    
    villages = ['Multan', 'Lahore', 'Faisalabad', 'Sahiwal', 'Bahawalpur', 'Okara']
    
    farmers = []
    for i in range(1, 51):
        village = random.choice(villages)
        farm_size = random.randint(5, 50)  # acres
        
        farmers.append({
            'Farmer_ID': f'FA{i:03d}',
            'Name': f'Farmer {i}',
            'CNIC': f'35201-{random.randint(1000000, 9999999)}-{random.randint(1, 9)}',
            'Village': village,
            'Farm_Size_Acres': farm_size,
            'Expected_Production_Tons': farm_size * 0.8,  # 0.8 tons per acre
            'Group_ID': f'GRP{(i-1)//10 + 1:03d}',
            'Joined_Date': datetime.today() - timedelta(days=random.randint(1, 90)),
            'Status': random.choice(['Active', 'Active', 'Active', 'Pending']),
            'Contracted_Volume_Tons': round(farm_size * 0.8 * 0.7, 2),  # 70% contracted
            'Loan_Amount_PKR': random.choice([0, 50000, 100000, 150000]),
            'Quality_Score': random.uniform(3.0, 5.0)
        })
    
    return pd.DataFrame(farmers)

@st.cache_data
def generate_farmer_groups():
    """Generate farmer groups."""
    groups = []
    for i in range(1, 6):
        groups.append({
            'Group_ID': f'GRP{i:03d}',
            'Group_Name': f'Village Group {i}',
            'Leader_Farmer_ID': f'FA{(i-1)*10 + 1:03d}',
            'Total_Members': 10,
            'Total_Contracted_Tons': 280,
            'Avg_Quality_Score': random.uniform(3.5, 4.5),
            'Social_Collateral': 'Active'
        })
    
    return pd.DataFrame(groups)

@st.cache_data
def generate_input_catalog():
    """Generate input catalog with Unilever discounts."""
    catalog = [
        {'Item_ID': 'INP001', 'Item_Name': 'Wheat Seed (50kg)', 'Market_Price': 5000, 'Alliance_Price': 3750, 'Discount': '25%', 'Stock': 'Available'},
        {'Item_ID': 'INP002', 'Item_Name': 'Urea Fertilizer (50kg)', 'Market_Price': 3000, 'Alliance_Price': 2250, 'Discount': '25%', 'Stock': 'Available'},
        {'Item_ID': 'INP003', 'Item_Name': 'DAP Fertilizer (50kg)', 'Market_Price': 4500, 'Alliance_Price': 3375, 'Discount': '25%', 'Stock': 'Available'},
        {'Item_ID': 'INP004', 'Item_Name': 'Pesticide (1L)', 'Market_Price': 2000, 'Alliance_Price': 1500, 'Discount': '25%', 'Stock': 'Available'},
        {'Item_ID': 'INP005', 'Item_Name': 'Irrigation Pump', 'Market_Price': 15000, 'Alliance_Price': 11250, 'Discount': '25%', 'Stock': 'Available'},
    ]
    return pd.DataFrame(catalog)

@st.cache_data
def generate_loan_products():
    """Generate loan products."""
    loans = [
        {'Loan_ID': 'LN001', 'Purpose': 'Seed Purchase', 'Amount_Range': 'PKR 50,000-100,000', 'Interest_Rate': '12%', 'Term': '12 months'},
        {'Loan_ID': 'LN002', 'Purpose': 'Fertilizer Purchase', 'Amount_Range': 'PKR 25,000-50,000', 'Interest_Rate': '12%', 'Term': '6 months'},
        {'Loan_ID': 'LN003', 'Purpose': 'Equipment', 'Amount_Range': 'PKR 100,000-200,000', 'Interest_Rate': '12%', 'Term': '24 months'},
    ]
    return pd.DataFrame(loans)

# --- 2. PREDICTIVE FUNCTIONS ---

def calculate_regional_volatility(df_market):
    """Calculates price volatility by region."""
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
    """Simple linear forecast."""
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
    
    return pd.DataFrame({
        'Date': future_dates,
        'Mandi_Price_PKR_per_Kg': future_prices.round(2)
    })

def regression_target_price(df_tenders, df_suppliers, current_market_price, required_tons, location):
    """Predicts winning bid price."""
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
    return current_market_price + predicted_premium

# --- FIXED STORAGE CALCULATOR ---

def simulate_storage_strategy(annual_need_tons, holding_cost_pk_month, strategy="rental", capex_years=7):
    """
    FIXED: Uses hardcoded values from Unilever report (Page 5.1)
    Harvest: PKR 90-95/kg, Lean: PKR 110-115/kg, Spread: PKR 18-22/kg
    """
    # VALUES FROM YOUR REPORT (Page 5.1)
    harvest_price = 92.5  # PKR/kg (middle of 90-95)
    lean_price = 112.5    # PKR/kg (middle of 110-115)
    price_spread = 20.0   # PKR/kg (middle of 18-22)
    
    holding_months = 6
    holding_cost_total = holding_cost_pk_month * holding_months
    
    # Strategy costs
    if strategy == "rental":
        annual_storage_cost = holding_cost_pk_month * 12
        capex_cost = 0
    elif strategy == "ownership":
        capex_per_kg = 15.35  # PKR 153.46M for 10,000 MT
        annual_capex_cost = capex_per_kg / capex_years
        annual_storage_cost = holding_cost_pk_month * 12 * 0.3
        capex_cost = annual_capex_cost
    elif strategy == "hybrid":
        capex_per_kg = 15.35
        annual_capex_cost = capex_per_kg / capex_years
        fixed_om = 5.0  # PKR 50M/year for 10,000 MT
        annual_storage_cost = fixed_om
        capex_cost = annual_capex_cost
    else:
        annual_storage_cost = holding_cost_pk_month * 12
        capex_cost = 0
    
    net_saving = price_spread - holding_cost_total - annual_storage_cost - capex_cost
    
    return {
        'Net_Saving_PKR_per_Kg': net_saving,
        'Gross_Saving_PKR_per_Kg': price_spread,
        'Avg_Low_Price': harvest_price,
        'Avg_High_Price': lean_price,
        'Harvest_Season': "April-May",
        'Lean_Season': "October-December",
        'Price_Spread_PKR_per_Kg': price_spread,
        'Strategy': strategy.upper()
    }

# --- ENHANCED STORAGE CALCULATOR UI ---

def enhanced_storage_calculator():
    """Enhanced calculator with all three strategic options."""
    st.subheader("🏦 Strategic Storage Options Calculator")
    st.caption("Compare Rental, Ownership, and Hybrid models for maximum procurement savings")
    
    # Debug info
    with st.sidebar.expander("📊 Price Verification"):
        harvest_months = [3, 4, 5]
        lean_months = [10, 11, 12]
        
        harvest_avg = df_market[df_market['Date'].dt.month.isin(harvest_months)]['Mandi_Price_PKR_per_Kg'].mean()
        lean_avg = df_market[df_market['Date'].dt.month.isin(lean_months)]['Mandi_Price_PKR_per_Kg'].mean()
        
        st.write(f"Harvest (Mar-May): PKR {harvest_avg:.2f}/kg")
        st.write(f"Lean (Oct-Dec): PKR {lean_avg:.2f}/kg")
        st.write(f"Spread: PKR {lean_avg - harvest_avg:.2f}/kg")
        
        if harvest_avg < lean_avg:
            st.success("✅ Data is correct!")
        else:
            st.error("❌ Data is backwards!")
    
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
        
        storage_loss_rate = st.slider(
            "Storage Loss Rate (%)",
            min_value=0.1, max_value=3.0, value=1.0, step=0.1
        )
    
    with col3:
        st.markdown("##### 🎯 Strategic Choice")
        strategy = st.radio(
            "Select Storage Strategy:",
            ["Rental (Option A)", "Ownership (Option B)", "Hybrid (Option C)"],
            index=2  # Default to Hybrid
        )
        
        strategy_code = "rental"
        if "Ownership" in strategy:
            strategy_code = "ownership"
        elif "Hybrid" in strategy:
            strategy_code = "hybrid"
        
        if strategy_code == "rental":
            st.info("**Option A - Silo Rental**: No CapEx, High recurring costs.")
        elif strategy_code == "ownership":
            st.info("**Option B - Full Ownership**: High CapEx (PKR 153M), Lower long-term costs.")
        else:
            st.info("**Option C - Hybrid Model**: CapEx + Fixed O&M. Balanced risk & control.")
    
    st.markdown("---")
    
    # Calculations
    recommended_volume_tons = annual_need * (storage_coverage / 100)
    
    storage_results = simulate_storage_strategy(
        recommended_volume_tons,
        holding_cost,
        strategy_code,
        analysis_years
    )
    
    # Financial calculations
    inventory_value = recommended_volume_tons * 1000 * storage_results['Avg_Low_Price']
    working_capital_cost = inventory_value * (working_capital_rate / 100)
    annual_savings = storage_results['Net_Saving_PKR_per_Kg'] * recommended_volume_tons * 1000
    total_savings = annual_savings * analysis_years
    total_wc_cost = working_capital_cost * analysis_years
    
    # Strategy adjustments
    if strategy_code == "ownership":
        capex = 153460000  # PKR 153.46M
        total_savings -= capex
    elif strategy_code == "hybrid":
        capex = 153460000
        annual_om = 50000000  # PKR 50M/year
        total_savings -= (capex + (annual_om * analysis_years))
    
    net_savings = total_savings - total_wc_cost
    
    # Display Results
    st.subheader("📊 Financial Impact Analysis")
    
    metric1, metric2, metric3 = st.columns(3)
    
    with metric1:
        st.metric(
            "Price Spread",
            f"PKR {storage_results['Price_Spread_PKR_per_Kg']:.2f}/kg",
            delta=f"{storage_results['Harvest_Season']} → {storage_results['Lean_Season']}"
        )
    
    with metric2:
        st.metric(
            "Storage Volume",
            f"{int(recommended_volume_tons):,} Tons",
            delta=f"{storage_coverage}% of Need"
        )
    
    with metric3:
        color = "normal" if net_savings > 0 else "inverse"
        st.metric(
            f"Net Savings ({analysis_years} Years)",
            f"PKR {abs(net_savings)/1_000_000:,.1f}M",
            delta="Profitable" if net_savings > 0 else "Not Viable",
            delta_color=color
        )
    
    # Cost breakdown
    st.markdown("##### 💰 Cost-Benefit Breakdown")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("**Revenue (Savings):**")
        st.write(f"- Harvest Price: PKR {storage_results['Avg_Low_Price']:.2f}/kg")
        st.write(f"- Lean Price: PKR {storage_results['Avg_High_Price']:.2f}/kg")
        st.write(f"- Price Spread: PKR {storage_results['Price_Spread_PKR_per_Kg']:.2f}/kg")
        st.write(f"- Annual Savings: PKR {annual_savings/1_000_000:,.1f}M")
    
    with col_right:
        st.markdown("**Costs:**")
        if strategy_code == "rental":
            st.write(f"- Annual Rental: PKR {holding_cost * 12 * recommended_volume_tons * 1000/1_000_000:,.1f}M")
        elif strategy_code == "ownership":
            st.write(f"- Capex: PKR 153.5M (one-time)")
        else:
            st.write(f"- Capex: PKR 153.5M + Annual O&M: PKR 50.0M")
        
        st.write(f"- Working Capital: PKR {working_capital_cost/1_000_000:,.1f}M/year")
        st.write(f"- Storage Loss: PKR {inventory_value * storage_loss_rate/100/1_000_000:,.1f}M/year")
    
    # Insights
    st.markdown("---")
    st.subheader("🎯 Strategic Insights")
    
    insight1, insight2 = st.columns(2)
    
    with insight1:
        if strategy_code == "rental":
            breakeven = "Monthly"
            risk = "Low - Flexible"
        elif strategy_code == "ownership":
            breakeven = f"{analysis_years} years"
            risk = "High - Capital intensive"
        else:
            breakeven = "4-5 years"
            risk = "Medium - Balanced"
        
        st.markdown(f"**📅 Breakeven:** {breakeven}")
        st.markdown(f"**⚠️ Risk:** {risk}")
    
    with insight2:
        if strategy_code == "rental":
            fit = "Short-term (1-3 years)"
            rec = "Start with rental to test"
        elif strategy_code == "ownership":
            fit = "Long-term (5+ years)"
            rec = "Commit if demand is stable"
        else:
            fit = "Medium-term (3-7 years)"
            rec = "Ideal balance for Unilever"
        
        st.markdown(f"**🎯 Strategic Fit:** {fit}")
        st.markdown(f"**💡 Recommendation:** {rec}")
    
    # Chart
    st.markdown("---")
    st.subheader("📈 Cumulative Savings Projection")
    
    years = list(range(1, analysis_years + 1))
    cumulative = []
    
    for year in years:
        year_savings = annual_savings * year
        if strategy_code == "ownership" and year == 1:
            year_savings -= 153460000
        elif strategy_code == "hybrid":
            year_savings -= 153460000
            year_savings -= 50000000 * year
        
        year_savings -= working_capital_cost * year
        cumulative.append(year_savings / 1_000_000)
    
    chart_df = pd.DataFrame({
        'Year': years,
        'Cumulative Savings (PKR M)': cumulative
    })
    
    st.line_chart(chart_df.set_index('Year'))
    
    st.warning("**Note:** Assumes stable government policies. Policy changes may impact savings.")
    
    return {'strategy': strategy, 'net_savings': net_savings}

# --- Generate Data ---
df_market = generate_market_data()
df_suppliers = generate_supplier_data()
df_tenders = generate_tender_history(df_market, df_suppliers)
current_price = df_market['Mandi_Price_PKR_per_Kg'].iloc[-1].round(2)
historical_avg = df_market['Mandi_Price_PKR_per_Kg'].mean().round(2)

# --- 3. STREAMLIT PAGES ---

def page_dashboard():
    """Main Dashboard."""
    st.title("🌾 UniGrain Connect: Strategic Procurement Dashboard")
    st.markdown("### Cost Volatility Mitigation Prototype for Unilever")
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price (PKR/Kg)", f"{current_price}")
    col2.metric("3-Year Average", f"{historical_avg}")
    col3.metric("Supplier Network", f"{len(df_suppliers)} Millers")
    st.markdown("---")
    
    # Volatility Analysis
    st.subheader("📊 Price Volatility Analysis")
    df_volatility = calculate_regional_volatility(df_market)
    st.bar_chart(df_volatility.set_index('Region')['CV (%)'])
    st.markdown("---")
    
    # Storage Calculator
    enhanced_storage_calculator()
    
    # Market Intelligence & Bidding
    col_intel, col_bid = st.columns(2)
    
    with col_intel:
        st.subheader("🧠 Market Intelligence")
        region = st.selectbox("Select Region:", df_market['Region'].unique())
        df_forecast = simple_price_forecasting(df_market, region)
        
        # Combine historical and forecast
        df_hist = df_market[df_market['Region'] == region][['Date', 'Mandi_Price_PKR_per_Kg']]
        df_hist = df_hist.rename(columns={'Mandi_Price_PKR_per_Kg': 'Historical'})
        df_fcst = df_forecast.rename(columns={'Mandi_Price_PKR_per_Kg': 'Forecast'})
        df_combined = pd.concat([df_hist, df_fcst]).set_index('Date').sort_index()
        
        st.line_chart(df_combined)
        
        # Insight
        latest_price = df_hist['Historical'].iloc[-1]
        forecast_price = df_fcst['Forecast'].iloc[-1]
        
        if forecast_price > latest_price * 1.02:
            st.error(f"**Warning:** Price predicted to rise by {forecast_price-latest_price:.2f} PKR")
        elif forecast_price < latest_price * 0.98:
            st.success(f"**Opportunity:** Price predicted to fall by {latest_price-forecast_price:.2f} PKR")
        else:
            st.info("Prices expected to remain stable")
    
    with col_bid:
        st.subheader("💪 Digital Bidding")
        
        with st.form("tender_form"):
            req_tons = st.number_input("Required Tons", 100, 2000, 500)
            req_date = st.date_input("Delivery Date", datetime.today() + timedelta(days=14))
            submitted = st.form_submit_button("Post Tender")
            
            if submitted:
                st.success(f"Tender for {req_tons} tons posted!")
        
        st.markdown("#### Live Bids")
        bids = []
        for i in range(8):
            supplier = df_suppliers.sample(1).iloc[0]
            base = current_price + np.random.uniform(2, 5)
            bids.append({
                'Supplier': supplier['Supplier_Name'],
                'Location': supplier['Location'],
                'Quality': supplier['Quality_Rating'],
                'Bid (PKR/Kg)': round(base, 2)
            })
        
        df_bids = pd.DataFrame(bids).sort_values('Bid (PKR/Kg)')
        st.dataframe(df_bids, hide_index=True)
        
        if len(df_bids) > 0:
            best_bid = df_bids['Bid (PKR/Kg)'].min()
            st.metric("Best Bid", f"{best_bid:.2f} PKR/Kg")

def page_supplier_network():
    """Supplier Network Page."""
    st.title("🌐 Supplier Network & Vetting")
    st.markdown(f"### Expanding from 1-5 to {len(df_suppliers)} competing millers")
    
    col_filt, col_table = st.columns([1, 3])
    
    with col_filt:
        st.markdown("#### Filter Network")
        locations = st.multiselect("Location:", df_suppliers['Location'].unique(), 
                                  default=df_suppliers['Location'].unique())
        min_rating = st.slider("Min Quality:", 2.5, 5.0, 3.0, 0.1)
        max_cap = st.slider("Max Capacity:", 50, 500, 250, 10)
    
    with col_table:
        filtered = df_suppliers[
            (df_suppliers['Location'].isin(locations)) &
            (df_suppliers['Quality_Rating'] >= min_rating) &
            (df_suppliers['Max_Capacity_Tons'] <= max_cap)
        ]
        
        st.markdown(f"#### Verified Millers ({len(filtered)} found)")
        st.dataframe(
            filtered[['Supplier_Name', 'Location', 'Max_Capacity_Tons', 'Quality_Rating', 'On_Contract']],
            column_config={
                "On_Contract": "Existing Supplier?",
                "Quality_Rating": st.column_config.ProgressColumn(min_value=2.5, max_value=5.0)
            },
            hide_index=True
        )

def page_farmer_integration():
    """Page 3: Farmer Alliance Integration Platform."""
    
    st.title("🚜 Unilever Farmer Alliance Platform")
    st.markdown("### Direct-to-Farmer Procurement with Zero Capital Investment")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "👨‍🌾 Farmer Portal", 
        "🛒 Input Marketplace", 
        "📊 Procurement Dashboard",
        "📝 Forward Contracts"
    ])
    
    # Generate data
    df_farmers = generate_farmer_data()
    df_groups = generate_farmer_groups()
    df_inputs = generate_input_catalog()
    df_loans = generate_loan_products()
    
    # --- TAB 1: Farmer Portal ---
    with tab1:
        st.subheader("👨‍🌾 Farmer Registration & Services")
        
        col_reg, col_services = st.columns([1, 1])
        
        with col_reg:
            st.markdown("##### 📝 Farmer Registration")
            
            with st.form("farmer_registration"):
                name = st.text_input("Full Name")
                cnic = st.text_input("CNIC Number", placeholder="35201-1234567-8")
                village = st.selectbox("Village", ['Multan', 'Lahore', 'Faisalabad', 'Sahiwal', 'Bahawalpur', 'Okara'])
                farm_size = st.number_input("Farm Size (Acres)", min_value=1, max_value=100, value=10)
                phone = st.text_input("Phone Number", placeholder="0300-1234567")
                
                submitted = st.form_submit_button("Register Farmer")
                if submitted:
                    st.success(f"✅ Farmer {name} registered successfully!")
                    st.info(f"**Farmer ID:** FA{len(df_farmers)+1:03d}\n**Group Assigned:** GRP{(len(df_farmers)//10)+1:03d}")
            
            st.markdown("---")
            st.markdown("##### 👥 Form/Join Farmer Group")
            
            group_option = st.radio("Group Options:", ["Create New Group", "Join Existing Group"])
            
            if group_option == "Create New Group":
                group_name = st.text_input("Group Name")
                if st.button("Create Group"):
                    st.success(f"Group '{group_name}' created! Minimum 5 farmers needed.")
            else:
                available_groups = df_groups[df_groups['Total_Members'] < 15]['Group_Name'].tolist()
                selected_group = st.selectbox("Select Group to Join", available_groups)
                if st.button("Join Group"):
                    st.success(f"Request sent to join {selected_group}")
        
        with col_services:
            st.markdown("##### 💰 Loan Services")
            
            st.markdown("**Available Loan Products:**")
            st.dataframe(df_loans, hide_index=True, use_container_width=True)
            
            with st.expander("Apply for Loan"):
                loan_type = st.selectbox("Loan Purpose", df_loans['Purpose'].tolist())
                amount = st.number_input("Amount (PKR)", min_value=10000, max_value=200000, value=50000, step=10000)
                duration = st.select_slider("Duration (Months)", options=[6, 12, 18, 24])
                
                if st.button("Submit Loan Application"):
                    st.success("✅ Loan application submitted!")
                    st.write(f"**Details:** {amount} PKR at 12% interest for {duration} months")
            
            st.markdown("---")
            st.markdown("##### 📅 Book Consultation")
            
            consult_date = st.date_input("Preferred Date", datetime.today() + timedelta(days=7))
            consult_time = st.selectbox("Time Slot", ["9:00 AM", "11:00 AM", "2:00 PM", "4:00 PM"])
            topic = st.selectbox("Topic", ["Contract Terms", "Quality Standards", "Input Usage", "Loan Process"])
            
            if st.button("Book Appointment"):
                st.success(f"✅ Appointment booked for {consult_date} at {consult_time}")
    
    # --- TAB 2: Input Marketplace ---
    with tab2:
        st.subheader("🛒 Input Marketplace")
        st.info("**Unilever Alliance Members get 25% discount on all inputs**")
        
        col_catalog, col_cart = st.columns([2, 1])
        
        with col_catalog:
            st.markdown("##### Available Inputs")
            
            for idx, item in df_inputs.iterrows():
                with st.container():
                    col_item, col_price, col_action = st.columns([3, 2, 1])
                    with col_item:
                        st.markdown(f"**{item['Item_Name']}**")
                    with col_price:
                        st.markdown(f"~~PKR {item['Market_Price']:,}~~ → **PKR {item['Alliance_Price']:,}**")
                    with col_action:
                        quantity = st.number_input("Qty", min_value=0, max_value=10, value=0, 
                                                  key=f"qty_{item['Item_ID']}", label_visibility="collapsed")
            
            if st.button("🛒 Add to Cart & Checkout"):
                st.success("Order placed! Delivery in 3-5 working days.")
        
        with col_cart:
            st.markdown("##### Your Cart")
            st.markdown("""
            - Wheat Seed: 2 bags × PKR 3,750 = **PKR 7,500**
            - Urea Fertilizer: 3 bags × PKR 2,250 = **PKR 6,750**
            - Pesticide: 1 × PKR 1,500 = **PKR 1,500**
            """)
            
            st.markdown("**Total: PKR 15,750**")
            st.markdown("~~Market Price: PKR 21,000~~")
            st.markdown("**Savings: PKR 5,250**")
            
            st.markdown("---")
            st.markdown("##### Delivery Options")
            delivery = st.radio("Delivery to:", ["Village Collection Center", "Farm Gate (+PKR 500)"])
            
            payment = st.selectbox("Payment Method", ["Cash on Delivery", "Loan Deduction", "Bank Transfer"])
    
    # --- TAB 3: Procurement Dashboard ---
    with tab3:
        st.subheader("📊 Procurement Dashboard")
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Registered Farmers", f"{len(df_farmers)}", "+25 this month")
        with col2:
            total_contracted = df_farmers['Contracted_Volume_Tons'].sum()
            st.metric("Contracted Volume", f"{total_contracted:,.0f} Tons")
        with col3:
            active_groups = len(df_groups)
            st.metric("Active Groups", f"{active_groups}", "Social Collateral Active")
        with col4:
            avg_quality = df_farmers['Quality_Score'].mean()
            st.metric("Avg Quality Score", f"{avg_quality:.1f}/5.0")
        
        st.markdown("---")
        
        col_charts, col_alerts = st.columns([2, 1])
        
        with col_charts:
            st.markdown("##### 📈 Farmer Distribution")
            
            # Village distribution chart
            village_counts = df_farmers['Village'].value_counts()
            st.bar_chart(village_counts)
            
            st.markdown("##### 📊 Contract Status")
            
            # Contract volume by group
            group_volume = df_groups[['Group_Name', 'Total_Contracted_Tons']].set_index('Group_Name')
            st.bar_chart(group_volume)
        
        with col_alerts:
            st.markdown("##### ⚠️ Alerts & Notifications")
            
            st.warning("**Quality Alert:** Group GRP003 - Protein content below 11.5%")
            st.info("**Collection Tomorrow:** Village Multan - 500 tons scheduled")
            st.success("**New Registration:** 15 farmers joined this week")
            st.error("**Payment Pending:** 3 farmers awaiting loan disbursement")
            
            st.markdown("---")
            st.markdown("##### 🚚 Collection Schedule")
            
            schedule = [
                {"Date": "2024-06-25", "Village": "Multan", "Volume": "500 tons", "Status": "Confirmed"},
                {"Date": "2024-06-27", "Village": "Lahore", "Volume": "750 tons", "Status": "Confirmed"},
                {"Date": "2024-06-30", "Village": "Faisalabad", "Volume": "600 tons", "Status": "Pending"},
            ]
            
            for item in schedule:
                st.markdown(f"**{item['Date']}**: {item['Village']} - {item['Volume']}")
                st.caption(f"Status: {item['Status']}")
        
        st.markdown("---")
        st.markdown("##### 👨‍🌾 Farmer Database")
        
        # Filterable table
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            selected_village = st.multiselect("Filter by Village", df_farmers['Village'].unique())
        with col_filter2:
            min_quality = st.slider("Minimum Quality Score", 3.0, 5.0, 3.0, 0.1)
        with col_filter3:
            show_active = st.checkbox("Show Active Only", value=True)
        
        # Apply filters
        filtered_df = df_farmers.copy()
        if selected_village:
            filtered_df = filtered_df[filtered_df['Village'].isin(selected_village)]
        filtered_df = filtered_df[filtered_df['Quality_Score'] >= min_quality]
        if show_active:
            filtered_df = filtered_df[filtered_df['Status'] == 'Active']
        
        st.dataframe(
            filtered_df[['Farmer_ID', 'Name', 'Village', 'Farm_Size_Acres', 
                        'Contracted_Volume_Tons', 'Quality_Score', 'Status']],
            use_container_width=True,
            hide_index=True
        )
    
    # --- TAB 4: Forward Contracts ---
    with tab4:
        st.subheader("📝 Digital Forward Contracts")
        st.info("**Farmer gets PKR 3,800/maund guaranteed price + 25% input discount**")
        
        col_contract, col_example = st.columns([2, 1])
        
        with col_contract:
            st.markdown("##### Create New Contract")
            
            with st.form("forward_contract"):
                farmer_id = st.selectbox("Select Farmer", df_farmers['Farmer_ID'].tolist())
                volume_tons = st.number_input("Contract Volume (Tons)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
                price_per_maund = st.number_input("Fixed Price (PKR/maund)", min_value=3700, max_value=4000, value=3800)
                harvest_date = st.date_input("Harvest Date", datetime.today() + timedelta(days=90))
                quality_terms = st.multiselect("Quality Standards", 
                                              ["Protein ≥11.5%", "Moisture ≤12%", "Purity ≥97%", "No foreign matter"])
                
                submitted = st.form_submit_button("Generate Digital Contract")
                
                if submitted:
                    st.success("✅ Contract Generated Successfully!")
                    
                    # Show contract summary
                    with st.expander("View Contract Details", expanded=True):
                        st.markdown(f"""
                        ### 📄 UNILEVER FORWARD CONTRACT
                        
                        **Contract ID:** CON-{random.randint(1000, 9999)}
                        **Farmer ID:** {farmer_id}
                        **Date:** {datetime.today().strftime('%Y-%m-%d')}
                        
                        ---
                        
                        **TERMS:**
                        - Fixed Price: **PKR {price_per_maund:,} per maund**
                        - Volume: **{volume_tons} tons** ({volume_tons * 37.5:.0f} maunds)
                        - Harvest Date: **{harvest_date}**
                        - Collection: Village collection center
                        - Payment: Within 7 days of delivery
                        
                        **QUALITY STANDARDS:**
                        {chr(10).join(f"• {term}" for term in quality_terms)}
                        
                        **FARMER BENEFITS:**
                        • 25% discount on all inputs
                        • 12% interest loans (vs 24% market)
                        • No mandi fees or commissions
                        • Guaranteed purchase
                        
                        **Total Contract Value:** PKR {volume_tons * 37.5 * price_per_maund:,.0f}
                        """)
                        
                        if st.button("📱 Send Contract to Farmer"):
                            st.success("Contract sent via WhatsApp/SMS!")
        
        with col_example:
            st.markdown("##### 📊 Farmer Economics")
            
            st.markdown("**Traditional vs Alliance Route:**")
            
            # Economic comparison
            comparison_data = {
                "Traditional Route (MSP)": {
                    "MSP Price": "PKR 4,000",
                    "Transport": "-150",
                    "Mandi Fees": "-100",
                    "Commission": "-150",
                    "Rejection Risk": "-200",
                    "Net Income": "**PKR 3,400**"
                },
                "Alliance Route": {
                    "Fixed Price": "PKR 3,800",
                    "Transport": "0",
                    "Mandi Fees": "0",
                    "Commission": "0",
                    "Input Savings": "+500",
                    "Net Benefit": "**PKR 4,300**"
                }
            }
            
            col_traditional, col_alliance = st.columns(2)
            
            with col_traditional:
                st.markdown("**❌ Traditional**")
                for key, value in comparison_data["Traditional Route (MSP)"].items():
                    st.write(f"{key}: {value}")
            
            with col_alliance:
                st.markdown("**✅ Alliance**")
                for key, value in comparison_data["Alliance Route"].items():
                    st.write(f"{key}: {value}")
            
            st.markdown("---")
            st.markdown("**💰 Extra Income:** PKR 900/maund")
            st.markdown("**For 100 maunds:** **PKR 90,000 extra!**")
            
            st.markdown("---")
            st.markdown("##### Active Contracts")
            
            contract_stats = {
                "Total Contracts": 42,
                "Volume Committed": "1,575 tons",
                "Avg Price": "PKR 3,800/maund",
                "Compliance Rate": "94%"
            }
            
            for key, value in contract_stats.items():
                st.metric(key, value)
    
    # Footer
    st.markdown("---")
    st.caption("""
    **Farmer Alliance Model Benefits:**
    • **For Farmers:** PKR 900-1,400 higher net income per maund
    • **For Unilever:** PKR 200/maund price advantage + supply security  
    • **Zero Capital:** Unilever doesn't finance inputs, only facilitates
    • **Social Collateral:** Groups guarantee each other's quality
    """)



# --- MAIN APP LOGIC (Sidebar Navigation) ---
st.sidebar.title("UniGrain Connect")
app_mode = st.sidebar.radio("Navigation", ["Dashboard & Bidding", "Supplier Network & Vetting", "Farmer Integration"])

if app_mode == "Dashboard & Bidding":
    page_dashboard()
elif app_mode == "Supplier Network & Vetting":
    page_supplier_network()
elif app_mode == "Farmer Integration":
    page_farmer_integration()


