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
@st.cache_data
def generate_mill_data():
    """Generate comprehensive mill data for toll processing"""
    np.random.seed(45)
    
    mill_types = ['Commercial', 'Medium Modern', 'Cooperative', 'Mega Mill']
    
    # Pakistan city coordinates
    city_coords = {
        'Multan': {'lat': 30.1575, 'lon': 71.5249},
        'Lahore': {'lat': 31.5497, 'lon': 74.3436},
        'Faisalabad': {'lat': 31.4180, 'lon': 73.0790},
        'Karachi': {'lat': 24.8607, 'lon': 67.0011},
        'Sukkur': {'lat': 27.7052, 'lon': 68.8574}
    }
    
    data = []
    for i in range(1, 31):
        mill_type = random.choice(mill_types)
        location = random.choice(list(city_coords.keys()))
        
        if mill_type == 'Commercial':
            capacity = random.randint(200, 500)
            quality = random.uniform(3.5, 4.8)
        elif mill_type == 'Medium Modern':
            capacity = random.randint(100, 200)
            quality = random.uniform(4.0, 5.0)
        elif mill_type == 'Cooperative':
            capacity = random.randint(50, 150)
            quality = random.uniform(3.0, 4.0)
        else:  # Mega Mill
            capacity = random.randint(400, 600)
            quality = random.uniform(3.8, 4.5)
        
        # Add some random variation to coordinates
        lat = city_coords[location]['lat'] + random.uniform(-0.1, 0.1)
        lon = city_coords[location]['lon'] + random.uniform(-0.1, 0.1)
        
        data.append({
            'Mill_ID': f'MILL{i:03d}',
            'Mill_Name': f'{mill_type} Mill {i}',
            'Location': location,
            'lat': lat,
            'lon': lon,
            'Mill_Type': mill_type,
            'Capacity_TPD': capacity,
            'Quality_Rating': round(quality, 1),
            'On_Contract': random.choice([True, False]),
            'Processing_Fee': random.randint(800, 1200),
            'Last_Audit': (datetime.today() - timedelta(days=random.randint(0, 90))).strftime('%Y-%m-%d'),
            'Contact_Person': f'Manager {i}',
            'Phone': f'0300-{random.randint(1000000, 9999999)}'
        })
    
    return pd.DataFrame(data)

@st.cache_data
def generate_iot_mill_data():
    """Generate simulated IoT data for mill operations"""
    np.random.seed(46)
    
    # Generate 24 hours of data
    hours = list(range(24))
    
    data = {
        'hour': hours,
        'throughput_tph': [random.uniform(3.5, 4.2) for _ in hours],
        'energy_kwh_per_ton': [random.uniform(80, 90) for _ in hours],
        'extraction_rate': [random.uniform(70, 72) for _ in hours],
        'temp_c': [random.uniform(25, 35) for _ in hours],
        'humidity': [random.uniform(45, 65) for _ in hours],
        'vibration': [random.uniform(0.1, 0.5) for _ in hours]
    }
    
    return pd.DataFrame(data)



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
            
            # Initialize session state for contract if not exists
            if 'contract_generated' not in st.session_state:
                st.session_state.contract_generated = False
            if 'contract_details' not in st.session_state:
                st.session_state.contract_details = {}
            
            with st.form("forward_contract"):
                farmer_id = st.selectbox("Select Farmer", df_farmers['Farmer_ID'].tolist())
                volume_tons = st.number_input("Contract Volume (Tons)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
                price_per_maund = st.number_input("Fixed Price (PKR/maund)", min_value=3700, max_value=4000, value=3800)
                harvest_date = st.date_input("Harvest Date", datetime.today() + timedelta(days=90))
                quality_terms = st.multiselect("Quality Standards", 
                                              ["Protein ≥11.5%", "Moisture ≤12%", "Purity ≥97%", "No foreign matter"])
                
                submitted = st.form_submit_button("Generate Digital Contract")
                
                if submitted:
                    # Store contract details in session state
                    st.session_state.contract_details = {
                        'farmer_id': farmer_id,
                        'volume_tons': volume_tons,
                        'price_per_maund': price_per_maund,
                        'harvest_date': harvest_date,
                        'quality_terms': quality_terms,
                        'contract_id': f"CON-{random.randint(1000, 9999)}",
                        'date': datetime.today().strftime('%Y-%m-%d')
                    }
                    st.session_state.contract_generated = True
                    st.rerun()
        
        # Show contract after form submission (OUTSIDE the form)
        if st.session_state.contract_generated and st.session_state.contract_details:
            contract = st.session_state.contract_details
            st.success("✅ Contract Generated Successfully!")
            
            with st.expander("View Contract Details", expanded=True):
                st.markdown(f"""
                ### 📄 UNILEVER FORWARD CONTRACT
                
                **Contract ID:** {contract['contract_id']}
                **Farmer ID:** {contract['farmer_id']}
                **Date:** {contract['date']}
                
                ---
                
                **TERMS:**
                - Fixed Price: **PKR {contract['price_per_maund']:,} per maund**
                - Volume: **{contract['volume_tons']} tons** ({contract['volume_tons'] * 37.5:.0f} maunds)
                - Harvest Date: **{contract['harvest_date']}**
                - Collection: Village collection center
                - Payment: Within 7 days of delivery
                
                **QUALITY STANDARDS:**
                {chr(10).join(f"• {term}" for term in contract['quality_terms'])}
                
                **FARMER BENEFITS:**
                • 25% discount on all inputs
                • 12% interest loans (vs 24% market)
                • No mandi fees or commissions
                • Guaranteed purchase
                
                **Total Contract Value:** PKR {contract['volume_tons'] * 37.5 * contract['price_per_maund']:,.0f}
                """)
                
                # Button OUTSIDE the form
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📱 Send Contract to Farmer", type="primary"):
                        st.success("Contract sent via WhatsApp/SMS!")
                        st.balloons()
                with col2:
                    if st.button("📥 Download PDF"):
                        st.info("PDF generation would be implemented here")
        
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
def page_toll_processing_management():
    """Toll Processing Management Platform"""
    st.title("🤝 Toll Processing Management")
    st.markdown("### Manage Network of Partner Mills")
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Mills", "8", "+2 this month")
    with col2:
        st.metric("Monthly Volume", "1,500 tons", "85% capacity")
    with col3:
        st.metric("Avg Quality Score", "4.2/5.0", "+0.3")
    with col4:
        st.metric("Cost Savings", "PKR 12.5M", "vs market")
    
    st.markdown("---")
    
    # Mill Performance Dashboard
    st.subheader("🏭 Mill Performance Dashboard")
    
    tab_overview, tab_quality, tab_financial, tab_contracts = st.tabs([
        "📊 Overview", "🔬 Quality Control", "💰 Financial", "📝 Contracts"
    ])
    
    with tab_overview:
        # Generate mill data
        df_mills = generate_mill_data()
        
        # Mill map with proper coordinates
        st.markdown("##### 📍 Mill Locations")
        try:
            # Filter out any null coordinates
            map_df = df_mills[['lat', 'lon', 'Mill_Name', 'Location']].dropna()
            if not map_df.empty:
                st.map(map_df, use_container_width=True)
            else:
                st.info("No location data available for mapping")
        except Exception as e:
            st.warning(f"Map display issue: {e}")
            # Show a simple table instead
            st.info("Mill Locations:")
            for _, mill in df_mills.iterrows():
                st.write(f"📍 **{mill['Mill_Name']}** - {mill['Location']}")
        
        st.markdown("---")
        
        # Mill filters
        st.markdown("##### 🔍 Filter Mills")
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            location_filter = st.multiselect(
                "Location", 
                df_mills['Location'].unique(),
                placeholder="All locations"
            )
        with col_filter2:
            capacity_filter = st.slider("Min Capacity (TPD)", 50, 500, 100, 10)
        with col_filter3:
            rating_filter = st.slider("Min Quality Rating", 3.0, 5.0, 3.5, 0.1)
        
        # Apply filters
        filtered_mills = df_mills.copy()
        if location_filter:
            filtered_mills = filtered_mills[filtered_mills['Location'].isin(location_filter)]
        filtered_mills = filtered_mills[filtered_mills['Capacity_TPD'] >= capacity_filter]
        filtered_mills = filtered_mills[filtered_mills['Quality_Rating'] >= rating_filter]
        
        st.markdown(f"**Found {len(filtered_mills)} mills matching criteria**")
        
        # Display mills in a nice table
        if not filtered_mills.empty:
            display_columns = ['Mill_Name', 'Location', 'Capacity_TPD', 
                              'Quality_Rating', 'Processing_Fee', 'On_Contract']
            
            # Format the table
            display_df = filtered_mills[display_columns].copy()
            display_df = display_df.rename(columns={
                'Mill_Name': 'Mill Name',
                'Capacity_TPD': 'Capacity (TPD)',
                'Quality_Rating': 'Quality',
                'Processing_Fee': 'Fee (PKR/ton)',
                'On_Contract': 'On Contract'
            })
            
            # Add color to quality column
            def color_quality(val):
                if val >= 4.5:
                    return 'color: green; font-weight: bold'
                elif val >= 4.0:
                    return 'color: blue'
                elif val >= 3.5:
                    return 'color: orange'
                else:
                    return 'color: red'
            
            styled_df = display_df.style.applymap(color_quality, subset=['Quality'])
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Quality": st.column_config.ProgressColumn(
                        "Quality",
                        min_value=0,
                        max_value=5,
                        format="%.1f"
                    ),
                    "On Contract": st.column_config.CheckboxColumn("On Contract")
                }
            )
            
            # Quick actions
            st.markdown("##### ⚡ Quick Actions")
            selected_mill = st.selectbox(
                "Select a mill for action:",
                filtered_mills['Mill_Name'].tolist()
            )
            
            col_action1, col_action2, col_action3 = st.columns(3)
            with col_action1:
                if st.button("📞 Contact", use_container_width=True):
                    mill_info = filtered_mills[filtered_mills['Mill_Name'] == selected_mill].iloc[0]
                    st.success(f"Contacting {mill_info['Contact_Person']} at {mill_info['Phone']}")
            with col_action2:
                if st.button("📋 Request Quote", use_container_width=True):
                    st.info(f"Quote request sent to {selected_mill}")
            with col_action3:
                if st.button("📊 View Details", use_container_width=True):
                    mill_info = filtered_mills[filtered_mills['Mill_Name'] == selected_mill].iloc[0]
                    with st.expander(f"Details for {selected_mill}", expanded=True):
                        st.write(f"**Mill ID:** {mill_info['Mill_ID']}")
                        st.write(f"**Type:** {mill_info['Mill_Type']}")
                        st.write(f"**Location:** {mill_info['Location']}")
                        st.write(f"**Capacity:** {mill_info['Capacity_TPD']} TPD")
                        st.write(f"**Quality Rating:** {mill_info['Quality_Rating']}/5.0")
                        st.write(f"**Processing Fee:** PKR {mill_info['Processing_Fee']}/ton")
                        st.write(f"**Last Audit:** {mill_info['Last_Audit']}")
                        st.write(f"**Contact:** {mill_info['Contact_Person']} - {mill_info['Phone']}")
        else:
            st.warning("No mills found with the selected filters. Try adjusting your criteria.")
    
    with tab_quality:
        st.subheader("🔬 Quality Control Dashboard")
        
        col_qc1, col_qc2 = st.columns(2)
        
        with col_qc1:
            st.markdown("##### Quality Metrics")
            quality_data = {
                'Parameter': ['Extraction Rate', 'Ash Content', 'Moisture', 'Protein'],
                'Target': ['≥71%', '≤0.48%', '≤12%', '≥11.5%'],
                'Avg Performance': ['70.8%', '0.46%', '11.2%', '11.7%'],
                'Compliance Rate': ['92%', '96%', '98%', '94%']
            }
            
            df_quality = pd.DataFrame(quality_data)
            
            # Add visual indicators
            def highlight_compliance(val):
                try:
                    compliance = float(val.replace('%', ''))
                    if compliance >= 95:
                        return 'background-color: #d4edda; color: #155724;'
                    elif compliance >= 90:
                        return 'background-color: #fff3cd; color: #856404;'
                    else:
                        return 'background-color: #f8d7da; color: #721c24;'
                except:
                    return ''
            
            styled_df = df_quality.style.applymap(
                highlight_compliance, 
                subset=['Compliance Rate']
            )
            
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
        
        with col_qc2:
            st.markdown("##### Recent Quality Tests")
            
            test_results = [
                {"Batch": "B20240615", "Mill": "Millers Co. 001", "Protein": "11.8%", "Status": "✅ Pass"},
                {"Batch": "B20240616", "Mill": "Millers Co. 002", "Protein": "11.3%", "Status": "✅ Pass"},
                {"Batch": "B20240617", "Mill": "Millers Co. 005", "Protein": "10.8%", "Status": "❌ Fail"},
                {"Batch": "B20240618", "Mill": "Millers Co. 003", "Protein": "11.9%", "Status": "✅ Pass"},
                {"Batch": "B20240619", "Mill": "Millers Co. 008", "Protein": "11.2%", "Status": "⚠️ Warning"},
            ]
            
            for test in test_results:
                col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
                with col1:
                    st.write(f"**{test['Batch']}**")
                with col2:
                    st.write(test['Mill'])
                with col3:
                    st.write(f"Protein: {test['Protein']}")
                with col4:
                    if test['Status'] == '✅ Pass':
                        st.success("Pass")
                    elif test['Status'] == '❌ Fail':
                        st.error("Fail")
                    else:
                        st.warning("Warning")
            
            # Quality trend chart
            st.markdown("##### Quality Trend - Last 30 Batches")
            
            trend_data = {
                'Batch': list(range(1, 31)),
                'Protein': [random.uniform(11.0, 12.0) for _ in range(30)],
                'Target': [11.5] * 30
            }
            
            df_trend = pd.DataFrame(trend_data)
            st.line_chart(df_trend.set_index('Batch')[['Protein', 'Target']])
    
    with tab_financial:
        st.subheader("💰 Financial Performance")
        
        # Cost comparison
        st.markdown("##### Cost Comparison Across Mills")
        
        cost_data = {
            'Mill': ['Millers Co. 001', 'Millers Co. 002', 'Millers Co. 003', 'Millers Co. 004', 'Market Average'],
            'Processing Fee': [1000, 1050, 950, 1100, 1200],
            'Transport Cost': [150, 200, 180, 170, 250],
            'Quality Bonus': [-50, 0, 100, -20, 0],
            'Total Cost': [1100, 1250, 1130, 1250, 1450],
            'Savings vs Market': [350, 200, 320, 200, 0]
        }
        
        df_costs = pd.DataFrame(cost_data)
        
        # Highlight best option
        def highlight_best(val):
            if val == df_costs['Total Cost'].min() and val != 1450:
                return 'background-color: #d4edda; font-weight: bold;'
            return ''
        
        styled_costs = df_costs.style.applymap(
            highlight_best, 
            subset=['Total Cost']
        )
        
        st.dataframe(styled_costs, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Payment tracker
        st.markdown("##### 💸 Batch Payment Tracker")
        
        payments = [
            {"Batch_ID": "B202406001", "Mill": "Millers Co. 001", "Tons": 100, 
             "Fee": "PKR 100,000", "Status": "✅ Paid", "Date": "2024-06-15"},
            {"Batch_ID": "B202406002", "Mill": "Millers Co. 002", "Tons": 150, 
             "Fee": "PKR 157,500", "Status": "⏳ Pending", "Date": "2024-06-20"},
            {"Batch_ID": "B202406003", "Mill": "Millers Co. 003", "Tons": 80, 
             "Fee": "PKR 76,000", "Status": "⚠️ Quality Hold", "Date": "2024-06-18"},
            {"Batch_ID": "B202406004", "Mill": "Millers Co. 001", "Tons": 120, 
             "Fee": "PKR 120,000", "Status": "✅ Paid", "Date": "2024-06-22"},
        ]
        
        for payment in payments:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
                with col1:
                    st.write(f"**{payment['Batch_ID']}**")
                with col2:
                    st.write(payment['Mill'])
                with col3:
                    st.write(f"{payment['Tons']} tons")
                with col4:
                    st.write(payment['Fee'])
                with col5:
                    if payment['Status'] == '✅ Paid':
                        st.success("Paid")
                    elif payment['Status'] == '⏳ Pending':
                        st.info("Pending")
                    else:
                        st.warning("On Hold")
                        if st.button("Review", key=f"review_{payment['Batch_ID']}"):
                            st.info(f"Quality issue for {payment['Batch_ID']}. Protein was 10.8% (target 11.5%)")
        
        # Financial summary
        st.markdown("---")
        st.markdown("##### 📊 Financial Summary")
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        
        with col_sum1:
            st.metric("Monthly Processing", "1,500 tons", "+200 vs last month")
        with col_sum2:
            st.metric("Avg Cost/Ton", "PKR 1,180", "-5% vs last month")
        with col_sum3:
            st.metric("Total Savings", "PKR 12.5M", "Year to date")
    
    with tab_contracts:
        st.subheader("📝 Contract Management")
        
        # Active contracts
        st.markdown("##### Active Contracts")
        
        contracts = [
            {"Contract_ID": "CON001", "Mill": "Millers Co. 001", 
             "Start_Date": "2024-01-01", "End_Date": "2024-12-31",
             "Volume_Tons": 5000, "Remaining_Tons": 1200, "Status": "🟢 Active"},
            {"Contract_ID": "CON002", "Mill": "Millers Co. 002", 
             "Start_Date": "2024-02-01", "End_Date": "2024-11-30",
             "Volume_Tons": 3000, "Remaining_Tons": 500, "Status": "🟢 Active"},
            {"Contract_ID": "CON003", "Mill": "Millers Co. 003", 
             "Start_Date": "2024-03-01", "End_Date": "2025-02-28",
             "Volume_Tons": 4000, "Remaining_Tons": 3200, "Status": "🟢 Active"},
            {"Contract_ID": "CON004", "Mill": "Millers Co. 004", 
             "Start_Date": "2023-11-01", "End_Date": "2024-10-31",
             "Volume_Tons": 2000, "Remaining_Tons": 300, "Status": "🟡 Expiring Soon"},
        ]
        
        for contract in contracts:
            with st.expander(f"📄 Contract {contract['Contract_ID']} - {contract['Mill']}", expanded=False):
                col_con1, col_con2, col_con3 = st.columns(3)
                
                with col_con1:
                    st.write(f"**Volume:** {contract['Volume_Tons']} tons")
                    progress = (contract['Volume_Tons'] - contract['Remaining_Tons']) / contract['Volume_Tons']
                    st.progress(progress)
                    st.caption(f"{contract['Remaining_Tons']} tons remaining")
                
                with col_con2:
                    st.write(f"**Duration:**")
                    st.write(f"Start: {contract['Start_Date']}")
                    st.write(f"End: {contract['End_Date']}")
                    
                    # Calculate days remaining
                    end_date = datetime.strptime(contract['End_Date'], '%Y-%m-%d')
                    days_left = (end_date - datetime.now()).days
                    if days_left < 30:
                        st.warning(f"{days_left} days remaining")
                    else:
                        st.info(f"{days_left} days remaining")
                
                with col_con3:
                    status_text = contract['Status']
                    if "🟢" in status_text:
                        st.success("Active")
                    elif "🟡" in status_text:
                        st.warning("Expiring Soon")
                    
                    # Action buttons
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if "🟡" in status_text:
                            if st.button("Renew", key=f"renew_{contract['Contract_ID']}", use_container_width=True):
                                st.success(f"Renewal process started for {contract['Contract_ID']}")
                    with col_btn2:
                        if st.button("View", key=f"view_{contract['Contract_ID']}", use_container_width=True):
                            st.info(f"Opening contract details for {contract['Mill']}")
        
        # New contract form
        st.markdown("---")
        st.markdown("##### 📝 Create New Contract")
        
        with st.form("new_contract_form"):
            col_form1, col_form2 = st.columns(2)
            
            with col_form1:
                mill_name = st.selectbox("Select Mill", ["Millers Co. 001", "Millers Co. 002", "Millers Co. 003", "New Mill"])
                volume_tons = st.number_input("Contract Volume (Tons)", min_value=100, max_value=10000, value=1000, step=100)
                price_per_ton = st.number_input("Price per Ton (PKR)", min_value=800, max_value=1500, value=1000)
            
            with col_form2:
                start_date = st.date_input("Start Date", datetime.today())
                duration_months = st.select_slider("Duration (Months)", options=[3, 6, 12, 24], value=12)
                quality_bonus = st.checkbox("Include Quality Bonus", value=True)
            
            submitted = st.form_submit_button("Create Contract")
            
            if submitted:
                contract_value = volume_tons * price_per_ton
                st.success(f"✅ Contract created successfully!")
                st.info(f"**Contract Value:** PKR {contract_value:,}")
                st.info(f"**Duration:** {start_date} to {start_date + timedelta(days=duration_months*30)}")
                
                # Simulate approval workflow
                st.markdown("##### 📋 Approval Workflow")
                workflow_steps = [
                    {"Step": "1. Contract Draft", "Status": "✅ Complete", "By": "Procurement Team"},
                    {"Step": "2. Legal Review", "Status": "⏳ In Progress", "By": "Legal Department"},
                    {"Step": "3. Finance Approval", "Status": "⏳ Pending", "By": "Finance Department"},
                    {"Step": "4. Final Signing", "Status": "⏳ Pending", "By": "Director"}
                ]
                
                for step in workflow_steps:
                    col_wf1, col_wf2, col_wf3 = st.columns([2, 2, 3])
                    with col_wf1:
                        st.write(step["Step"])
                    with col_wf2:
                        if "✅" in step["Status"]:
                            st.success("Complete")
                        elif "⏳" in step["Status"]:
                            st.info("In Progress")
                        else:
                            st.warning("Pending")
                    with col_wf3:
                        st.caption(step["By"])
def page_mill_operations():
    """Mill Operations Control Tower"""
    st.title("🏭 Mill Operations Control Tower")
    st.markdown("### Real-time Monitoring for Owned/Leased Mill")
    
    # Simulated IoT data
    iot_data = generate_iot_mill_data()
    
    # Real-time dashboard
    st.subheader("🔄 Real-time Operations Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Throughput", "98 TPD", "Target: 100 TPD")
        st.progress(0.98)
    
    with col2:
        st.metric("OEE (Overall Equipment Effectiveness)", "86%", "-2% from target")
        st.progress(0.86)
    
    with col3:
        st.metric("Extraction Rate", "71.2%", "Target: 72%")
        st.progress(0.712)
    
    with col4:
        st.metric("Energy Consumption", "85 kWh/ton", "-5% vs budget")
        st.progress(0.85)
    
    st.markdown("---")
    
    # Tabs for different operational views
    tab_monitor, tab_quality, tab_maintenance, tab_inventory = st.tabs([
        "📊 Process Monitoring", "🔬 Quality Control", "🔧 Maintenance", "📦 Inventory"
    ])
    
    with tab_monitor:
        st.subheader("Process Flow Monitoring")
        
        # Simulated process flow
        process_stages = [
            {"Stage": "Wheat Intake", "Status": "🟢 Normal", "Current": "25 tons/hr"},
            {"Stage": "Cleaning", "Status": "🟢 Normal", "Current": "24 tons/hr"},
            {"Stage": "Conditioning", "Status": "🟡 Warning", "Current": "22 tons/hr"},
            {"Stage": "Milling", "Status": "🟢 Normal", "Current": "4.1 tons/hr"},
            {"Stage": "Sifting", "Status": "🟢 Normal", "Current": "4.0 tons/hr"},
            {"Stage": "Packaging", "Status": "🟢 Normal", "Current": "3.9 tons/hr"},
        ]
        
        # Visual process flow
        cols = st.columns(len(process_stages))
        for idx, stage in enumerate(process_stages):
            with cols[idx]:
                st.markdown(f"**{stage['Stage']}**")
                if "🟢" in stage['Status']:
                    st.success(stage['Status'])
                elif "🟡" in stage['Status']:
                    st.warning(stage['Status'])
                else:
                    st.error(stage['Status'])
                st.caption(stage['Current'])
        
        # Real-time charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("##### Throughput (Last 24 Hours)")
            throughput_data = {
                'Hour': list(range(24)),
                'Throughput (TPH)': [random.uniform(3.5, 4.2) for _ in range(24)]
            }
            st.line_chart(pd.DataFrame(throughput_data).set_index('Hour'))
        
        with col_chart2:
            st.markdown("##### Power Consumption")
            power_data = {
                'Hour': list(range(24)),
                'Power (kWh/ton)': [random.uniform(80, 90) for _ in range(24)]
            }
            st.line_chart(pd.DataFrame(power_data).set_index('Hour'))
    
    with tab_quality:
        st.subheader("Real-time Quality Monitoring")
        
        # Quality parameters
        quality_params = {
            'Parameter': ['Protein %', 'Moisture %', 'Ash Content %', 'Gluten Index', 'Falling Number'],
            'Target': ['11.5-12.5', '≤12.0', '≤0.48', '≥85', '≥250'],
            'Current': ['11.8', '11.2', '0.46', '88', '280'],
            'Status': ['🟢 In Spec', '🟢 In Spec', '🟢 In Spec', '🟢 In Spec', '🟢 In Spec'],
            'Trend': ['↗️ Stable', '↘️ Improving', '→ Stable', '→ Stable', '↗️ Improving']
        }
        
        st.dataframe(pd.DataFrame(quality_params), hide_index=True)
        
        # Statistical Process Control Chart
        st.subheader("SPC Chart - Protein Content")
        
        # Generate SPC data
        spc_data = {
            'Batch': list(range(1, 31)),
            'Protein': [random.uniform(11.3, 12.1) for _ in range(30)],
            'Upper_Limit': [12.5] * 30,
            'Lower_Limit': [11.5] * 30,
            'Target': [12.0] * 30
        }
        
        df_spc = pd.DataFrame(spc_data)
        st.line_chart(df_spc.set_index('Batch')[['Protein', 'Upper_Limit', 'Lower_Limit', 'Target']])
    
    with tab_maintenance:
        st.subheader("Predictive Maintenance Dashboard")
        
        # Equipment health monitoring
        equipment = [
            {"Equipment": "Roller Mill #1", "Health": 92, "Last_Maintenance": "30 days ago", "Next_Due": "15 days"},
            {"Equipment": "Roller Mill #2", "Health": 85, "Last_Maintenance": "45 days ago", "Next_Due": "5 days"},
            {"Equipment": "Plansifter #1", "Health": 78, "Last_Maintenance": "60 days ago", "Next_Due": "Overdue"},
            {"Equipment": "Purifier", "Health": 95, "Last_Maintenance": "25 days ago", "Next_Due": "35 days"},
            {"Equipment": "Conveyor System", "Health": 88, "Last_Maintenance": "40 days ago", "Next_Due": "20 days"},
        ]
        
        for item in equipment:
            col_eq1, col_eq2, col_eq3, col_eq4 = st.columns([2, 2, 2, 1])
            with col_eq1:
                st.write(f"**{item['Equipment']}**")
            with col_eq2:
                st.progress(item['Health']/100)
                st.caption(f"{item['Health']}% health")
            with col_eq3:
                if "Overdue" in item['Next_Due']:
                    st.error(f"Due: {item['Next_Due']}")
                elif int(item['Next_Due'].split()[0]) < 10:
                    st.warning(f"Due in: {item['Next_Due']}")
                else:
                    st.info(f"Due in: {item['Next_Due']}")
            with col_eq4:
                if st.button("Schedule", key=f"maint_{item['Equipment']}"):
                    st.success(f"Maintenance scheduled for {item['Equipment']}")
    
    with tab_inventory:
        st.subheader("Raw Material & Finished Goods Inventory")
        
        col_inv1, col_inv2 = st.columns(2)
        
        with col_inv1:
            st.markdown("##### Raw Wheat Inventory")
            
            wheat_inventory = [
                {"Silo": "Silo A", "Wheat_Type": "Hard Wheat", "Quantity_Tons": 250, "Days_Old": 15},
                {"Silo": "Silo B", "Wheat_Type": "Medium Wheat", "Quantity_Tons": 180, "Days_Old": 8},
                {"Silo": "Silo C", "Wheat_Type": "Soft Wheat", "Quantity_Tons": 120, "Days_Old": 22},
                {"Total": "", "Wheat_Type": "All Types", "Quantity_Tons": 550, "Days_Old": "Avg 15"}
            ]
            
            st.dataframe(pd.DataFrame(wheat_inventory), hide_index=True)
            
            # Blending calculator
            st.markdown("##### Blending Calculator")
            
            hard_pct = st.slider("Hard Wheat %", 0, 100, 60)
            medium_pct = st.slider("Medium Wheat %", 0, 100, 30)
            soft_pct = 100 - hard_pct - medium_pct
            
            if soft_pct < 0:
                st.error("Total must be 100%")
            else:
                st.info(f"Blend: {hard_pct}% Hard, {medium_pct}% Medium, {soft_pct}% Soft")
                
                # Calculate expected protein
                expected_protein = (hard_pct*12.5 + medium_pct*11.5 + soft_pct*10.5)/100
                st.metric("Expected Protein", f"{expected_protein:.1f}%")
        
        with col_inv2:
            st.markdown("##### Finished Flour Inventory")
            
            flour_inventory = [
                {"Batch": "F202406001", "Type": "Noodle Flour", "Quantity_Tons": 45, 
                 "Production_Date": "2024-06-01", "Shelf_Life_Days": 10},
                {"Batch": "F202406002", "Type": "Noodle Flour", "Quantity_Tons": 32, 
                 "Production_Date": "2024-06-02", "Shelf_Life_Days": 9},
                {"Batch": "F202406003", "Type": "Bread Flour", "Quantity_Tons": 28, 
                 "Production_Date": "2024-06-03", "Shelf_Life_Days": 8},
                {"Total": "", "Type": "All Types", "Quantity_Tons": 105, 
                 "Production_Date": "", "Shelf_Life_Days": "Avg 9"}
            ]
            
            st.dataframe(pd.DataFrame(flour_inventory), hide_index=True)
            
            # FIFO dispatch planner
            st.markdown("##### FIFO Dispatch Planning")
            
            dispatch_qty = st.number_input("Dispatch Quantity (Tons)", min_value=1, max_value=100, value=50)
            
            if st.button("Calculate FIFO Dispatch"):
                st.success("**Dispatch Plan:**")
                st.write("1. Batch F202406001: 45 tons")
                st.write("2. Batch F202406002: 5 tons")
                st.write("**Total:** 50 tons (Freshness optimized)")


# --- MAIN APP LOGIC (Sidebar Navigation) ---
st.sidebar.title("UniGrain Connect")
app_mode = st.sidebar.radio("Navigation", [
    "Dashboard & Bidding", 
    "Supplier Network & Vetting", 
    "Farmer Integration",
    "🔄 Toll Processing Management",  # NEW
    "🏭 Mill Operations Dashboard"    # NEW
])

if app_mode == "Dashboard & Bidding":
    page_dashboard()
elif app_mode == "Supplier Network & Vetting":
    page_supplier_network()
elif app_mode == "Farmer Integration":
    page_farmer_integration()
elif app_mode == "🔄 Toll Processing Management":  # NEW
    page_toll_processing_management()
elif app_mode == "🏭 Mill Operations Dashboard":  # NEW
    page_mill_operations()


