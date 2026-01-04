import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import seaborn as sns # for the heatmap
import matplotlib.pyplot as plt # for the charts

# --- Configuration ---
st.set_page_config(layout="wide", page_title="UniGrain Connect Prototype")

# --- 1. DATA GENERATION FUNCTIONS ---

@st.cache_data
def generate_market_data(days=365 * 5): # Increased to 5 years for better seasonality patterns
    """Simulates fluctuating daily wheat prices for key regions."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Regions with simulated different volatility levels
    regions = ['Karachi (Sindh)', 'Multan (Punjab)', 'Faisalabad (Punjab)', 'Sukkur (Sindh)']
    
    data = []
    base_price = 100
    np.random.seed(42) 
    
    for date in dates:
        for region in regions:
            # Simulate regional volatility difference (Higher volatility for Sindh)
            volatility_multiplier = 1.0
            if 'Sindh' in region:
                volatility_multiplier = 1.5 
            
            price_offset = {'Karachi (Sindh)': 3, 'Multan (Punjab)': -2, 'Faisalabad (Punjab)': 1, 'Sukkur (Sindh)': -1}[region]
            
            # Simulate seasonality and random walk
            seasonal_factor = 5 * np.sin(date.timetuple().tm_yday * 2 * np.pi / 365)
            noise = np.random.normal(0, 1) * volatility_multiplier
            
            price = base_price + price_offset + seasonal_factor + noise + 0.05 * (date - start_date).days / 365
            
            data.append({
                'Date': date,
                'Region': region,
                'Mandi_Price_PKR_per_Kg': round(price, 2),
                'Volatility_Index': round(abs(noise) + abs(seasonal_factor * 0.1), 2),
                # Added these columns for the Heatmap:
                'Year': date.year,
                'Month': date.month,
                'Week': date.isocalendar()[1]
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

def simulate_storage_strategy(df_market, annual_need_tons, holding_cost_pk_month):
    """Simulates the financial benefit of a buy-and-store strategy."""
    last_full_year = datetime.today().year - 1
    df_sim = df_market[df_market['Year'] >= last_full_year - 1].copy()
    
    # Low Season (Q2) vs High Season (Q4)
    df_low = df_sim[df_sim['Month'].isin([3, 4, 5])]
    avg_low_price = df_low['Mandi_Price_PKR_per_Kg'].mean()
    
    df_high = df_sim[df_sim['Month'].isin([10, 11, 12])]
    avg_high_price = df_high['Mandi_Price_PKR_per_Kg'].mean()
    
    holding_months = 6  
    holding_cost_total_pk_kg = holding_cost_pk_month * holding_months
    
    gross_saving_pk_kg = avg_high_price - avg_low_price
    net_saving_pk_kg = gross_saving_pk_kg - holding_cost_total_pk_kg
    
    storage_volume_kg = annual_need_tons * 1000 
    total_cost_avoidance = net_saving_pk_kg * storage_volume_kg
    
    return {
        'Net_Saving_PKR_per_Kg': net_saving_pk_kg,
        'Gross_Saving_PKR_per_Kg': gross_saving_pk_kg,
        'Avg_Low_Price': avg_low_price,
        'Avg_High_Price': avg_high_price,
        'Total_Avoidance_PKR': total_cost_avoidance
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
        st.markdown("##### Regional Risk Scorecard")
        st.dataframe(df_volatility[['Region', 'CV (%)', 'Mean_Price']], use_container_width=True, hide_index=True)

    st.info(f"**Decision Insight:** The region with the highest CV ({df_volatility['Region'].iloc[0]}) should be prioritized for long-term strategies like **Silo Storage**.")
    st.markdown("---")
    
    # --- ANALYTICS AND BIDDING ---
    col_intelligence, col_marketplace = st.columns([1, 1])

    with col_intelligence:
        st.subheader("🧠 Predictive Intelligence")
        st.markdown("#### **Price Risk Forecasting (30-Day Outlook)**")
        forecast_region = st.selectbox('Select Region:', df_market['Region'].unique())
        
        df_forecast = simple_price_forecasting(df_market, forecast_region)
        df_plot = df_market[df_market['Region'] == forecast_region].rename(columns={'Mandi_Price_PKR_per_Kg': 'Historical Price'})
        df_forecast = df_forecast.rename(columns={'Mandi_Price_PKR_per_Kg': 'Forecast Price'})
        df_full = pd.merge(df_plot[['Date', 'Historical Price']], df_forecast, on='Date', how='outer')
        df_full = df_full.set_index('Date').sort_index()
        
        st.line_chart(df_full)
        
        # --- Target Price Regression Widget ---
        st.markdown("#### **Target Price Estimator**")
        market_price = st.number_input("Input Current Mandi Price", value=current_price, format="%.2f")
        required_tons = st.slider("Required Volume (Tons)", 100, 2000, 500)
        delivery_loc = st.selectbox("Delivery Location", df_suppliers['Location'].unique())

        target_price = regression_target_price(df_tenders, df_suppliers, market_price, required_tons, delivery_loc)
        st.success(f"**Predicted Target Winning Bid: {target_price:.2f} PKR/Kg**")

    with col_marketplace:
        st.subheader("💪 Digital Bidding Marketplace")
        st.markdown("#### **Create New Reverse Auction Tender**")
        with st.form("new_tender_form"):
            req_tons = st.number_input("Required Fine Flour Volume", min_value=100, value=500)
            req_date = st.date_input("Required Delivery Date", datetime.today() + timedelta(days=14))
            submitted = st.form_submit_button("Post Live Tender (Simulated)")
            if submitted:
                st.success(f"Tender for **{req_tons} Tons** posted successfully!")

        st.markdown("---")
        st.markdown("#### **Live Bidding Window**")
        np.random.seed(100)
        bids = []
        for i in range(10): 
            supplier = df_suppliers.sample(1).iloc[0]
            bid_price = target_price + np.random.uniform(-1.5, 2.5) 
            bids.append({
                'Supplier': supplier['Supplier_Name'],
                'Location': supplier['Location'],
                'Quality Rating': supplier['Quality_Rating'],
                'Bid Price': round(bid_price, 2)
            })
        df_bids = pd.DataFrame(bids).sort_values(by='Bid Price')
        st.dataframe(df_bids, use_container_width=True, hide_index=True)
        st.metric("Lowest Bid Price", f"{df_bids['Bid Price'].min():.2f} PKR/Kg")

def page_strategic_planning():
    """Page 3: NEW Strategic Planning Page."""
    st.title("♟️ Strategic Planning & Risk Management")
    st.markdown("### Long-Term Buying Strategy & Market Stress Testing")
    
    # 1. THE GOLDEN CALENDAR (Heatmap)
    st.subheader("📅 The 'Golden Calendar' (Seasonality Heatmap)")
    st.markdown("Identifies the historically cheapest weeks to buy based on 5 years of data.")
    

    # Prepare Heatmap Data
    heatmap_data = df_market.groupby(['Month', 'Week'])['Mandi_Price_PKR_per_Kg'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index="Month", columns="Week", values="Mandi_Price_PKR_per_Kg")
    
    # Display as a colored table (Simple Heatmap replacement)
    st.dataframe(heatmap_pivot.style.background_gradient(cmap='RdYlGn_r', axis=None), use_container_width=True)
    st.caption("Green = Historically Low Prices (Buy Zone) | Red = Historically High Prices (Risk Zone)")
    
    st.markdown("---")

    # 2. THE CRASH TEST SIMULATOR
    st.subheader("💥 'Crash Test' Simulator (Macro-Economic Stress Test)")
    st.markdown("Simulate how external shocks (Fuel, Dollar) impact your wheat procurement costs.")
    
    
    col_sim_input, col_sim_output = st.columns(2)
    
    with col_sim_input:
        st.markdown("#### Adjust Market Shocks")
        diesel_shock = st.slider("Diesel Price Shock (%)", 0, 50, 0, format="%d%%")
        dollar_shock = st.slider("USD/PKR Exchange Rate Shock (%)", 0, 50, 0, format="%d%%")
        
        # Simple Logic: 
        # 10% Diesel increase = 2% Wheat Price increase (Transport)
        # 10% Dollar increase = 4% Wheat Price increase (Export parity/Inputs)
        impact_diesel = diesel_shock * 0.2
        impact_dollar = dollar_shock * 0.4
        total_impact_percent = impact_diesel + impact_dollar
        
        new_price = current_price * (1 + total_impact_percent/100)
    
    with col_sim_output:
        st.markdown("#### Impact Analysis")
        st.metric("Projected Wheat Price", f"{new_price:.2f} PKR", f"+{total_impact_percent:.1f}% Impact", delta_color="inverse")
        
        cost_increase_per_ton = (new_price - current_price) * 1000
        annual_impact_million = (cost_increase_per_ton * 10000) / 1000000 # Assuming 10k tons annual
        
        st.error(f"**Financial Risk:** A {total_impact_percent:.1f}% price hike adds **PKR {annual_impact_million:.1f} Million** to your annual procurement bill (for 10k Tons).")
        
        if total_impact_percent > 5:
            st.warning("Recommendation: Initiate **Forward Buying** contracts immediately to lock in current rates.")
        else:
            st.success("Market remains within absorbable limits.")

    st.markdown("---")
    
    # 3. STORAGE CALCULATOR (Keep this here as it fits strategy)
    st.subheader("🏦 Optimal Storage Strategy Calculator")
    
    # Hardcoded inputs for simplicity in this view
    res = simulate_storage_strategy(df_market, 10000, 0.5)
    st.metric("Potential Annual Savings (Silo Investment)", f"PKR {res['Total_Avoidance_PKR']/1000000:.1f} Million")
    st.caption("Savings achieved by buying in Green Zone (March) vs Red Zone (November).")


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


# --- MAIN APP LOGIC (Sidebar Navigation) ---
st.sidebar.title("UniGrain Connect")
app_mode = st.sidebar.radio("Navigation", ["Dashboard & Bidding", "Strategic Planning", "Supplier Network & Vetting"])

if app_mode == "Dashboard & Bidding":
    page_dashboard()
elif app_mode == "Strategic Planning":
    page_strategic_planning()
elif app_mode == "Supplier Network & Vetting":
    page_supplier_network()
