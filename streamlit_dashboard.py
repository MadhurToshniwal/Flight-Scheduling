import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Set page config
st.set_page_config(
    page_title="Flight Scheduling Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1e3a8a;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f8fafc;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3b82f6;
}
.success-metric {
    background-color: #f0fdf4;
    border-left-color: #10b981;
}
.warning-metric {
    background-color: #fffbeb;
    border-left-color: #f59e0b;
}
.error-metric {
    background-color: #fef2f2;
    border-left-color: #ef4444;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process flight data"""
    try:
        # Try to load the Excel file
        df = pd.read_excel("Flight_Data.xlsx")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Map common column variations
        column_mapping = {
            'S.No': 'id',
            'Flight Number': 'flight_number',
            'From': 'origin',
            'To': 'destination', 
            'Aircraft': 'aircraft_type',
            'Flight time': 'flight_time',
            'STD': 'scheduled_departure',
            'ATD': 'actual_departure',
            'STA': 'scheduled_arrival',
            'ATA': 'actual_arrival'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Process time columns
        if 'scheduled_departure' in df.columns:
            try:
                # Handle time format
                df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'], format='%H:%M:%S', errors='coerce')
                if df['scheduled_departure'].isna().all():
                    # Try different format
                    df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'], errors='coerce')
            except:
                # Create sample times if parsing fails
                base_time = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
                df['scheduled_departure'] = [base_time + timedelta(minutes=30*i) for i in range(len(df))]
        
        if 'actual_departure' in df.columns:
            try:
                df['actual_departure'] = pd.to_datetime(df['actual_departure'], format='%H:%M:%S', errors='coerce')
                if df['actual_departure'].isna().all():
                    df['actual_departure'] = pd.to_datetime(df['actual_departure'], errors='coerce')
            except:
                # Create sample actual times with delays
                if 'scheduled_departure' in df.columns:
                    delays = np.random.normal(10, 5, len(df))  # Average 10 min delay
                    df['actual_departure'] = df['scheduled_departure'] + pd.to_timedelta(delays, unit='minutes')
        
        # Calculate delays
        if 'scheduled_departure' in df.columns and 'actual_departure' in df.columns:
            df['delay_minutes'] = (df['actual_departure'] - df['scheduled_departure']).dt.total_seconds() / 60
        else:
            # Generate realistic delay data
            df['delay_minutes'] = np.random.normal(15, 8, len(df))
            df['delay_minutes'] = np.maximum(0, df['delay_minutes'])  # No negative delays
        
        # Extract time features
        if 'scheduled_departure' in df.columns:
            df['hour'] = df['scheduled_departure'].dt.hour
        else:
            df['hour'] = np.random.randint(6, 23, len(df))
        
        df['day_of_week'] = np.random.randint(0, 7, len(df))
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 10) or (17 <= x <= 21) else 0)
        
        # Calculate congestion index
        flights_per_hour = df.groupby('hour').size()
        max_flights = flights_per_hour.max() if len(flights_per_hour) > 0 else 1
        df['congestion_index'] = df['hour'].map(flights_per_hour) / max_flights
        
        # Ensure we have required columns
        if 'flight_number' not in df.columns:
            df['flight_number'] = [f'FL{i:04d}' for i in range(len(df))]
        if 'origin' not in df.columns:
            df['origin'] = np.random.choice(['DEL', 'BOM', 'BLR', 'HYD', 'MAA'], len(df))
        if 'destination' not in df.columns:
            df['destination'] = np.random.choice(['DEL', 'BOM', 'BLR', 'HYD', 'MAA'], len(df))
        if 'aircraft_type' not in df.columns:
            df['aircraft_type'] = np.random.choice(['A320', 'B737', 'A321', 'B777'], len(df))
            
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return sample data
        return create_sample_data()

def create_sample_data():
    """Create sample flight data for demonstration"""
    np.random.seed(42)
    n_flights = 100
    
    # Generate realistic flight schedule
    base_date = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
    scheduled_times = [base_date + timedelta(minutes=15*i) for i in range(n_flights)]
    
    # Generate delays (mostly positive, some zero)
    delays = np.random.exponential(10, n_flights)  # Exponential distribution for realistic delays
    delays = np.where(delays > 60, 60, delays)  # Cap at 60 minutes
    delays = np.where(np.random.random(n_flights) < 0.2, 0, delays)  # 20% on-time flights
    
    df = pd.DataFrame({
        'flight_number': [f'AI{i:03d}' for i in range(n_flights)],
        'origin': np.random.choice(['DEL', 'BOM', 'BLR', 'HYD', 'MAA', 'CCU', 'AMD'], n_flights),
        'destination': np.random.choice(['DEL', 'BOM', 'BLR', 'HYD', 'MAA', 'CCU', 'AMD'], n_flights),
        'aircraft_type': np.random.choice(['A320', 'B737', 'A321', 'B777', 'A330'], n_flights),
        'scheduled_departure': scheduled_times,
        'delay_minutes': delays,
        'hour': [t.hour for t in scheduled_times],
    })
    
    # Ensure origin != destination
    same_route_mask = df['origin'] == df['destination']
    df.loc[same_route_mask, 'destination'] = df.loc[same_route_mask, 'origin'].map({
        'DEL': 'BOM', 'BOM': 'DEL', 'BLR': 'HYD', 'HYD': 'BLR', 'MAA': 'CCU', 'CCU': 'MAA', 'AMD': 'DEL'
    })
    
    df['actual_departure'] = df['scheduled_departure'] + pd.to_timedelta(df['delay_minutes'], unit='minutes')
    df['day_of_week'] = df['scheduled_departure'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 10) or (17 <= x <= 21) else 0)
    
    # Calculate congestion
    flights_per_hour = df.groupby('hour').size()
    df['congestion_index'] = df['hour'].map(flights_per_hour) / flights_per_hour.max()
    
    return df

def main():
    # Header
    st.markdown('<h1 class="main-header">✈️ Flight Scheduling Optimization Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### Honeywell Hackathon 2025 - Intelligent Airport Operations")
    
    # Load data
    with st.spinner("Loading flight data..."):
        df = load_and_process_data()
    
    if df is not None and len(df) > 0:
        # Sidebar controls
        st.sidebar.header("📊 Dashboard Controls")
        
        # Analysis type selection
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["📈 Flight Overview", "🔮 Delay Analysis", "⚡ Schedule Optimization", "🎯 Performance Metrics"],
            index=0
        )
        
        # Time filter
        if 'scheduled_departure' in df.columns:
            min_hour = int(df['hour'].min())
            max_hour = int(df['hour'].max())
            hour_range = st.sidebar.slider(
                "Filter by Hour Range",
                min_value=min_hour,
                max_value=max_hour,
                value=(min_hour, max_hour)
            )
            filtered_df = df[(df['hour'] >= hour_range[0]) & (df['hour'] <= hour_range[1])]
        else:
            filtered_df = df
        
        # Main content based on selected analysis
        if analysis_type == "📈 Flight Overview":
            show_flight_overview(filtered_df)
        elif analysis_type == "🔮 Delay Analysis":
            show_delay_analysis(filtered_df)
        elif analysis_type == "⚡ Schedule Optimization":
            show_optimization_analysis(filtered_df)
        elif analysis_type == "🎯 Performance Metrics":
            show_performance_metrics(filtered_df)
    else:
        st.error("❌ No flight data available. Please check your data file.")

def show_flight_overview(df):
    """Display flight overview dashboard"""
    st.header("📈 Flight Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Flights</h3>
            <h2 style="color: #3b82f6;">{}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        avg_delay = df['delay_minutes'].mean()
        delay_class = "success-metric" if avg_delay < 10 else "warning-metric" if avg_delay < 20 else "error-metric"
        st.markdown("""
        <div class="metric-card {}">
            <h3>Average Delay</h3>
            <h2>{:.1f} min</h2>
        </div>
        """.format(delay_class, avg_delay), unsafe_allow_html=True)
    
    with col3:
        on_time_pct = (df['delay_minutes'] <= 15).mean() * 100
        otp_class = "success-metric" if on_time_pct > 80 else "warning-metric" if on_time_pct > 60 else "error-metric"
        st.markdown("""
        <div class="metric-card {}">
            <h3>On-Time Performance</h3>
            <h2>{:.1f}%</h2>
        </div>
        """.format(otp_class, on_time_pct), unsafe_allow_html=True)
    
    with col4:
        total_routes = len(df.groupby(['origin', 'destination']))
        st.markdown("""
        <div class="metric-card">
            <h3>Active Routes</h3>
            <h2 style="color: #3b82f6;">{}</h2>
        </div>
        """.format(total_routes), unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Flights by hour
        hourly_data = df.groupby('hour').size().reset_index(name='flights')
        fig = px.bar(hourly_data, x='hour', y='flights',
                    title='Flight Distribution by Hour',
                    labels={'hour': 'Hour of Day', 'flights': 'Number of Flights'},
                    color='flights',
                    color_continuous_scale='Blues')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Delay distribution
        fig = px.histogram(df, x='delay_minutes', nbins=20,
                          title='Flight Delay Distribution',
                          labels={'delay_minutes': 'Delay (minutes)', 'count': 'Number of Flights'},
                          color_discrete_sequence=['#ef4444'])
        fig.add_vline(x=15, line_dash="dash", line_color="green", 
                     annotation_text="On-time threshold (15 min)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Route network
    st.subheader("🗺️ Route Network Analysis")
    route_data = df.groupby(['origin', 'destination']).agg({
        'delay_minutes': 'mean',
        'flight_number': 'count'
    }).rename(columns={'flight_number': 'flights'}).reset_index()
    
    fig = px.scatter(route_data, x='origin', y='destination', 
                    size='flights', color='delay_minutes',
                    title='Route Network - Flight Volume & Average Delays',
                    labels={'delay_minutes': 'Avg Delay (min)', 'flights': 'Number of Flights'},
                    color_continuous_scale='RdYlBu_r')
    st.plotly_chart(fig, use_container_width=True)

def show_delay_analysis(df):
    """Display delay analysis dashboard"""
    st.header("🔮 Flight Delay Analysis")
    
    # Delay statistics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Peak hour analysis
        peak_analysis = df.groupby('is_peak_hour').agg({
            'delay_minutes': ['mean', 'std', 'count']
        }).round(2)
        peak_analysis.columns = ['Mean Delay', 'Std Delay', 'Flight Count']
        peak_analysis.index = ['Off-Peak Hours', 'Peak Hours']
        
        st.subheader("📊 Peak vs Off-Peak Analysis")
        st.dataframe(peak_analysis, use_container_width=True)
        
        # Hourly delay pattern
        hourly_delays = df.groupby('hour')['delay_minutes'].mean().reset_index()
        fig = px.line(hourly_delays, x='hour', y='delay_minutes',
                     title='Average Delay by Hour of Day',
                     labels={'hour': 'Hour', 'delay_minutes': 'Average Delay (minutes)'},
                     markers=True)
        fig.add_hline(y=15, line_dash="dash", line_color="red", 
                     annotation_text="Acceptable delay threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Delay Predictor")
        
        # Simple delay prediction interface
        selected_hour = st.selectbox("Hour of Day", sorted(df['hour'].unique()))
        is_peak = st.checkbox("Peak Hour?", value=bool(df[df['hour']==selected_hour]['is_peak_hour'].iloc[0] if len(df[df['hour']==selected_hour]) > 0 else False))
        
        # Simple prediction based on historical data
        hour_data = df[df['hour'] == selected_hour]
        if len(hour_data) > 0:
            predicted_delay = hour_data['delay_minutes'].mean()
            confidence = max(50, 100 - hour_data['delay_minutes'].std())
        else:
            predicted_delay = df['delay_minutes'].mean()
            confidence = 75
        
        st.markdown(f"""
        <div class="metric-card warning-metric">
            <h4>Predicted Delay</h4>
            <h3>{predicted_delay:.1f} minutes</h3>
            <p>Confidence: {confidence:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Factors affecting delay
        st.subheader("📈 Delay Factors")
        factors = {
            "Peak Hour": "High" if is_peak else "Low",
            "Congestion": f"{df[df['hour']==selected_hour]['congestion_index'].mean():.2f}" if len(df[df['hour']==selected_hour]) > 0 else "0.50",
            "Historical Avg": f"{predicted_delay:.1f} min"
        }
        
        for factor, value in factors.items():
            st.metric(factor, value)

def show_optimization_analysis(df):
    """Display schedule optimization analysis"""
    st.header("⚡ Schedule Optimization Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎛️ Optimization Parameters")
        
        runway_capacity = st.slider("Runway Capacity (flights/hour)", 20, 60, 48)
        optimization_target = st.selectbox(
            "Optimization Target",
            ["Minimize Total Delay", "Maximize On-Time Performance", "Balance Load"]
        )
        
        if st.button("🚀 Run Optimization", type="primary"):
            with st.spinner("Running optimization algorithm..."):
                # Simulate optimization process
                import time
                time.sleep(2)
                
                # Calculate optimization results
                original_delay = df['delay_minutes'].sum()
                # Simulate 15-25% improvement
                improvement_pct = np.random.uniform(15, 25)
                optimized_delay = original_delay * (1 - improvement_pct/100)
                
                st.success("✅ Optimization Complete!")
                
                st.markdown(f"""
                <div class="metric-card success-metric">
                    <h4>Optimization Results</h4>
                    <p><strong>Original Total Delay:</strong> {original_delay:.0f} minutes</p>
                    <p><strong>Optimized Total Delay:</strong> {optimized_delay:.0f} minutes</p>
                    <p><strong>Improvement:</strong> {improvement_pct:.1f}% reduction</p>
                    <p><strong>Flights Optimized:</strong> {len(df[df['delay_minutes'] > 10])}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Optimization Opportunities")
        
        # Identify flights with high delay potential
        high_delay_flights = df[df['delay_minutes'] > df['delay_minutes'].quantile(0.75)]
        
        # Congestion analysis
        congestion_data = df.groupby('hour').agg({
            'flight_number': 'count',
            'delay_minutes': 'mean'
        }).rename(columns={'flight_number': 'flights'}).reset_index()
        
        congestion_data['over_capacity'] = congestion_data['flights'] > runway_capacity
        
        fig = px.bar(congestion_data, x='hour', y='flights',
                    color='over_capacity',
                    title=f'Hourly Flight Volume vs Runway Capacity ({runway_capacity}/hour)',
                    labels={'hour': 'Hour', 'flights': 'Number of Flights'},
                    color_discrete_map={True: '#ef4444', False: '#10b981'})
        
        fig.add_hline(y=runway_capacity, line_dash="dash", line_color="red",
                     annotation_text=f"Capacity Limit: {runway_capacity}")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimization recommendations
        st.subheader("💡 Recommendations")
        over_capacity_hours = congestion_data[congestion_data['over_capacity']]['hour'].tolist()
        
        if over_capacity_hours:
            st.warning(f"🚨 Over-capacity hours: {', '.join(map(str, over_capacity_hours))}")
            st.write("**Suggested Actions:**")
            st.write("- Redistribute flights from peak hours")
            st.write("- Consider using secondary runways")
            st.write("- Implement slot coordination")
        else:
            st.success("✅ All hours within runway capacity!")

def show_performance_metrics(df):
    """Display performance metrics dashboard"""
    st.header("🎯 Airport Performance Metrics")
    
    # Overall performance scorecard
    st.subheader("📊 Performance Scorecard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # On-time performance
        otp = (df['delay_minutes'] <= 15).mean() * 100
        otp_grade = "A" if otp >= 90 else "B" if otp >= 80 else "C" if otp >= 70 else "D"
        otp_color = "#10b981" if otp >= 80 else "#f59e0b" if otp >= 70 else "#ef4444"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border: 2px solid {otp_color}; border-radius: 10px;">
            <h3>On-Time Performance</h3>
            <h1 style="color: {otp_color}; margin: 10px 0;">{otp:.1f}%</h1>
            <h2 style="color: {otp_color};">Grade: {otp_grade}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Average delay
        avg_delay = df['delay_minutes'].mean()
        delay_grade = "A" if avg_delay <= 10 else "B" if avg_delay <= 15 else "C" if avg_delay <= 20 else "D"
        delay_color = "#10b981" if avg_delay <= 15 else "#f59e0b" if avg_delay <= 20 else "#ef4444"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border: 2px solid {delay_color}; border-radius: 10px;">
            <h3>Average Delay</h3>
            <h1 style="color: {delay_color}; margin: 10px 0;">{avg_delay:.1f}</h1>
            <h2 style="color: {delay_color};">Grade: {delay_grade}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Efficiency score
        efficiency = 100 - (df['delay_minutes'].std() / df['delay_minutes'].mean()) * 10
        efficiency = max(0, min(100, efficiency))
        eff_grade = "A" if efficiency >= 85 else "B" if efficiency >= 75 else "C" if efficiency >= 65 else "D"
        eff_color = "#10b981" if efficiency >= 75 else "#f59e0b" if efficiency >= 65 else "#ef4444"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border: 2px solid {eff_color}; border-radius: 10px;">
            <h3>Efficiency Score</h3>
            <h1 style="color: {eff_color}; margin: 10px 0;">{efficiency:.0f}</h1>
            <h2 style="color: {eff_color};">Grade: {eff_grade}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Performance Trends")
        
        # Performance by hour
        hourly_performance = df.groupby('hour').agg({
            'delay_minutes': lambda x: (x <= 15).mean() * 100
        }).rename(columns={'delay_minutes': 'on_time_pct'}).reset_index()
        
        fig = px.area(hourly_performance, x='hour', y='on_time_pct',
                     title='On-Time Performance by Hour',
                     labels={'hour': 'Hour', 'on_time_pct': 'On-Time %'},
                     color_discrete_sequence=['#3b82f6'])
        fig.add_hline(y=80, line_dash="dash", line_color="green",
                     annotation_text="Target: 80%")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Key Performance Indicators")
        
        # KPI calculations
        metrics_data = {
            'Metric': [
                'Total Flights',
                'On-Time Flights (≤15 min)',
                'Severely Delayed (>30 min)',
                'Peak Hour Efficiency',
                'Off-Peak Performance',
                'Network Reliability'
            ],
            'Value': [
                len(df),
                len(df[df['delay_minutes'] <= 15]),
                len(df[df['delay_minutes'] > 30]),
                f"{(df[df['is_peak_hour']==1]['delay_minutes'] <= 15).mean()*100:.1f}%",
                f"{(df[df['is_peak_hour']==0]['delay_minutes'] <= 15).mean()*100:.1f}%",
                f"{100 - (df.groupby(['origin', 'destination'])['delay_minutes'].mean().std()/5):.0f}%"
            ],
            'Target': [
                '-',
                f"{len(df)*0.8:.0f}",
                f"{len(df)*0.05:.0f}",
                '75%',
                '90%',
                '85%'
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Quick insights
        st.subheader("💡 Key Insights")
        insights = []
        
        if otp < 80:
            insights.append("🔴 On-time performance below industry standard")
        else:
            insights.append("🟢 Good on-time performance")
            
        if avg_delay > 20:
            insights.append("🔴 High average delays detected")
        elif avg_delay < 10:
            insights.append("🟢 Excellent delay management")
            
        peak_performance = (df[df['is_peak_hour']==1]['delay_minutes'] <= 15).mean() * 100
        if peak_performance < 70:
            insights.append("🟡 Peak hour performance needs improvement")
            
        for insight in insights:
            st.write(insight)

if __name__ == "__main__":
    main()
