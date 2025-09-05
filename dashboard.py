import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from flight_implementation_code import FlightDataCollector, DelayPredictor, GeneticScheduleOptimizer, CascadingImpactAnalyzer

# Set wide layout and custom theme
st.set_page_config(
    page_title="Flight Scheduling Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("✈️ Flight Scheduling Optimization Dashboard")

# Load data
def apply_custom_style():
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Flight_Data.xlsx")
        
        # Convert STD and ATD to datetime, combining with a default date if necessary
        default_date = datetime.now().date()
        df['STD'] = pd.to_datetime(df['STD'].apply(lambda x: f"{default_date} {x}" if isinstance(x, str) else x), errors='coerce')
        df['ATD'] = pd.to_datetime(df['ATD'].apply(lambda x: f"{default_date} {x}" if isinstance(x, str) else x), errors='coerce')
        
        # Check for time-related columns
        found_columns = {
            'scheduled': None,
            'actual': None,
            'date': None,
            'hour': None
        }
        
        # Find available time columns
        time_columns = ['scheduled_time', 'departure_time', 'STD', 'std', 'scheduled_departure']
        actual_columns = ['actual_time', 'actual_departure', 'ATD', 'atd']
        date_columns = ['date', 'flight_date', 'departure_date']
        hour_columns = ['hour', 'departure_hour', 'scheduled_hour']
        
        for col in time_columns:
            if col in df.columns:
                found_columns['scheduled'] = col
                break
                
        for col in actual_columns:
            if col in df.columns:
                found_columns['actual'] = col
                break
                
        for col in date_columns:
            if col in df.columns:
                found_columns['date'] = col
                break
                
        for col in hour_columns:
            if col in df.columns:
                found_columns['hour'] = col
                break
        
        # Create scheduled_time based on available columns
        if not found_columns['scheduled']:
            if found_columns['date'] and found_columns['hour']:
                # Combine date and hour
                date_col = found_columns['date']
                hour_col = found_columns['hour']
                df['date_temp'] = pd.to_datetime(df[date_col])
                df['scheduled_time'] = df.apply(
                    lambda row: row['date_temp'].replace(
                        hour=int(row[hour_col]),
                        minute=0, second=0
                    ),
                    axis=1
                )
            elif found_columns['hour']:
                # Create from hour only
                base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                df['scheduled_time'] = df[found_columns['hour']].apply(
                    lambda x: base_date + timedelta(hours=int(x))
                )
            else:
                # Create sample schedule
                st.warning("No time columns found. Creating sample schedule...")
                base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                df['scheduled_time'] = [base_date + timedelta(minutes=30*i) for i in range(len(df))]
        else:
            df['scheduled_time'] = pd.to_datetime(df[found_columns['scheduled']])
        
        # Step 3: Process flight numbers
        st.info("Step 3: Processing flight information...")
        flight_number_cols = ['flight_number', 'flight_num', 'flight', 'FlightNumber', 'Flight Number']
        if not any(col in df.columns for col in flight_number_cols):
            df['flight_number'] = [f'FL{i:04d}' for i in range(len(df))]
        else:
            df['flight_number'] = df[next(col for col in flight_number_cols if col in df.columns)]
        
        # Step 4: Process delays
        st.info("Step 4: Processing delay information...")
        if 'delay_minutes' not in df.columns:
            if found_columns['actual'] and found_columns['scheduled']:
                df['delay_minutes'] = (
                    pd.to_datetime(df[found_columns['actual']]) - 
                    pd.to_datetime(df[found_columns['scheduled']])
                ).dt.total_seconds() / 60
            else:
                st.warning("No delay information found. Using simulated delays...")
                df['delay_minutes'] = np.random.normal(15, 10, size=len(df))
                df['delay_minutes'] = df['delay_minutes'].clip(0, 120)
        
        # Step 5: Extract features for analysis
        st.info("Step 5: Extracting additional features...")
        df['hour'] = df['scheduled_time'].dt.hour
        df['day_of_week'] = df['scheduled_time'].dt.dayofweek
        df['month'] = df['scheduled_time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_peak_hour'] = df['hour'].apply(
            lambda x: 1 if (7 <= x <= 10) or (17 <= x <= 21) else 0
        )
        
        # Calculate congestion index
        flights_per_hour = df.groupby('hour').size()
        max_flights = flights_per_hour.max()
        df['congestion_index'] = df['hour'].map(flights_per_hour) / max_flights
        
        st.success("✅ Data processing completed successfully!")
        return df

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.write("💡 Please ensure your Excel file contains at least one of these:")
        st.write("- scheduled_time or departure_time")
        st.write("- date and hour columns")
        st.write("- hour column")
        return None

# Main content
def main():
    # Apply custom styling
    apply_custom_style()
    
    # Load the data
    df = load_data()
    
    if df is not None:
        # Sidebar
        st.sidebar.header("Dashboard Controls")
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Flight Overview", "Delay Predictions", "Schedule Optimization", "Cascading Impact"]
        )

        if analysis_type == "Flight Overview":
            st.header("✈️ Flight Overview")
            
            # Summary metrics in a clean layout
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Flights", len(df))
            
            with col2:
                avg_delay = df['delay_minutes'].mean()
                st.metric("Average Delay", f"{avg_delay:.1f} min")
            
            with col3:
                on_time = (df['delay_minutes'] <= 15).mean() * 100
                st.metric("On-Time Performance", f"{on_time:.1f}%")
            
            with col4:
                total_routes = len(df.groupby(['origin', 'destination']))
                st.metric("Total Routes", total_routes)
            
            # Main visualizations
            tabs = st.tabs(["📊 Flight Analysis", "🛫 Route Network", "📋 Flight Details"])
            
            with tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Flight distribution by hour
                    hourly_data = df.groupby('hour').size().reset_index(name='count')
                    fig = px.bar(hourly_data, x='hour', y='count',
                               title='Flights by Hour',
                               labels={'hour': 'Hour of Day', 'count': 'Number of Flights'})
                    fig.update_layout(bargap=0.2)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Delay distribution
                    fig = px.histogram(df, x='delay_minutes',
                                     title='Delay Distribution',
                                     labels={'delay_minutes': 'Delay (minutes)'},
                                     color_discrete_sequence=['#1f77b4'])
                    fig.update_layout(bargap=0.2)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:
                # Route network visualization
                routes = df.groupby(['origin', 'destination']).size().reset_index(name='flights')
                
                fig = go.Figure()
                
                # Add flight routes as lines
                for _, route in routes.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[route['origin'], route['destination']],
                        y=[0, 0],
                        mode='lines+markers',
                        name=f"{route['origin']} → {route['destination']}",
                        hovertemplate=f"Route: {route['origin']} → {route['destination']}<br>"
                                    f"Flights: {route['flights']}"
                    ))
                
                fig.update_layout(
                    title='Flight Route Network',
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[2]:
                # Flight details table with better formatting
                st.dataframe(
                    df[['flight_number', 'origin', 'destination', 'scheduled_time', 
                        'actual_departure', 'delay_minutes']]
                    .sort_values('scheduled_time')
                    .style.format({
                        'scheduled_time': lambda x: x.strftime('%Y-%m-%d %H:%M'),
                        'actual_departure': lambda x: x.strftime('%Y-%m-%d %H:%M'),
                        'delay_minutes': '{:.1f}'
                    }),
                    height=400
                )
            
            with col2:
                st.subheader("Flight Distribution")
                if 'Origin' in df.columns and 'Destination' in df.columns:
                    route_counts = df.groupby(['Origin', 'Destination']).size().reset_index(name='count')
                    st.dataframe(route_counts)

        elif analysis_type == "Delay Predictions":
            st.header("🔮 Delay Predictions")
            
            predictor = DelayPredictor()
            X, y = predictor.prepare_features(df)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Model Training Results")
                metrics = predictor.train(X, y)
                
                # Show metrics
                metrics_df = pd.DataFrame({
                    'Metric': ['Mean Absolute Error (Training)', 'Mean Absolute Error (Test)', 
                             'R² Score (Training)', 'R² Score (Test)'],
                    'Value': [metrics['train_mae'], metrics['test_mae'],
                            metrics['train_r2'], metrics['test_r2']]
                })
                st.dataframe(metrics_df, hide_index=True)
                
                # Feature importance plot
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': metrics['feature_importance'].keys(),
                    'Importance': metrics['feature_importance'].values()
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df, x='Feature', y='Importance',
                           title='Feature Importance in Delay Prediction')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Predict New Flight Delay")
                hour = st.slider("Hour of Day", 0, 23, 12)
                is_weekend = st.checkbox("Is Weekend?")
                congestion = st.slider("Congestion Index", 0.0, 1.0, 0.5)
                
                # Create sample for prediction
                sample = pd.DataFrame({
                    'hour': [hour],
                    'day_of_week': [6 if is_weekend else 3],
                    'month': [datetime.now().month],
                    'is_weekend': [1 if is_weekend else 0],
                    'is_peak_hour': [1 if (7 <= hour <= 10) or (17 <= hour <= 21) else 0],
                    'congestion_index': [congestion]
                })
                
                predicted_delay = predictor.predict(sample)[0]
                
                st.markdown(f"""
                    <div class="metric-card">
                        <h3>Predicted Delay</h3>
                        <h2>{predicted_delay:.1f} minutes</h2>
                    </div>
                """, unsafe_allow_html=True)

        elif analysis_type == "Schedule Optimization":
            st.header("⚡ Schedule Optimization")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Optimization Parameters")
                runway_capacity = st.slider("Runway Capacity (flights/hour)", 20, 60, 48)
                
                if st.button("Run Optimization", type="primary"):
                    optimizer = GeneticScheduleOptimizer(df, runway_capacity=runway_capacity)
                    results = optimizer.optimize()
                    
                    st.success("Optimization Complete!")
                    
                    # Show improvement metrics
                    improvements = results['improvement']
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4>Optimization Results</h4>
                            <p>Original Total Delay: {improvements['original_total_delay']:.1f} minutes</p>
                            <p>Optimized Total Delay: {improvements['optimized_total_delay']:.1f} minutes</p>
                            <p>Delay Reduction: {improvements['delay_reduction_percent']:.1f}%</p>
                            <p>Flights Optimized: {improvements['flights_optimized']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Optimization Progress")
                fig = px.line(y=results['fitness_history'], 
                            labels={'index': 'Generation', 'value': 'Fitness Score'},
                            title='Optimization Progress by Generation')
                st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Cascading Impact":
            st.header("🔄 Cascading Impact Analysis")
            
            try:
                analyzer = CascadingImpactAnalyzer(df)
                
                # Get critical flights
                critical_flights = analyzer.identify_critical_flights(top_n=10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Most Critical Flights")
                    st.dataframe(
                        critical_flights[['flight_number', 'direct_delay', 
                                        'affected_flights', 'cascading_impact']],
                        hide_index=True
                    )
                
                with col2:
                    st.subheader("Impact Network")
                    network_data = analyzer.visualize_network()
                    
                    # Create network visualization using plotly
                    fig = go.Figure()
                    
                    # Add edges
                    for edge in network_data['edges']:
                        fig.add_trace(go.Scatter(
                            x=[network_data['nodes'][edge['source']]['x'],
                               network_data['nodes'][edge['target']]['x']],
                            y=[network_data['nodes'][edge['source']]['y'],
                               network_data['nodes'][edge['target']]['y']],
                            mode='lines',
                            line=dict(width=1, color='#888'),
                            hoverinfo='none'
                        ))
                    
                    # Add nodes
                    fig.add_trace(go.Scatter(
                        x=[node['x'] for node in network_data['nodes']],
                        y=[node['y'] for node in network_data['nodes']],
                        mode='markers+text',
                        marker=dict(
                            size=[min(20, 5 + node['delay']/2) for node in network_data['nodes']],
                            color=[node['delay'] for node in network_data['nodes']],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=[node['label'] for node in network_data['nodes']],
                        hovertext=[f"Flight: {node['label']}<br>Delay: {node['delay']:.1f} min" 
                                 for node in network_data['nodes']],
                        textposition='top center'
                    ))
                    
                    fig.update_layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error in Cascading Impact Analysis: {str(e)}")
                st.info("This analysis requires flight network data with scheduled times and dependencies.")

if __name__ == "__main__":
    main()
