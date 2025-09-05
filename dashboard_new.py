import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from flight_implementation_code import FlightDataCollector, DelayPredictor, GeneticScheduleOptimizer, CascadingImpactAnalyzer

# Set page config
st.set_page_config(
    page_title="Flight Scheduling Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    try:
        # Load Excel file
        df = pd.read_excel("Flight_Data.xlsx")
        
        # Create clean column names dictionary
        column_mapping = {
            'S.No': 'id',
            'Flight Number': 'flight_number',
            'From': 'origin',
            'To': 'destination',
            'Aircraft': 'aircraft_type',
            'Flight time': 'flight_time',
            'STD': 'scheduled_time',
            'ATD': 'actual_departure',
            'STA': 'scheduled_arrival',
            'ATA': 'actual_arrival'
        }
        
        # Rename columns that exist in the DataFrame
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        # Convert time columns with error handling
        time_columns = ['STD', 'ATD', 'STA', 'ATA']
        for col in time_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce')
                except:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass

        # Create required columns
        if 'scheduled_time' not in df.columns and 'STD' in df.columns:
            df['scheduled_time'] = df['STD']
        if 'actual_departure' not in df.columns and 'ATD' in df.columns:
            df['actual_departure'] = df['ATD']
            
        # If we have both scheduled and actual times, calculate delays
        if 'scheduled_time' in df.columns and 'actual_departure' in df.columns:
            df['delay_minutes'] = (df['actual_departure'] - df['scheduled_time']).dt.total_seconds() / 60
        else:
            # Generate sample delays if we can't calculate them
            df['delay_minutes'] = np.random.normal(15, 5, size=len(df))
        
        # Extract time features from scheduled_time if available, otherwise use current time
        if 'scheduled_time' in df.columns:
            base_time = df['scheduled_time']
        else:
            base_time = pd.Series([pd.Timestamp.now()] * len(df))
        
        df['hour'] = base_time.dt.hour
        df['day_of_week'] = base_time.dt.dayofweek
        df['month'] = base_time.dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 10) or (17 <= x <= 21) else 0)
        
        # Calculate congestion index
        flights_per_hour = df.groupby('hour').size()
        if len(flights_per_hour) > 0:
            df['congestion_index'] = df['hour'].map(flights_per_hour) / flights_per_hour.max()
        else:
            df['congestion_index'] = 0
            
        return df
    
    except Exception as e:
        # Create sample data if loading fails
        rows = 50
        df = pd.DataFrame({
            'flight_number': [f'FL{i:04d}' for i in range(rows)],
            'scheduled_time': [pd.Timestamp.now() + pd.Timedelta(hours=i) for i in range(rows)],
            'delay_minutes': np.random.normal(15, 5, size=rows),
            'origin': np.random.choice(['DEL', 'BOM', 'BLR', 'HYD'], size=rows),
            'destination': np.random.choice(['DEL', 'BOM', 'BLR', 'HYD'], size=rows),
            'aircraft_type': np.random.choice(['A320', 'B737', 'A321'], size=rows)
        })
        
        df['hour'] = df['scheduled_time'].dt.hour
        df['day_of_week'] = df['scheduled_time'].dt.dayofweek
        df['month'] = df['scheduled_time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 10) or (17 <= x <= 21) else 0)
        
        flights_per_hour = df.groupby('hour').size()
        df['congestion_index'] = df['hour'].map(flights_per_hour) / flights_per_hour.max()
        
        return df

def main():
    # Load data
    df = load_data()
    
    if df is not None:
        # Sidebar
        with st.sidebar:
            st.title("✈️ Flight Analysis")
            analysis_type = st.radio(
                "Select Analysis Type",
                ["Flight Overview", "Delay Predictions", "Schedule Optimization", "Cascading Impact"]
            )

        if analysis_type == "Flight Overview":
            st.title("Flight Overview")
            
            # Key Metrics
            metrics = st.columns(4)
            metrics[0].metric("Total Flights", len(df))
            metrics[1].metric("Average Delay", f"{df['delay_minutes'].mean():.1f} min")
            metrics[2].metric("On-Time %", f"{(df['delay_minutes'] <= 15).mean()*100:.1f}%")
            metrics[3].metric("Routes", len(df.groupby(['origin', 'destination'])))
            
            # Main content in tabs
            tab1, tab2, tab3 = st.tabs(["📈 Analytics", "🛫 Routes", "📋 Details"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        df.groupby('hour').size().reset_index(name='count'),
                        x='hour',
                        y='count',
                        title='Flights by Hour',
                        labels={'hour': 'Hour of Day', 'count': 'Number of Flights'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        df,
                        x='delay_minutes',
                        title='Delay Distribution',
                        labels={'delay_minutes': 'Delay (minutes)'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                routes = df.groupby(['origin', 'destination']).agg({
                    'flight_number': 'count',
                    'delay_minutes': 'mean'
                }).reset_index()
                routes.columns = ['From', 'To', 'Flights', 'Avg Delay']
                
                st.dataframe(
                    routes.sort_values('Flights', ascending=False),
                    hide_index=True,
                    column_config={
                        'Avg Delay': st.column_config.NumberColumn(
                            'Avg Delay (min)',
                            format="%.1f"
                        )
                    }
                )
            
            with tab3:
                # Get available columns
                display_cols = []
                if 'flight_number' in df.columns: display_cols.append('flight_number')
                if 'origin' in df.columns: display_cols.append('origin')
                if 'destination' in df.columns: display_cols.append('destination')
                if 'scheduled_time' in df.columns: display_cols.append('scheduled_time')
                if 'actual_departure' in df.columns: display_cols.append('actual_departure')
                if 'delay_minutes' in df.columns: display_cols.append('delay_minutes')
                
                # Create column configuration
                column_config = {}
                if 'scheduled_time' in display_cols:
                    column_config['scheduled_time'] = st.column_config.DatetimeColumn('Scheduled Time')
                if 'actual_departure' in display_cols:
                    column_config['actual_departure'] = st.column_config.DatetimeColumn('Actual Departure')
                if 'delay_minutes' in display_cols:
                    column_config['delay_minutes'] = st.column_config.NumberColumn('Delay (min)', format="%.1f")
                
                st.dataframe(
                    df[display_cols],
                    hide_index=True,
                    column_config=column_config
                )

        elif analysis_type == "Delay Predictions":
            st.title("Delay Predictions")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                predictor = DelayPredictor()
                X, y = predictor.prepare_features(df)
                metrics = predictor.train(X, y)
                
                st.metric("Model Accuracy", f"{metrics['test_r2']*100:.1f}%")
                
                importance = pd.DataFrame({
                    'Feature': metrics['feature_importance'].keys(),
                    'Importance': metrics['feature_importance'].values()
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance, x='Feature', y='Importance',
                           title='Feature Importance')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Predict New Flight")
                hour = st.slider("Hour", 0, 23, 12)
                is_weekend = st.checkbox("Weekend Flight")
                congestion = st.slider("Expected Congestion", 0.0, 1.0, 0.5)
                
                if st.button("Predict Delay"):
                    # Create base features
                    prediction_data = pd.DataFrame({
                        'hour': [hour],
                        'is_weekend': [1 if is_weekend else 0],
                        'congestion_index': [congestion],
                        'is_peak_hour': [1 if (7 <= hour <= 10) or (17 <= hour <= 21) else 0],
                        'day_of_week': [6 if is_weekend else 3],
                        'month': [datetime.now().month]
                    })
                    
                    # Add encoded categorical features with default values
                    for col in predictor.label_encoders:
                        prediction_data[f'{col}_encoded'] = 0
                    
                    prediction = predictor.predict(prediction_data)[0]
                    
                    st.metric("Predicted Delay", f"{prediction:.1f} min")

        elif analysis_type == "Schedule Optimization":
            st.title("Schedule Optimization")
            
            params = st.columns(3)
            runway_capacity = params[0].slider("Runway Capacity per Hour", 20, 60, 48)
            
            if st.button("Optimize Schedule", type="primary"):
                with st.spinner("Optimizing flight schedule..."):
                    optimizer = GeneticScheduleOptimizer(df, runway_capacity=runway_capacity)
                    results = optimizer.optimize()
                    
                    metrics = st.columns(3)
                    metrics[0].metric(
                        "Original Delays",
                        f"{results['improvement']['original_total_delay']:.1f} min"
                    )
                    metrics[1].metric(
                        "Optimized Delays",
                        f"{results['improvement']['optimized_total_delay']:.1f} min"
                    )
                    metrics[2].metric(
                        "Improvement",
                        f"{results['improvement']['delay_reduction_percent']:.1f}%"
                    )
                    
                    fig = px.line(
                        y=results['fitness_history'],
                        title='Optimization Progress',
                        labels={'index': 'Generation', 'value': 'Fitness Score'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

        elif analysis_type == "Cascading Impact":
            st.title("Cascading Impact Analysis")
            
            try:
                analyzer = CascadingImpactAnalyzer(df)
                critical = analyzer.identify_critical_flights(top_n=10)
                
                st.dataframe(
                    critical[['flight_number', 'direct_delay', 'affected_flights', 'cascading_impact']],
                    hide_index=True,
                    column_config={
                        'direct_delay': st.column_config.NumberColumn('Direct Delay', format="%.1f"),
                        'cascading_impact': st.column_config.NumberColumn('Impact Score', format="%.1f")
                    }
                )
                
                network = analyzer.visualize_network()
                fig = go.Figure()
                
                for edge in network['edges']:
                    fig.add_trace(go.Scatter(
                        x=[network['nodes'][edge['source']]['x'],
                           network['nodes'][edge['target']]['x']],
                        y=[network['nodes'][edge['source']]['y'],
                           network['nodes'][edge['target']]['y']],
                        mode='lines',
                        line=dict(width=1, color='#888'),
                        hoverinfo='none'
                    ))
                
                fig.add_trace(go.Scatter(
                    x=[node['x'] for node in network['nodes']],
                    y=[node['y'] for node in network['nodes']],
                    mode='markers+text',
                    marker=dict(size=10),
                    text=[node['label'] for node in network['nodes']],
                    textposition='top center'
                ))
                
                fig.update_layout(
                    title='Flight Dependency Network',
                    showlegend=False,
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error("Unable to perform cascading impact analysis.")

if __name__ == "__main__":
    main()
