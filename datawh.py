import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
from datetime import datetime
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # Import the model used

# Define the database URL
DATABASE_URL = "sqlite:///ecommerce_warehouse.db"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Function to run SQL queries and return results as a pandas DataFrame
@st.cache_data
def run_query(query):
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
    return df

# Basic Streamlit App
st.title("E-commerce Data Warehouse Dashboard")

st.write("Welcome to the e-commerce data warehouse dashboard. Use the sections below to explore the data and insights.")

# Example: Displaying a sample of the sales_fact table
st.header("Sales Data Sample")
sales_sample_query = "SELECT * FROM sales_fact LIMIT 10;"
sales_sample_df = run_query(sales_sample_query)
st.dataframe(sales_sample_df)

# --- Sales Trends Visualization with Filtering ---
st.header("Sales Trends Over Time")

# SQL query to get daily total sales
sales_trend_query = """
SELECT
    t.full_date,
    SUM(sf.line_total) AS daily_sales
FROM
    sales_fact AS sf
JOIN
    time_dimension AS t ON sf.date_key = t.date_key
GROUP BY
    t.full_date
ORDER BY
    t.full_date;
"""
sales_trend_df = run_query(sales_trend_query)

# Convert full_date to datetime for plotting and filtering
sales_trend_df['full_date'] = pd.to_datetime(sales_trend_df['full_date'])

# Add date range filter
min_date = sales_trend_df['full_date'].min().date()
max_date = sales_trend_df['full_date'].max().date()

start_date = st.sidebar.date_input('Start date', min_date)
end_date = st.sidebar.date_input('End date', max_date)

# Filter data based on selected date range
filtered_sales_trend_df = sales_trend_df[(sales_trend_df['full_date'].dt.date >= start_date) & (sales_trend_df['full_date'].dt.date <= end_date)]

# Create a line chart using Plotly Express with filtered data
fig_sales_trend = px.line(filtered_sales_trend_df, x='full_date', y='daily_sales', title='Daily Sales Trend (Filtered)')
st.plotly_chart(fig_sales_trend)


# --- Customer Segmentation (RFM) Visualization ---
st.header("Customer Segmentation (RFM)")

# Define a reference date for recency calculation (e.g., today's date or a date after the last order)
# Assuming the last order date in the simulated data is the latest date
query_latest_date = """
SELECT MAX(full_date) FROM time_dimension;
"""
latest_date = run_query(query_latest_date).iloc[0, 0]
# Explicitly convert to datetime
latest_date = pd.to_datetime(latest_date)
reference_date = latest_date + pd.Timedelta(days=1) # Use the day after the last order as the reference

# SQL query for RFM Analysis
query_rfm = f"""
SELECT
    c.customer_id,
    c.name,
    JULIANDAY('{reference_date.strftime('%Y-%m-%d')}') - JULIANDAY(MAX(t.full_date)) AS Recency,
    COUNT(DISTINCT sf.order_id) AS Frequency,
    SUM(sf.line_total) AS Monetary
FROM
    customer_dimension AS c
JOIN
    sales_fact AS sf ON c.customer_id = sf.customer_id
JOIN
    time_dimension AS t ON sf.date_key = t.date_key
GROUP BY
    c.customer_id, c.name
ORDER BY
    Recency ASC, Frequency DESC, Monetary DESC;
"""
df_rfm = run_query(query_rfm)

st.write("RFM Analysis Results:")
st.dataframe(df_rfm.head()) # Displaying the RFM values for the first few customers

# Example: Scatter plot of Recency vs Frequency
fig_rfm_scatter = px.scatter(df_rfm, x='Recency', y='Frequency', size='Monetary', hover_name='name',
                             title='RFM Analysis: Recency vs. Frequency (Monetary as Size)')
st.plotly_chart(fig_rfm_scatter)

# --- Product Performance Visualization ---
st.header("Product Performance")

# SQL query for Product Performance Analysis
query_product_performance = """
SELECT
    p.product_id,
    p.product_name,
    p.category,
    SUM(sf.quantity) AS total_quantity_sold,
    SUM(sf.line_total) AS total_revenue
FROM
    product_dimension AS p
JOIN
    sales_fact AS sf ON p.product_id = sf.product_id
GROUP BY
    p.product_id, p.product_name, p.category
ORDER BY
    total_revenue DESC;
"""
df_product_performance = run_query(query_product_performance)

st.write("Top 10 Products by Revenue:")
st.dataframe(df_product_performance.head(10))

# Example: Bar chart of top 10 products by revenue
fig_top_products = px.bar(df_product_performance.head(10), x='product_name', y='total_revenue',
                          title='Top 10 Products by Revenue')
st.plotly_chart(fig_top_products)


# --- Customer Churn Prediction ---
st.header("Customer Churn Prediction")

st.write("Predict whether a customer is likely to churn based on their behavior.")

# Define the expected features for the model
expected_features = ['Recency', 'Frequency', 'Monetary', 'total_orders', 'avg_time_between_orders']

# Add input fields for manual prediction
st.subheader("Manual Prediction")
with st.form("churn_prediction_form"):
    recency = st.number_input("Recency (days since last order)", min_value=0.0, value=30.0)
    frequency = st.number_input("Frequency (number of unique orders)", min_value=1, value=2)
    monetary = st.number_input("Monetary (total spent)", min_value=0.0, value=100.0)
    total_orders = st.number_input("Total Orders", min_value=1, value=3)
    avg_time_between_orders = st.number_input("Average Time Between Orders (days)", min_value=0.0, value=15.0)

    predict_button = st.form_submit_button("Predict Churn")

if predict_button:
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[recency, frequency, monetary, total_orders, avg_time_between_orders]],
                              columns=expected_features)

    # Preprocess the input data using the scaler
    # Need to reload/re-fit scaler if not cached, or ensure it's saved from training
    # For this example, let's assume we'll retrain/refit on the available data for simplicity in the app.
    # In a real application, the scaler should be saved during model training.
    # Let's refit the scaler on the data used for training in the previous steps (assuming it's available)
    # A better approach is to save the scaler. For demonstration, we'll refit on a sample of data.

    # Fetch data to fit the scaler (replace with loading the saved scaler in production)
    query_features = """
    SELECT
        JULIANDAY('{reference_date.strftime('%Y-%m-%d')}') - JULIANDAY(MAX(t.full_date)) AS Recency,
        COUNT(DISTINCT sf.order_id) AS Frequency,
        SUM(sf.line_total) AS Monetary,
        COUNT(DISTINCT sf.order_id) AS total_orders,
        AVG(JULIANDAY(t.full_date) - lag(JULIANDAY(t.full_date), 1, JULIANDAY(t.full_date)) OVER (PARTITION BY sf.customer_id ORDER BY t.full_date)) AS avg_time_between_orders
    FROM
        sales_fact AS sf
    JOIN
        time_dimension AS t ON sf.date_key = t.date_key
    GROUP BY
        sf.customer_id
    ;
    """
    all_features_df = run_query(query_features)
    # Handle potential NaN values in avg_time_between_orders for customers with single orders
    all_features_df['avg_time_between_orders'] = all_features_df['avg_time_between_orders'].fillna(0)


    scaler = StandardScaler()
    scaler.fit(all_features_df[expected_features]) # Fit the scaler on the training data features

    scaled_input_data = scaler.transform(input_data)

    # Load the trained model (assuming it was saved as 'churn_model.pkl')
    # For this example, we'll retrain the model for simplicity in the app.
    # In a real application, the model should be saved and loaded.
    # Retrain the model for demonstration purposes (replace with loading saved model)
    from sklearn.model_selection import train_test_split
    X_temp = all_features_df[expected_features]
    # Assuming churn label is also in all_features_df or can be derived.
    # For simplicity, let's quickly define a churn label based on Recency for demonstration
    churn_threshold_days = 90
    y_temp = (all_features_df['Recency'] >= churn_threshold_days).astype(int)


    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train_temp, y_train_temp) # Train the model

    # Predict churn probability
    churn_probability = model.predict_proba(scaled_input_data)[:, 1]
    churn_class = model.predict(scaled_input_data)[0]

    st.subheader("Prediction Result:")
    if churn_class == 1:
        st.error(f"Prediction: Customer is likely to churn.")
    else:
        st.success(f"Prediction: Customer is unlikely to churn.")
    st.info(f"Churn Probability: {churn_probability[0]:.2f}")

# Add a visualization for churn distribution (based on calculated churn from data)
st.subheader("Churn Distribution in Dataset")
query_churn_distribution = f"""
SELECT
    CASE WHEN JULIANDAY('{reference_date.strftime('%Y-%m-%d')}') - JULIANDAY(MAX(t.full_date)) >= {churn_threshold_days} THEN 'Churned' ELSE 'Not Churned' END AS churn_status,
    COUNT(DISTINCT sf.customer_id) AS number_of_customers
FROM
    sales_fact AS sf
JOIN
    time_dimension AS t ON sf.date_key = t.date_key
GROUP BY
    churn_status;
"""
df_churn_distribution = run_query(query_churn_distribution)

fig_churn_distribution = px.pie(df_churn_distribution, values='number_of_customers', names='churn_status',
                                title='Distribution of Churned vs. Non-Churned Customers')
st.plotly_chart(fig_churn_distribution)
