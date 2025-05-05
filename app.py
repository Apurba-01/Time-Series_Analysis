import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Title
st.title("Time-Series Analysis Weather Forecaste")

# Sidebar - Upload and options
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset", df.head())

    # Select features and target
    st.sidebar.header("2. Select Features and Target")
    all_columns = df.columns.tolist()
    target = st.sidebar.selectbox("Select target variable", all_columns)
    features = st.sidebar.multiselect("Select feature variables", [col for col in all_columns if col != target])

    # Algorithm selection
    st.sidebar.header("3. Choose Algorithm")
    algorithm = st.sidebar.selectbox(
        "Select regression algorithm",
        ("Linear Regression", "Decision Tree", "Random Forest")
    )

    # Button to trigger training
    if st.sidebar.button("Train Model"):
        if not features:
            st.error("Please select at least one feature variable.")
        else:
            # Prepare data
            X = df[features]
            y = df[target]
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

            # Initialize model
            if algorithm == "Linear Regression":
                model = LinearRegression()
            elif algorithm == "Decision Tree":
                model = DecisionTreeRegressor(random_state=0)
            else:
                model = RandomForestRegressor(n_estimators=100, max_depth=90, random_state=0)

            # Train
            model.fit(train_X, train_y)
            preds = model.predict(test_X)

            # Metrics
            r2 = r2_score(test_y, preds)
            mae = mean_absolute_error(test_y, preds)
            mse = mean_squared_error(test_y, preds)

            st.write("## Model Performance")
            st.write(f"**Algorithm:** {algorithm}")
            st.write(f"R² Score: {r2:.2f}")
            st.write(f"Mean Absolute Error: {mae:.2f}")
            st.write(f"Mean Squared Error: {mse:.2f}")

            # Visualizations
            st.write("### Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.scatter(test_y, preds)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

            st.write("### Residuals Distribution")
            residuals = test_y - preds
            fig2, ax2 = plt.subplots()
            ax2.hist(residuals, bins=20)
            ax2.set_xlabel("Residual")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Residuals Distribution")
            st.pyplot(fig2)
