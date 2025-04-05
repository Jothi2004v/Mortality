import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import AdaBoostRegressor
import numpy as np

# Page setup
st.set_page_config(page_title="Neonatal Prediction", page_icon="👶", layout="wide")
st.title("Neonatal Mortality Analysis")
st.markdown("<style>div.block-container{padding-top:60px;}</style>", unsafe_allow_html=True)
st.caption("*Predicts neonatal mortality rates using machine learning*")

# Dataset file uploader
file = st.file_uploader("📂 Upload your file (Excel)", type=["xlsx"])

# Check if file is not None
if file is not None:
    df = pd.read_excel(file)
    st.write("**Uploaded file name:**", file.name)
    st.markdown("---")

    # Display the data
    st.subheader("***Data set***")
    st.write(df.head(8))
    st.markdown("---")

    # Sidebar country selection
    st.sidebar.header("Select Country for Prediction")
    country = st.sidebar.selectbox("Choose a country", df['Country Name'].unique())
    df_country = df[df['Country Name'] == country]

    # Select year
    if not df_country.empty:
        # Get min and max years
        min_year, max_year = int(df_country['Year'].min()), int(df_country['Year'].max())

        # Year selection slider
        st.subheader("***Select Year***")
        selected_year = st.slider("", min_value=min_year, max_value=max_year, value=(min_year, max_year))

        # Filter data based on selected years
        df_year = df_country[df_country["Year"].between(selected_year[0], selected_year[1])]
        
        # Display selected data
        st.subheader(f"***{country} Selected Year Data ({selected_year[0]} - {selected_year[1]})***")
        st.write(df_year)

        # Download the data
        csv_filename = f"{country}_MortalityRate.csv"
        csv_data = df_year.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv_data, file_name=csv_filename, mime="text/csv")
        st.markdown("---")

        
        # Data cleaning
        st.markdown("***<h1 style='text-align: center;'>Data Cleaning</h1>***", unsafe_allow_html=True)

        # Drop unnecessary columns
        st.subheader("***Remove Unnecessary Columns***")
        columns_to_drop = ['REF_AREA', 'Indicator', 'Unit of measure', 'UPPER_BOUND', 'LOWER_BOUND', 
                        'Observation Status', 'INDICATOR', 'SEX', 'Sex', 'WEALTH_QUINTILE', 
                        'Wealth Quintile', 'DATA_SOURCE', 'UNIT_MEASURE', 'DATA_SOURCE', 'OBS_STATUS']

        df_year.drop(columns=[col for col in columns_to_drop if col in df_year.columns], inplace=True)

        st.write(df_year)
        st.markdown("---")

        # Column info
        st.subheader("***Column Info***")
        col_info = { 'Column' : df_year.columns, 
                     "Non-Null Count" : df_year.notnull().sum(),
                     'Dtype' : df_year.dtypes}
        col_info_df = pd.DataFrame(col_info)
        st.dataframe(col_info_df)
        st.markdown("---")
        
        st.markdown("***<h1 style='text-align: center;'>Defining Variables</h1>***", unsafe_allow_html=True)
        col_1,col_2 = st.columns([2,6])
        with col_1:
            # Features (X)
            st.subheader("***Fatures (X)***")   
            X = df_year.drop(columns=['Mortality Rate'], errors='ignore')      
            X = pd.get_dummies(X, drop_first=True)      
            st.write(X)
        with col_2:
            # Target (y)
            st.subheader("***Target (y)***")
            y = df_year['Mortality Rate']
            st.write(y)
        st.markdown("---")

        # Train-test split (80% training, 20% testing)
        if y is not None:
            st.subheader("***Split into training and testing sets***")        
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            st.write(f"**x_train shape:** {X_train.shape}")
            st.write(f"**x_test shape:**  {X_test.shape}")
            st.write(f"**y_train shape:** {y_train.shape}")
            st.write(f"**y_test shape:** {y_test.shape}")
            st.markdown("---")

            # Train XGBoost Model
            xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            xgb_model.fit(X_train, y_train)
            
            #Train ADABoost Model
            ada_model = AdaBoostRegressor(n_estimators=100, random_state=42)
            ada_model.fit(X_train, y_train)

            # Prediction
            y_pred_xgb=xgb_model.predict(X_test)
            y_pred_ada = ada_model.predict(X_test)

            #XGBoost
            xgb_mape =  mean_absolute_percentage_error(y_test, y_pred_xgb)
            xgb_accuracy = 100 * (1- xgb_mape)

            #ADABoost
            ada_mape =  mean_absolute_percentage_error(y_test, y_pred_ada)
            ada_accuracy = 100 * (1- ada_mape)

            # Model Metrics
            st.markdown("*<h1 style='text-align: center;'>📊 Model Performance Metrics</h1>*", unsafe_allow_html=True)
            model_1,model_2 = st.columns(2)

            with model_1:
                st.subheader("***XG-Boost Model***")
                st.write(f"**📌 Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred_xgb):.2f}")
                st.write(f"**📌 Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred_xgb):.2f}")
                st.write(f"**📌 R-squared (R² Score):** {r2_score(y_test, y_pred_xgb):.2f}")
                st.write(f"**📌 Mean Absolute Percentage Error (MAPE):** {mean_absolute_percentage_error(y_test,y_pred_xgb):.2f}")
                st.write(f"**📌 Accuracy:** {xgb_accuracy:.2f}%")

            with model_2:
                st.subheader("***ADA-Boost Model***")
                st.write(f"**📌 Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred_ada):.2f}")
                st.write(f"**📌 Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred_ada):.2f}")
                st.write(f"**📌 R-squared (R² Score):** {r2_score(y_test, y_pred_ada):.2f}")
                st.write(f"**📌 Mean Absolute Percentage Error (MAPE):** {mean_absolute_percentage_error(y_test,y_pred_ada):.2f}")
                st.write(f"**📌 Accuracy:** {ada_accuracy:.2f}%")

            st.markdown("---")
        
        # Visualizations
        st.markdown("*<h1 style='text-align: center;'>📈 Visualizations</h1>*", unsafe_allow_html=True)

        # Bar graph
        st.subheader("***Bar Graph***")
        if df_year.empty:
            st.warning("No Data Available")
        else:
           fig = px.bar(df_year, x='Year' , y='Mortality Rate',
                        labels={'Year' : 'Year', 'Mortality Rate' : 'Mortaliy Rate (%)'},
                        title=f'Neonatal Mortality Rate in {country} ({selected_year[0]} - {selected_year[1]})',
                        color_discrete_sequence=["#0096FF"])
           fig.update_layout(xaxis_title='Year', yaxis_title='Mortality Rate (%)', template ='plotly_white',width=900, height=500)
           st.plotly_chart(fig)
        st.markdown("---")

        # Box plot
        st.markdown("<h2>Box Plot:</h2>", unsafe_allow_html=True)
        if df_year.empty:
            st.warning("No Data Available")
        else:
            fig = px.box(df_year, 
                 y="Mortality Rate", 
                 color_discrete_sequence=["#0096FF"], 
                 title=f"Mortality Rate Distribution in {country}")

            fig.update_layout(yaxis_title="Mortality Rate (%)", 
                            template="plotly_white", 
                            width=800, height=500)

            st.plotly_chart(fig)
        st.markdown("---")

        # Scatter plot        
        st.markdown("<h2>Scatter plot:</h2>", unsafe_allow_html=True)

        if df_year.empty:
            st.warning("No Data Available")
        else:
            fig = px.line(df_year, 
                  x="Year", 
                  y="Mortality Rate", 
                  markers=True,  
                  title=f"Neonatal Mortality Trend in {country}",
                  line_shape="linear",
                  color_discrete_sequence=["#0096FF"]) 
            
            fig.update_layout(xaxis_title="Year", 
                      yaxis_title="Mortality Rate (%)", 
                      template="plotly_white", 
                      width=900, height=500)
            st.plotly_chart(fig)

        st.markdown("---")

        # Forecast future mortality rates
        st.subheader("📈 Exponential Smoothing Forecast")

        if y is not None and len(y) > 2:
            model_es = ExponentialSmoothing(df_year['Mortality Rate'], trend='add', damped_trend=True)
            es_fit = model_es.fit()

            future_years = np.arange(max_year + 1, max_year + 6)
            es_forecast = es_fit.forecast(5)

            # Add random fluctuations to avoid a strictly decreasing trend
            random_fluctuation = np.random.uniform(-0.5, 0.5, size=len(es_forecast))
            es_forecast_adjusted = np.clip(es_forecast + random_fluctuation, 0, None)

            future_df = pd.DataFrame({'Year': future_years, 'Predicted Mortality Rate': es_forecast_adjusted})
            st.write(future_df)

            # Plot actual vs. forecasted values using Matplotlib
            st.markdown("<h2>Actual vs Forecasted Mortality Rate</h2>", unsafe_allow_html=True)
            fig_combined = px.line(title=f"Neonatal Mortality Rate Trend & Prediction ({country})")
            fig_combined.add_scatter(x=df_year["Year"], y=df_year["Mortality Rate"], 
                                        mode='lines+markers', name="Actual", line=dict(color="#0096FF"))
            fig_combined.add_scatter(x=future_df["Year"], y=future_df["Predicted Mortality Rate"], 
                                        mode='lines+markers', name="Predicted", line=dict(color="#FF5733"))
            fig_combined.add_vrect(x0=future_years[0], x1=future_years[-1], 
                                    fillcolor="rgba(255, 87, 51, 0.2)", opacity=0.6,
                                    layer="below", line_width=0)
            fig_combined.update_layout(yaxis_title="Mortality Rate (%)",xaxis_title="Year",
                            template="plotly_white", 
                            width=1000, height=500)
            st.plotly_chart(fig_combined)

