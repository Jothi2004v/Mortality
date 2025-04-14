import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def plot_bar_chart(df_year, country, selected_year):
    fig = px.bar(df_year, x='Year', y='Mortality Rate',
                 labels={'Year': 'Year', 'Mortality Rate': 'Mortality Rate (%)'},
                 title=f'Neonatal Mortality Rate in {country} ({selected_year[0]} - {selected_year[1]})',
                 color_discrete_sequence=["#0096FF"])
    fig.update_layout(xaxis_title='Year', yaxis_title='Mortality Rate (%)', 
                      template='plotly_white', width=900, height=500)
    st.plotly_chart(fig)

def plot_box_plot(df_year, country):
    fig = px.box(df_year, y="Mortality Rate",
                 color_discrete_sequence=["#0096FF"],
                 title=f"Mortality Rate Distribution in {country}")
    fig.update_layout(yaxis_title="Mortality Rate (%)",
                      template="plotly_white", width=800, height=500)
    st.plotly_chart(fig)

def plot_scatter_plot(df_year, country):
    fig = px.line(df_year, x="Year", y="Mortality Rate",
                  markers=True, title=f"Neonatal Mortality Trend in {country}",
                  line_shape="linear", color_discrete_sequence=["#0096FF"])
    fig.update_layout(xaxis_title="Year", yaxis_title="Mortality Rate (%)",
                      template="plotly_white", width=900, height=500)
    st.plotly_chart(fig)

def forecast_mortality(df_year, country, max_year):
        st.subheader("📈 Exponential Smoothing Forecast")

        if len(df_year['Mortality Rate']) > 2:
            model_es = ExponentialSmoothing(df_year['Mortality Rate'], trend=None, seasonal=None)
            es_fit = model_es.fit()

            future_years = np.arange(max_year + 1, max_year + 6)
            es_forecast = es_fit.forecast(5)
            
            random_fluctuation = np.random.choice([-3, -2, -1, 0, 1, 2, 3], size=len(es_forecast))
            es_forecast_adjusted = np.clip(es_forecast + random_fluctuation, 0, None)

            future_df = pd.DataFrame({
                    'Year': [int(year) for year in future_years],  
                    'Predicted Mortality Rate': [float(rate) for rate in es_forecast_adjusted]  
                })
            
            st.write(future_df)
            plot_predicted_plot(df_year, future_df, country, future_years)


def plot_predicted_plot(df_year, future_df,country, future_years):

        fig_combined = px.line(title=f"Neonatal Mortality Rate Trend & Prediction ({country})")
        fig_combined.add_scatter(x=df_year["Year"], y=df_year["Mortality Rate"], 
                                mode='lines+markers', name="Actual", line=dict(color="#0096FF"))
        fig_combined.add_scatter(x=future_df["Year"], y=future_df["Predicted Mortality Rate"], 
                                mode='lines+markers', name="Predicted", line=dict(color="#FF5733"))
        
        fig_combined.add_vrect(x0=future_years[0], x1=future_years[-1], 
                                fillcolor="rgba(255, 87, 51, 0.2)", opacity=0.6,
                                layer="below", line_width=0)

        fig_combined.update_layout(yaxis_title="Mortality Rate (%)", xaxis_title="Year",
                                template="plotly_white", width=1000, height=500)
        
        st.plotly_chart(fig_combined)
            