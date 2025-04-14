import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.ensemble import AdaBoostRegressor
from visualizations import plot_bar_chart, plot_box_plot, plot_scatter_plot, forecast_mortality

def show_home():
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    with st.expander("Upload the DataSet"):
        file = st.file_uploader("📂 Upload your file (Excel)", type=["xlsx"])
        st.markdown("<style>div.block-container{padding-top:7em;}</style>", unsafe_allow_html=True)

    if file is not None:
        st.session_state.uploaded_file = file

    if st.session_state.uploaded_file is not None:
        df = pd.read_excel(st.session_state.uploaded_file)
        st.write("**Uploaded file name:**", st.session_state.uploaded_file.name)
        st.markdown("---")

        st.subheader("***Data set***")
        st.write(df.head(8))
        st.markdown("---")

        st.sidebar.header("Select Country for Prediction")
        country = st.sidebar.selectbox("Choose a country", df['Country Name'].unique())
        df_country = df[df['Country Name'] == country]


        if not df_country.empty:
            min_year, max_year = int(df_country['Year'].min()), int(df_country['Year'].max())

            st.subheader("***Select Year***")
            selected_year = st.slider("Year", min_value=min_year, max_value=max_year, value=(min_year, max_year))

            df_year = df_country[df_country["Year"].between(selected_year[0], selected_year[1])]
            
            st.subheader(f"***{country} Selected Year Data ({selected_year[0]} - {selected_year[1]})***")
            st.write(df_year)

            csv_filename = f"{country}_MortalityRate.csv"
            csv_data = df_year.to_csv(index=False).encode('utf-8')
            st.download_button("Download Data", data=csv_data, file_name=csv_filename, mime="text/csv")
            st.markdown("---")

            st.markdown("***<h1 style='text-align: center;'>Data Cleaning</h1>***", unsafe_allow_html=True)

            st.subheader("***Remove Unnecessary Columns***")
            columns_to_drop = ['REF_AREA', 'Indicator', 'Unit of measure', 'UPPER_BOUND', 'LOWER_BOUND', 
                            'Observation Status', 'INDICATOR', 'SEX', 'Sex', 'WEALTH_QUINTILE', 
                            'Wealth Quintile', 'DATA_SOURCE', 'UNIT_MEASURE', 'DATA_SOURCE', 'OBS_STATUS','Country Name']

            df_year = df_year.copy()
            df_year.drop(columns=[col for col in columns_to_drop if col in df_year.columns], inplace=True)

            st.write(df_year)
            st.markdown("---")

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
                st.subheader("***Fatures (X)***")   
                X = df_year.drop(columns=['Mortality Rate'], errors='ignore')       
                st.write(X)

            with col_2:
                st.subheader("***Target (y)***")
                y = df_year['Mortality Rate']
                st.write(y)
            st.markdown("---")

            if y is not None:
                st.subheader("***Split into training and testing sets***")        
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                st.write(f"**x_train shape:** {X_train.shape}")
                st.write(f"**x_test shape:**  {X_test.shape}")
                st.write(f"**y_train shape:** {y_train.shape}")
                st.write(f"**y_test shape:** {y_test.shape}")
                st.markdown("---")

                xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                xgb_model.fit(X_train, y_train)
                
                ada_model = AdaBoostRegressor(n_estimators=100, random_state=42)
                ada_model.fit(X_train, y_train)

                y_pred_xgb=xgb_model.predict(X_test)
                y_pred_ada = ada_model.predict(X_test)

                xgb_mape =  mean_absolute_percentage_error(y_test, y_pred_xgb)
                xgb_accuracy = 100 * (1- xgb_mape)

                ada_mape =  mean_absolute_percentage_error(y_test, y_pred_ada)
                ada_accuracy = 100 * (1- ada_mape)

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
            
                st.markdown("*<h1 style='text-align: center;'>📈 Visualizations</h1>*", unsafe_allow_html=True)

                if not df_year.empty:
                    st.subheader("***Bar Graph***")
                    plot_bar_chart(df_year, country, selected_year)

                    st.subheader("***Box Plot***")
                    plot_box_plot(df_year, country)

                    st.subheader("***Scatter Plot***")
                    plot_scatter_plot(df_year, country)

                    st.markdown("<h2>Actual vs Forecasted Mortality Rate</h2>", unsafe_allow_html=True)
                    forecast_mortality(df_year, country, max_year)
