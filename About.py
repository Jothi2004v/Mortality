import streamlit as st
import pandas as pd

def show_about():
        st.header("About")

        st.write("Welcome to **Neonatal Prediction using Machine Learning**," \
        " an advanced predictive tool designed to ""assess neonatal health risks using cutting-edge machine learning techniques." \
        " ""This application leverages **XGBoost** and **AdaBoost** algorithms to provide accurate predictions """ \
        "based on input medical parameters.")  
        
        st.subheader("Why Neonatal Prediction?")
        st.write("Neonatal health is a critical area in medical science, as early  identification of potential risks can significantly" \
        " ""improve infant survival rates and reduce complications. Our AI-powered tool assists healthcare professionals in:")

        st.markdown("✅ Predicting neonatal health risks early.")
        st.markdown("✅ Supporting clinical decision-making.")
        st.markdown("✅ Improving newborn care and reducing complications.")

        st.subheader("Machine Learning Models Used")
        st.markdown("🔹 **XGBoost (Extreme Gradient Boosting)** – A highly efficient and scalable gradient boosting algorithm that provides high accuracy.")
        st.markdown("🔹 **AdaBoost (Adaptive Boosting)** – A boosting technique that improves weak learners to create a strong predictive model.")

        st.subheader("How It Works?")
        st.markdown("1️⃣ Input relevant neonatal health parameters.")
        st.markdown("2️⃣ The system processes the data using **XGBoost** and **AdaBoost** models.")
        st.markdown("3️⃣ Predictions and insights are displayed for better decision-making")

        st.subheader("Disclaimer")
        st.write("This application is a **decision-support tool** and should not be used as a substitute for professional medical advice. " \
        "" "Always consult a healthcare expert for accurate diagnosis and treatment.")

        st.subheader("Download the data set Here")
        data = pd.read_excel("sample.xlsx")
        csv_filename = "Mortality_Rate.csv"
        csv_data = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Dataset", data=csv_data, file_name=csv_filename,mime="text/csv")