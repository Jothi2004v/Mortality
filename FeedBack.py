import streamlit as st
import pandas as pd
from gspread_pandas import Spread
from google.oauth2 import service_account

def show_feedback():
    st.header("***📋FeedBack***")
    st.markdown("**Please provide your feedback below.**")

    SERVICE_ACCOUNT_FILE = "neonatal-mortality-rate-0ae8d579ab7a.json"
    SPREADSHEET_ID = "1xcLLgFT4mInFsNr4BoaxW8we-hxTGgSaYP0DnJdiWXc" 

    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        spread = Spread(SPREADSHEET_ID, creds=credentials)

        try:
            existing_data = spread.sheet_to_df(index=False).dropna(how="all")
        except Exception:
            existing_data = pd.DataFrame(columns=["Name", "Email", "Feedback"])  

        network_available = True  # Internet is available
    except Exception:
        st.warning("⚠️ No internet connection. Please connect to the network and try again.")
        existing_data = pd.DataFrame(columns=["Name", "Email", "Feedback"])  
        network_available = False 

    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False

    with st.form(key="feedback_form"):
        name = st.text_input(label="👤 Name", placeholder="Enter your name")
        email = st.text_input(label="✉︎ Email", placeholder="Enter your email")
        feedback = st.text_area(label="✍🏻 Feedback", placeholder="Write your feedback here...")

        submit_button = st.form_submit_button(label="Submit Feedback")

        if submit_button:
            if not name or not email or not feedback:
                st.warning("All fields are required.")
            elif not network_available:
                st.warning("❌ Cannot submit feedback without internet. Please connect to the internet and try again.")
            else:
                feedback_data = pd.DataFrame(
                    [{"Name": name, "Email": email, "Feedback": feedback}]
                )
                updated_df = pd.concat([existing_data, feedback_data], ignore_index=True)
                spread.df_to_sheet(updated_df, index=False, sheet="Feedback", replace=True)
                st.session_state["submitted"] = True

    if st.session_state["submitted"]:
        st.success("Thank you for your feedback! 🎉")
        st.balloons()
