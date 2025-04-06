import streamlit as st
import pandas as pd
from gspread_pandas import Spread
from google.oauth2 import service_account
import socket
import json

# Function to check internet connection
def is_connected():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

# Main feedback function
def show_feedback():
    st.header("***📋 FeedBack***")
    st.markdown("**Please provide your feedback below.**")

    # Check for internet connection
    if not is_connected():
        st.warning("⚠️ No internet connection. Please connect to the network and try again.")
        return

    SPREADSHEET_ID = "1xcLLgFT4mInFsNr4BoaxW8we-hxTGgSaYP0DnJdiWXc"

    try:
        service_account_info = st.secrets["connections"]["gsheets"]

        # Load credentials
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )

        # Connect to Google Sheet
        spread = Spread(SPREADSHEET_ID, creds=credentials)
    except Exception as e:
        st.error(f"❌ Could not connect to Google Sheets: {e}")
        return

    # Read existing data (optional)
    try:
        existing_data = spread.sheet_to_df(index=False).dropna(how="all")
    except Exception:
        existing_data = pd.DataFrame(columns=["Name", "Email", "Feedback"])

    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False

    # Feedback Form
    with st.form(key="feedback_form"):
        name = st.text_input(label="👤 Name", placeholder="Enter your name")
        email = st.text_input(label="✉︎ Email", placeholder="Enter your email")
        feedback = st.text_area(label="✍🏻 Feedback", placeholder="Write your feedback here...")

        submit_button = st.form_submit_button(label="Submit Feedback")

        if submit_button:
            if not name or not email or not feedback:
                st.warning("⚠️ All fields are required.")
            else:
                try:
                    # Create feedback DataFrame
                    feedback_data = pd.DataFrame(
                        [{"Name": name, "Email": email, "Feedback": feedback}]
                    )

                    # Append and update
                    updated_df = pd.concat([existing_data, feedback_data], ignore_index=True)
                    spread.df_to_sheet(updated_df, index=False, sheet="Feedback", replace=True)

                    st.session_state["submitted"] = True
                except Exception as e:
                    st.error(f"❌ Failed to submit feedback: {e}")

    if st.session_state["submitted"]:
        st.success("✅ Thank you for your feedback! 🎉")
        st.balloons()
