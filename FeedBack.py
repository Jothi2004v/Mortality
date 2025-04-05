import streamlit as st
import pandas as pd
from gspread_pandas import Spread, Client
from google.oauth2 import service_account
def show_feedback():
    st.header("FeedBack")
    st.markdown("Please provide your feedback below.")

    # Establish Google Sheets connection
    SERVICE_ACCOUNT_FILE = "NMR.json"
    SPREADSHEET_ID = "1xcLLgFT4mInFsNr4BoaxW8we-hxTGgSaYP0DnJdiWXc" 

    # Load credentials with Google Drive API access
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )

    # Connect to the Google Sheet
    spread = Spread(SPREADSHEET_ID, creds=credentials)

    # Read existing data from the "Feedback" sheet
    try:
        existing_data = spread.sheet_to_df(index=False).dropna(how="all")
    except Exception:
        existing_data = pd.DataFrame(columns=["Name", "Email", "Feedback"])  

    # Initialize session state variables
    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False

    # Feedback Form
    with st.form(key="feedback_form"):
        name = st.text_input(label="Name*", placeholder="Enter your name")
        email = st.text_input(label="Email*", placeholder="Enter your email")
        feedback = st.text_area(label="Feedback*", placeholder="Write your feedback here...")

        submit_button = st.form_submit_button(label="Submit Feedback")

        # If the submit button is pressed
        if submit_button:
            if not name or not email or not feedback:
                st.warning("All fields are required.")
            else:
                # Create a new row
                feedback_data = pd.DataFrame(
                    [{"Name": name, "Email": email, "Feedback": feedback}]
                )

                # Append to existing data
                updated_df = pd.concat([existing_data, feedback_data], ignore_index=True)

                # Update Google Sheets
                spread.df_to_sheet(updated_df, index=False, sheet="Feedback", replace=True)

                # Set session state variable
                st.session_state["submitted"] = True

    # Show success message and balloons only if form was submitted
    if st.session_state["submitted"]:
        st.success("Thank you for your feedback! 🎉")
        st.balloons()
