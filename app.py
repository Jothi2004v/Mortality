import streamlit as st
from streamlit_option_menu import option_menu
import About
import FeedBack
import Home

st.set_page_config(page_title="Neonatal Prediction", page_icon="👶", layout="wide")
st.markdown("<style>div.block-container{margin-top:-60px;}</style>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
        .centered-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
    </style>
    <h1 class="centered-title">Neonatal Mortality Analysis</h1>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

with st.sidebar:
    selected_menu = option_menu(
        menu_title=None,
        options=["Home", "About", "FeedBack"],
        icons=["house-fill","exclamation-circle-fill","envelope-fill"],
        default_index=0
    )

if selected_menu == "Home":
    Home.show_home()

if selected_menu == "About":
    About.show_about()

if selected_menu == "FeedBack":
    FeedBack.show_feedback()

