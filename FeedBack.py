import streamlit as st

def show_feedback():
    st.header("***📋 FeedBack***")
    st.markdown("**Please provide your feedback below.**")

    if "submitted" not in st.session_state:
        st.session_state["submitted"] = False

    with st.form(key="feedback_form"):
        name = st.text_input(label="👤 Name", placeholder="Enter your name")
        email = st.text_input(label="✉︎ Email", placeholder="Enter your email")
        feedback = st.text_area(label="✍🏻 Feedback", placeholder="Write your feedback here...")

        submit_button = st.form_submit_button(label="Submit Feedback")

        if submit_button:
            if not name or not email or not feedback:
                st.warning("⚠️ All fields are required.")
            else:
                st.session_state["submitted"] = True
                st.session_state["feedback_data"] = {
                    "Name": name,
                    "Email": email,
                    "Feedback": feedback
                }

    if st.session_state["submitted"]:
        st.success("✅ Thank you for your feedback! 🎉")
        st.balloons()
        st.write("### 👇 Your submitted feedback:")
        st.write(st.session_state["feedback_data"])
