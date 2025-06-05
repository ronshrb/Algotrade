import streamlit as st
from mpt_interactive import display_mpt_page

# Set page configuration
st.set_page_config(
    page_title="Modern Portfolio Theory Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0277BD;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        color: #424242;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Display the MPT page
display_mpt_page()

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; color: #777;">
    <p>Modern Portfolio Theory Optimizer</p>
</div>
""", unsafe_allow_html=True)
