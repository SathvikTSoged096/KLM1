import streamlit as st
import json

st.title("Pampa Bharata Rasa Analyzer")

# Load data
with open("pampa_final.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Input
user_input = st.text_area("Enter Kannada Text")

if st.button("Search"):
    for item in data:
        if user_input in item["text"]:
            st.write("Rasa:", item["rasa"])
            st.write("Sthayi Bhava:", item["sthayi_bhava"])
            st.write("Sanchari Bhava:", item["sanchari_bhava"])
            break