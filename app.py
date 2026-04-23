import streamlit as st
import requests

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="Rasa Analyzer", layout="centered")

st.title("🎭 Pampa Bharata Rasa Analyzer")

st.write("Enter Kannada text to classify Rasa, Sthayi Bhava, and Sanchari Bhava")

# Get API key from secrets
API_KEY = st.secrets["SARVAM_API_KEY"]

url = "https://api.sarvam.ai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# -------------------------------
# Parser function
# -------------------------------
def parse_output(output):
    lines = output.split("\n")

    rasa, sthayi, sanchari = "", "", ""

    for line in lines:
        if "Rasa" in line:
            rasa = line.split(":")[-1].strip()
        elif "Sthayi" in line:
            sthayi = line.split(":")[-1].strip()
        elif "Sanchari" in line:
            sanchari = line.split(":")[-1].strip()

    return rasa, sthayi, sanchari


# -------------------------------
# User Input
# -------------------------------
text = st.text_area("Enter Kannada Text", height=150)

if st.button("Analyze"):

    if not text.strip():
        st.warning("Please enter some text")
    else:
        with st.spinner("Analyzing with Sarvam AI..."):

            prompt = f"""
            STRICT FORMAT:
            Rasa: <value>
            Sthayi Bhava: <value>
            Sanchari Bhava: <value>

            Classify the following Kannada text:

            {text}
            """

            payload = {
                "model": "sarvam-m",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }

            try:
                response = requests.post(url, headers=headers, json=payload)
                result = response.json()

                output_text = result["choices"][0]["message"]["content"]

                rasa, sthayi, sanchari = parse_output(output_text)

                # -------------------------------
                # Display Results
                # -------------------------------
                st.success("✅ Analysis Complete")

                st.subheader("Results:")
                st.write("🎭 **Rasa:**", rasa)
                st.write("💭 **Sthayi Bhava:**", sthayi)
                st.write("🌊 **Sanchari Bhava:**", sanchari)

                st.expander("🔍 Raw Output").write(output_text)

            except Exception as e:
                st.error("Error connecting to Sarvam API")
                st.write(e)
