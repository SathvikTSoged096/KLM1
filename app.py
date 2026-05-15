import streamlit as st
import requests
import base64
import tempfile

# =====================================
# SARVAM API KEY
# =====================================
SARVAM_API_KEY = st.secrets["SARVAM_API_KEY"]

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Sarvam Kannada TTS",
    layout="centered"
)

st.title("🎤 Kannada Text To Speech")

# =====================================
# USER INPUT
# =====================================
text = st.text_area(
    "Enter Kannada Text"
)

# =====================================
# TTS FUNCTION
# =====================================
def generate_tts(text):

    url = "https://api.sarvam.ai/text-to-speech"

    payload = {
        "text": text,
        "target_language_code": "kn-IN",
        "speaker": "meera",
        "model": "bulbul:v3"
    }

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers
    )

    # DEBUG
    st.write("Status Code:", response.status_code)

    if response.status_code != 200:

        st.error("TTS Generation Failed")
        st.write(response.text)
        return None

    result = response.json()

    # =====================================
    # BASE64 AUDIO
    # =====================================
    audio_base64 = result["audios"][0]

    # DECODE AUDIO
    audio_bytes = base64.b64decode(
        audio_base64
    )

    # SAVE TEMP FILE
    temp_audio = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    temp_audio.write(audio_bytes)

    return temp_audio.name

# =====================================
# GENERATE AUDIO
# =====================================
if st.button("Generate Audio"):

    if not text:

        st.warning(
            "Please enter text"
        )

    else:

        with st.spinner(
            "Generating Audio..."
        ):

            audio_path = generate_tts(text)

            if audio_path:

                st.success(
                    "Audio Generated Successfully"
                )

                # PLAY AUDIO
                st.audio(
                    audio_path,
                    format="audio/wav"
                )

                # DOWNLOAD BUTTON
                with open(
                    audio_path,
                    "rb"
                ) as file:

                    st.download_button(
                        label="Download Audio",
                        data=file,
                        file_name="sarvam_audio.wav",
                        mime="audio/wav"
                    )
