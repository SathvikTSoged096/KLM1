import streamlit as st
import json
import requests
import whisper
from sentence_transformers import SentenceTransformer, util
import tempfile

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Kannada QA System",
    layout="centered"
)

st.title("🎙️ Kannada QA System (Pampa Bharata)")

# -------------------------------
# LOAD MODELS (CACHED)
# -------------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_whisper():
    return whisper.load_model("tiny")

embed_model = load_embed_model()
whisper_model = load_whisper()

# -------------------------------
# LOAD DATASET
# -------------------------------
with open(
    "pampa_sarvam_structured.json",
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)

texts = [item["text"] for item in data]

# -------------------------------
# CACHE EMBEDDINGS
# -------------------------------
@st.cache_resource
def get_embeddings(texts):
    return embed_model.encode(
        texts,
        convert_to_tensor=True
    )

embeddings = get_embeddings(texts)

# -------------------------------
# SARVAM API SETUP
# -------------------------------
SARVAM_API_KEY = st.secrets["SARVAM_API_KEY"]

sarvam_headers = {
    "Authorization": f"Bearer {SARVAM_API_KEY}",
    "Content-Type": "application/json"
}

sarvam_url = "https://api.sarvam.ai/v1/chat/completions"

# -------------------------------
# ELEVENLABS API SETUP
# -------------------------------
ELEVEN_API_KEY = st.secrets["ELEVEN_API_KEY"]

VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

# -------------------------------
# ELEVENLABS TTS FUNCTION
# -------------------------------
def text_to_speech(text):

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/"
        f"{VOICE_ID}"
    )

    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2"
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers
    )

    return response.content

# -------------------------------
# INPUT MODE
# -------------------------------
mode = st.radio(
    "Choose Input Type",
    ["Text", "Voice"]
)

query = ""

# -------------------------------
# TEXT INPUT
# -------------------------------
if mode == "Text":

    query = st.text_input(
        "Enter your question in Kannada"
    )

# -------------------------------
# VOICE INPUT
# -------------------------------
else:

    audio_file = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3"]
    )

    if audio_file:

        with st.spinner(
            "🎤 Transcribing audio..."
        ):

            result = whisper_model.transcribe(
                audio_file
            )

            query = result["text"]

            st.write(
                "Recognized Text:",
                query
            )

# -------------------------------
# MAIN PROCESS
# -------------------------------
if st.button("Get Answer"):

    if not query:

        st.warning(
            "Please enter or upload input"
        )

    else:

        with st.spinner("⚡ Processing..."):

            # --------------------------------
            # STEP 1: RETRIEVE CONTEXT
            # --------------------------------
            query_embedding = embed_model.encode(
                query,
                convert_to_tensor=True
            )

            scores = util.cos_sim(
                query_embedding,
                embeddings
            )[0]

            top_k = scores.topk(3)

            context = " ".join([
                texts[idx]
                for idx in top_k.indices
            ])

            # --------------------------------
            # STEP 2: SARVAM ANSWER
            # --------------------------------
            prompt = f"""
            Answer in Kannada in 1-2 lines only.

            Question:
            {query}

            Context:
            {context}
            """

            payload = {
                "model": "sarvam-m",

                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                "temperature": 0.2
            }

            try:

                response = requests.post(
                    sarvam_url,
                    headers=sarvam_headers,
                    json=payload
                )

                result = response.json()

                answer = (
                    result["choices"][0]
                    ["message"]["content"]
                )

                # --------------------------------
                # DISPLAY ANSWER
                # --------------------------------
                st.subheader(
                    "📌 Short Answer:"
                )

                st.success(answer)

                # --------------------------------
                # GENERATE ELEVENLABS AUDIO
                # --------------------------------
                audio_data = text_to_speech(
                    answer
                )

                # Save temp mp3
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp3"
                ) as tmp_file:

                    tmp_file.write(audio_data)

                    audio_path = tmp_file.name

                # --------------------------------
                # PLAY AUDIO
                # --------------------------------
                st.audio(
                    audio_path,
                    format="audio/mp3"
                )

                # --------------------------------
                # OPTIONAL CONTEXT
                # --------------------------------
                with st.expander(
                    "📜 Retrieved Context"
                ):

                    st.write(context)

            except Exception as e:

                st.error(
                    "Error generating response"
                )

                st.write(e)
