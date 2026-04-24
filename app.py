import streamlit as st
import json
import requests
import whisper
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Pampa QA", layout="centered")

st.title("🎙️ Kannada QA System (Pampa Bharata)")

# -------------------------------
# Load models (CACHED)
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
# Load dataset
# -------------------------------
with open("pampa_sarvam_structured.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data]

# -------------------------------
# Cache embeddings
# -------------------------------
@st.cache_resource
def get_embeddings(texts):
    return embed_model.encode(texts, convert_to_tensor=True)

embeddings = get_embeddings(texts)

# -------------------------------
# Sarvam API setup
# -------------------------------
API_KEY = st.secrets["SARVAM_API_KEY"]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

url = "https://api.sarvam.ai/v1/chat/completions"

# -------------------------------
# Input mode
# -------------------------------
mode = st.radio("Choose Input Type", ["Text", "Voice"])

query = ""

if mode == "Text":
    query = st.text_input("Enter Kannada question")

else:
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"])
    
    if audio_file:
        with st.spinner("Transcribing..."):
            result = whisper_model.transcribe(audio_file)
            query = result["text"]
            st.write("Recognized:", query)

# -------------------------------
# MAIN PROCESS
# -------------------------------
if st.button("Get Answer"):

    if not query:
        st.warning("Enter or upload input")
    else:
        with st.spinner("Processing..."):

            # Step 1: Retrieval
            query_embedding = embed_model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(query_embedding, embeddings)[0]

            top_k = scores.topk(3)
            context = " ".join([texts[idx] for idx in top_k.indices])

            # Step 2: AEO (Sarvam)
            prompt = f"""
            Answer in Kannada in 1-2 lines only.

            Question: {query}

            Context:
            {context}
            """

            payload = {
                "model": "sarvam-m",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }

            try:
                response = requests.post(url, headers=headers, json=payload)
                result = response.json()

                answer = result["choices"][0]["message"]["content"]

                st.subheader("📌 Short Answer:")
                st.success(answer)

            except:
                st.error("Sarvam API error")

            # Optional: show retrieved text
            with st.expander("📜 Retrieved Context"):
                st.write(context)
