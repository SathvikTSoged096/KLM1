import streamlit as st
import json
import requests
import whisper
from sentence_transformers import SentenceTransformer, util

st.title("🎙️ Pampa Bharata AEO QA")

# Load models
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
whisper_model = whisper.load_model("base")

# Load dataset
with open("pampa_sarvam_structured.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
embeddings = embed_model.encode(texts, convert_to_tensor=True)

# API setup
API_KEY = st.secrets["SARVAM_API_KEY"]
url = "https://api.sarvam.ai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Input options
option = st.radio("Choose Input Type", ["Text", "Voice"])

if option == "Text":
    query = st.text_input("Enter question")

else:
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"])
    if audio_file:
        query = whisper_model.transcribe(audio_file)["text"]
        st.write("Recognized Text:", query)

# Search
if st.button("Get Answer"):

    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]

    top_k = scores.topk(3)
    context = " ".join([texts[idx] for idx in top_k.indices])

    # AEO
    prompt = f"""
    Answer briefly in Kannada (1-2 lines).

    Question: {query}
    Context: {context}
    """

    payload = {
        "model": "sarvam-m",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=payload)
    answer = response.json()["choices"][0]["message"]["content"]

    st.subheader("📌 Short Answer:")
    st.write(answer)
