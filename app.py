import streamlit as st
import json
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Pampa Bharata QA", layout="centered")

st.title("📜 Pampa Bharata QA System")
st.write("Ask a question in Kannada and get relevant verse")

# Load model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load dataset
with open("pampa_annotated.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data]

# Create embeddings (cached)
@st.cache_resource
def get_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

embeddings = get_embeddings(texts)

# User input
query = st.text_input("Enter your question in Kannada")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a question")
    else:
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, embeddings)[0]
        best_idx = scores.argmax()

        st.subheader("📖 Relevant Verse:")
        st.write(texts[best_idx])
