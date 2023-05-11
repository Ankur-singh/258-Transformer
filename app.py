import streamlit as st
from utils import generate, get_model_and_enc


@st.cache_resource()
def load_model_and_enc():
    from config import config

    return get_model_and_enc(config, weights="model.pth")


model, enc = load_model_and_enc()

st.title("Text Generation with Transformers")
prompt = st.text_input("Enter a prompt", "It was a dark and stormy night")

cols = st.columns(3)
n_tokens = cols[0].slider("Number of tokens to generate", 50, 500, 200, 10)
temperature = cols[1].slider("Temperature", 0.5, 2.0, 1.2, 0.1)
top_k = cols[2].slider("Top-k", 0, 100, 5, 1)


if st.button("Generate"):
    st.write(generate(model, enc, prompt, n_tokens, temperature, top_k))
