import streamlit as st
from utils import generate, get_model_and_enc


@st.cache_resource()
def load_model_and_enc():
    from config import config

    return get_model_and_enc(config, weights="model.pth")


model, enc = load_model_and_enc()

st.title("Text Generation with Transformers")
prompt = st.text_input("Enter a prompt", "It was a dark and stormy night")
n_tokens = st.slider("Number of tokens to generate", 50, 500, 200, 10)
if st.button("Generate"):
    st.write(generate(model, enc, prompt, n_tokens))
