import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_summarizer():
    model_name = "facebook/bart-large-cnn"  # Correct model name from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def summarize_text(text, tokenizer, model):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

st.title("Text Summarization App")

user_input = st.text_area("Enter text to summarize:", height=200)

if st.button("Summarize"):
    if user_input.strip():
        tokenizer, model = load_summarizer()
        summary = summarize_text(user_input, tokenizer, model)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter text to summarize.")
