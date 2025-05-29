import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Cache the model and tokenizer to optimize performance
@st.cache_resource
def load_summarizer():
    model_name = "facebook/bart-large-cnn"  # Correct model name from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Function to generate a summary
def summarize_text(text, tokenizer, model):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# App Title
st.title("📝 AI-Powered Text Summarizer")

# App Description
st.markdown("""
This app leverages advanced AI technology to provide concise and accurate summaries of your text.
Simply paste your content below and let the magic happen! 🚀
""")

# User Input Section
user_input = st.text_area("🔍 Enter Text to Summarize:", height=200, placeholder="Paste your text here...")

# Summarize Button
if st.button("✨ Generate Summary"):
    if user_input.strip():
        with st.spinner("⏳ Generating summary, please wait..."):
            tokenizer, model = load_summarizer()
            summary = summarize_text(user_input, tokenizer, model)
        st.success("✅ Summary Generated!")
        st.subheader("📋 Summary:")
        st.write(summary)
    else:
        st.warning("⚠️ Please enter text to summarize.")

# Footer
st.markdown("""
---
Made with ❤️ using Streamlit and Hugging Face Transformers.
""")
