import streamlit as st
from PyPDF2 import PdfReader
import re
from collections import Counter


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    num_pages = len(reader.pages)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return text, num_pages


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def summarize_text(text, num_sentences=5):
    text = clean_text(text)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return "No readable text found in the PDF."

    if len(sentences) <= num_sentences:
        return "\n\n".join(sentences)

    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "as", "it", "its", "from", "than",
        "then", "so", "such", "into", "about", "over", "after", "before", "through",
        "you", "he", "she", "they", "we", "i", "his", "her", "their", "our",
        "my", "your", "me", "them", "who", "what", "when", "where", "why", "how"
    }

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    word_freq = Counter(filtered_words)

    sentence_scores = {}

    for sentence in sentences:
        sentence_words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        filtered_sentence_words = [word for word in sentence_words if word not in stop_words]

        if not filtered_sentence_words:
            continue

        score = sum(word_freq.get(word, 0) for word in filtered_sentence_words)
        sentence_scores[sentence] = score / len(filtered_sentence_words)

    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    selected_sentences = ranked_sentences[:num_sentences]

    ordered_summary = [sentence for sentence in sentences if sentence in selected_sentences]

    return "\n\n".join(ordered_summary)


st.set_page_config(
    page_title="AI PDF Summarizer",
    page_icon="📄",
    layout="centered"
)

st.title("AI PDF Summarizer")
st.write("Upload a PDF file and get a quick summary.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
num_sentences = st.slider("Number of summary sentences", min_value=3, max_value=10, value=5)

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")

    if st.button("Summarize PDF"):
        with st.spinner("Reading and summarizing the PDF..."):
            text, num_pages = extract_text_from_pdf(uploaded_file)

            if not text.strip():
                st.error("No readable text was found in the PDF.")
            else:
                summary = summarize_text(text, num_sentences=num_sentences)

                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"Number of pages: {num_pages}")
                with col2:
                    st.info(f"Extracted characters: {len(text)}")

                st.subheader("Summary")
                st.write(summary)

                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )

                with st.expander("Show extracted text"):
                    st.write(text[:5000])