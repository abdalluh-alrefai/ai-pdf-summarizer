import os
import re
from collections import Counter

import streamlit as st
from PyPDF2 import PdfReader

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


ARABIC_STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك", "هو", "هي",
    "هم", "هن", "كما", "كان", "كانت", "يكون", "يمكن", "لقد", "قد", "تم", "ثم",
    "أو", "و", "أن", "إن", "الى", "ما", "لا", "لم", "لن", "كل", "أي", "أحد",
    "بعد", "قبل", "بين", "ضمن", "عند", "لدى", "هناك", "هنا", "أيضًا", "ايضا",
    "بشكل", "حول", "خلال", "أكثر", "اقل", "أقل", "جدا", "جداً", "حتى", "اذا",
    "إذا", "حيث", "مثل", "تمت", "به", "بها", "له", "لها", "منه", "منها", "عليه",
    "عليها", "الذي", "التي", "الذين", "اللاتي", "اللواتي", "وهو", "وهي", "بأن",
    "فإن", "فقد", "كلها", "جميع", "بعض", "أمام", "خاصة", "خصوصًا", "خصوصا"
}

ENGLISH_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of",
    "with", "by", "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "as", "it", "its", "from", "than", "then", "so", "such", "into",
    "about", "over", "after", "before", "through", "you", "he", "she", "they", "we",
    "i", "his", "her", "their", "our", "my", "your", "me", "them", "who", "what",
    "when", "where", "why", "how", "can", "could", "should", "would", "will", "shall",
    "do", "does", "did", "have", "has", "had", "not", "no", "yes", "than", "also"
}


def detect_language(text: str) -> str:
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    english_chars = re.findall(r'[A-Za-z]', text)

    if len(arabic_chars) > len(english_chars):
        return "ar"
    return "en"


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    num_pages = len(reader.pages)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text.strip(), num_pages


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str, lang: str):
    text = clean_text(text)
    if lang == "ar":
        sentences = re.split(r'(?<=[\.!\؟!])\s+', text)
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(text: str, lang: str):
    if lang == "ar":
        words = re.findall(r'[\u0600-\u06FF]+', text)
        stopwords = ARABIC_STOPWORDS
    else:
        words = re.findall(r'[A-Za-z]+', text.lower())
        stopwords = ENGLISH_STOPWORDS

    filtered = []
    for word in words:
        w = word.lower()
        if len(w) >= 3 and w not in stopwords:
            filtered.append(w)
    return filtered


def extract_keywords(text: str, lang: str, top_n: int = 8):
    words = tokenize_words(text, lang)
    if not words:
        return []

    freq = Counter(words)
    keywords = [word for word, _ in freq.most_common(top_n)]
    return keywords


def local_summarize(text: str, lang: str, num_sentences: int = 5):
    sentences = split_sentences(text, lang)

    if not sentences:
        return "لم يتم العثور على نص قابل للقراءة داخل الملف." if lang == "ar" else "No readable text found in the PDF."

    if len(sentences) <= num_sentences:
        return "\n\n".join(sentences)

    words = tokenize_words(text, lang)
    word_freq = Counter(words)

    sentence_scores = {}
    for sentence in sentences:
        if lang == "ar":
            sentence_words = re.findall(r'[\u0600-\u06FF]+', sentence)
            stopwords = ARABIC_STOPWORDS
        else:
            sentence_words = re.findall(r'[A-Za-z]+', sentence.lower())
            stopwords = ENGLISH_STOPWORDS

        filtered_sentence_words = [
            w.lower() for w in sentence_words
            if len(w) >= 3 and w.lower() not in stopwords
        ]

        if not filtered_sentence_words:
            continue

        score = sum(word_freq.get(word, 0) for word in filtered_sentence_words)
        sentence_scores[sentence] = score / len(filtered_sentence_words)

    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    selected_sentences = ranked_sentences[:num_sentences]
    ordered_summary = [sentence for sentence in sentences if sentence in selected_sentences]

    return "\n\n".join(ordered_summary)


def ai_summarize_with_openai(text: str, lang: str, num_sentences: int = 5):
    api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY", None)

    if not api_key or OpenAI is None:
        return None

    client = OpenAI(api_key=api_key)

    if lang == "ar":
        system_prompt = "أنت مساعد احترافي يلخص المستندات العربية بوضوح ودقة."
        user_prompt = f"""
لخص النص التالي في حوالي {num_sentences} نقاط أو جمل قصيرة وواضحة.
ركز على الأفكار الأساسية فقط.
إذا كان النص قصيرًا جدًا، أعد أهم محتواه باختصار.

النص:
{text[:12000]}
"""
    else:
        system_prompt = "You are a professional assistant that summarizes English documents clearly and accurately."
        user_prompt = f"""
Summarize the following text in about {num_sentences} clear bullet points or short sentences.
Focus on the main ideas only.
If the text is very short, return its key content briefly.

Text:
{text[:12000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def render_keywords(keywords):
    if not keywords:
        return

    chips_html = ""
    for kw in keywords:
        chips_html += f'<span class="keyword-chip">{kw}</span> '

    st.markdown(chips_html, unsafe_allow_html=True)


st.set_page_config(
    page_title="AI PDF Summarizer Pro",
    page_icon="📄",
    layout="centered",
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}
.hero-box {
    padding: 1.3rem 1.4rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(31,41,55,0.95), rgba(17,24,39,0.95));
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    margin-bottom: 0.35rem;
}
.hero-subtitle {
    color: #cbd5e1;
    font-size: 1rem;
}
.metric-card {
    padding: 0.9rem 1rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}
.metric-number {
    font-size: 1.4rem;
    font-weight: 700;
}
.metric-label {
    color: #cbd5e1;
    font-size: 0.9rem;
}
.keyword-chip {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    margin: 0.2rem 0.25rem 0.2rem 0;
    border-radius: 999px;
    background: #1d4ed8;
    color: white;
    font-size: 0.88rem;
    font-weight: 600;
}
.summary-box {
    padding: 1rem 1rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
}
.small-note {
    color: #cbd5e1;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <div class="hero-title">AI PDF Summarizer Pro</div>
    <div class="hero-subtitle">
        ارفع ملف PDF، استخرج النص، احصل على ملخص وكلمات مفتاحية، مع دعم العربية والإنجليزية.
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    summarization_mode = st.radio(
        "Summarization mode",
        ["Smart Local", "OpenAI"],
        index=0,
        help="OpenAI يحتاج API Key. إذا لم يوجد المفتاح، سيتم الرجوع للتلخيص المحلي.",
    )
    num_sentences = st.slider("Number of summary sentences", min_value=3, max_value=12, value=5)
    show_text = st.checkbox("Show extracted text", value=False)
    st.caption("Smart Local مجاني. OpenAI يعطي نتائج أفضل عند توفر المفتاح.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")

    if st.button("Summarize PDF", use_container_width=True):
        with st.spinner("Reading and summarizing the PDF..."):
            text, num_pages = extract_text_from_pdf(uploaded_file)

            if not text.strip():
                st.error("No readable text was found in the PDF.")
            else:
                lang = detect_language(text)
                keywords = extract_keywords(text, lang, top_n=8)

                summary = None
                model_used = "Local"

                if summarization_mode == "OpenAI":
                    summary = ai_summarize_with_openai(text, lang, num_sentences=num_sentences)
                    if summary:
                        model_used = "OpenAI"
                    else:
                        summary = local_summarize(text, lang, num_sentences=num_sentences)
                        model_used = "Local fallback"

                else:
                    summary = local_summarize(text, lang, num_sentences=num_sentences)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{num_pages}</div>
                        <div class="metric-label">Pages</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{len(text)}</div>
                        <div class="metric-label">Characters</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{'Arabic' if lang == 'ar' else 'English'}</div>
                        <div class="metric-label">{model_used}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.subheader("Keywords")
                render_keywords(keywords)

                st.subheader("Summary")
                st.markdown(f'<div class="summary-box">{summary.replace(chr(10), "<br><br>")}</div>', unsafe_allow_html=True)

                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                if show_text:
                    with st.expander("Show extracted text"):
                        st.write(text[:8000])

else:
    st.info("ابدأ برفع ملف PDF لتجربة التطبيق.")