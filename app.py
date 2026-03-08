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
    "do", "does", "did", "have", "has", "had", "not", "no", "yes", "also"
}


def detect_language(text: str) -> str:
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    english_chars = re.findall(r'[A-Za-z]', text)
    return "ar" if len(arabic_chars) > len(english_chars) else "en"


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
        sentences = re.split(r'(?<=[\.\!\؟])\s+', text)
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
    return [word for word, _ in freq.most_common(top_n)]


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


def local_key_points(text: str, lang: str, num_points: int = 5):
    summary = local_summarize(text, lang, num_sentences=num_points)
    if lang == "ar":
        points = [f"- {s.strip()}" for s in summary.split("\n\n") if s.strip()]
    else:
        points = [f"- {s.strip()}" for s in summary.split("\n\n") if s.strip()]
    return "\n".join(points)


def local_explain_document(text: str, lang: str):
    summary = local_summarize(text, lang, num_sentences=4)
    if lang == "ar":
        return f"هذا المستند يتحدث باختصار عن:\n\n{summary}"
    return f"This document is mainly about:\n\n{summary}"


def local_generate_questions(text: str, lang: str, num_questions: int = 5):
    sentences = split_sentences(text, lang)
    selected = sentences[:num_questions]

    questions = []
    for i, sentence in enumerate(selected, start=1):
        short_sentence = sentence[:120].strip()
        if lang == "ar":
            questions.append(f"{i}. ما المقصود بهذه الفكرة: {short_sentence}؟")
        else:
            questions.append(f"{i}. What does this idea mean: {short_sentence}?")

    if not questions:
        return "لا توجد أسئلة قابلة للتوليد." if lang == "ar" else "No questions could be generated."

    return "\n".join(questions)


def ai_text_task(prompt_system: str, prompt_user: str):
    api_key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY", None)

    if not api_key or OpenAI is None:
        return None, "OpenAI API key is missing."

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        return None, str(e)


def ai_summarize_with_openai(text: str, lang: str, num_sentences: int = 5):
    if lang == "ar":
        system_prompt = "أنت مساعد احترافي يلخص المستندات العربية بوضوح ودقة."
        user_prompt = f"""لخص النص التالي في حوالي {num_sentences} جمل أو نقاط قصيرة وواضحة.
ركز على الأفكار الأساسية فقط.

النص:
{text[:12000]}
"""
    else:
        system_prompt = "You are a professional assistant that summarizes English documents clearly and accurately."
        user_prompt = f"""Summarize the following text in about {num_sentences} clear bullet points or short sentences.
Focus on the main ideas only.

Text:
{text[:12000]}
"""
    return ai_text_task(system_prompt, user_prompt)


def ai_explain_document(text: str, lang: str):
    if lang == "ar":
        system_prompt = "أنت مساعد يشرح المستندات بطريقة مبسطة وواضحة."
        user_prompt = f"""اشرح هذا المستند بلغة بسيطة ومفهومة لطالب مبتدئ.
اجعل الشرح مختصرًا وواضحًا.

النص:
{text[:12000]}
"""
    else:
        system_prompt = "You explain documents in a simple and beginner-friendly way."
        user_prompt = f"""Explain this document in simple terms for a beginner.
Keep it clear and concise.

Text:
{text[:12000]}
"""
    return ai_text_task(system_prompt, user_prompt)


def ai_generate_questions(text: str, lang: str, num_questions: int = 5):
    if lang == "ar":
        system_prompt = "أنت مساعد تعليمي يولد أسئلة مفيدة من المستندات."
        user_prompt = f"""ولد {num_questions} أسئلة مفيدة وواضحة من هذا المستند.
أعدها على شكل قائمة مرقمة فقط.

النص:
{text[:12000]}
"""
    else:
        system_prompt = "You generate useful study questions from documents."
        user_prompt = f"""Generate {num_questions} useful and clear questions from this document.
Return them as a numbered list only.

Text:
{text[:12000]}
"""
    return ai_text_task(system_prompt, user_prompt)


def ai_key_points(text: str, lang: str, num_points: int = 5):
    if lang == "ar":
        system_prompt = "أنت مساعد يستخرج أهم النقاط من المستندات."
        user_prompt = f"""استخرج {num_points} من أهم النقاط الأساسية من هذا المستند.
أعدها كنقاط مختصرة فقط.

النص:
{text[:12000]}
"""
    else:
        system_prompt = "You extract the most important key points from documents."
        user_prompt = f"""Extract {num_points} key points from this document.
Return them as short bullet points only.

Text:
{text[:12000]}
"""
    return ai_text_task(system_prompt, user_prompt)


def render_keywords(keywords):
    if not keywords:
        return

    chips_html = ""
    for kw in keywords:
        chips_html += f'<span class="keyword-chip">{kw}</span> '
    st.markdown(chips_html, unsafe_allow_html=True)


st.set_page_config(
    page_title="AI Document Analyzer",
    page_icon="📄",
    layout="centered",
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 920px;
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
.output-box {
    padding: 1rem 1rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <div class="hero-title">AI Document Analyzer</div>
    <div class="hero-subtitle">
        ارفع ملف PDF لتحصل على ملخص، كلمات مفتاحية، شرح مبسط، أسئلة، وأهم النقاط.
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Analysis mode",
        ["Smart Local", "OpenAI"],
        index=0,
        help="OpenAI يحتاج API key ورصيد. عند الفشل سيتم الرجوع للحلول المحلية.",
    )
    num_sentences = st.slider("Summary sentences", min_value=3, max_value=12, value=5)
    num_questions = st.slider("Number of questions", min_value=3, max_value=10, value=5)
    num_points = st.slider("Number of key points", min_value=3, max_value=10, value=5)
    show_text = st.checkbox("Show extracted text", value=False)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")

    if st.button("Analyze Document", use_container_width=True):
        with st.spinner("Reading and analyzing the PDF..."):
            text, num_pages = extract_text_from_pdf(uploaded_file)

            if not text.strip():
                st.error("No readable text was found in the PDF.")
            else:
                lang = detect_language(text)
                keywords = extract_keywords(text, lang, top_n=8)

                model_used = "Local"

                if mode == "OpenAI":
                    summary, summary_err = ai_summarize_with_openai(text, lang, num_sentences)
                    explanation, explain_err = ai_explain_document(text, lang)
                    questions, questions_err = ai_generate_questions(text, lang, num_questions)
                    key_points, points_err = ai_key_points(text, lang, num_points)

                    if summary and explanation and questions and key_points:
                        model_used = "OpenAI"
                    else:
                        st.warning("OpenAI is unavailable right now, so Smart Local was used instead.")
                        summary = local_summarize(text, lang, num_sentences)
                        explanation = local_explain_document(text, lang)
                        questions = local_generate_questions(text, lang, num_questions)
                        key_points = local_key_points(text, lang, num_points)
                        model_used = "Local fallback"
                else:
                    summary = local_summarize(text, lang, num_sentences)
                    explanation = local_explain_document(text, lang)
                    questions = local_generate_questions(text, lang, num_questions)
                    key_points = local_key_points(text, lang, num_points)

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
                    language_name = "Arabic" if lang == "ar" else "English"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{language_name}</div>
                        <div class="metric-label">{model_used}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.subheader("Keywords")
                render_keywords(keywords)

                st.subheader("Summary")
                st.markdown(f'<div class="output-box">{summary.replace(chr(10), "<br><br>")}</div>', unsafe_allow_html=True)

                st.subheader("Explain Document")
                st.markdown(f'<div class="output-box">{explanation.replace(chr(10), "<br><br>")}</div>', unsafe_allow_html=True)

                st.subheader("Key Points")
                st.markdown(f'<div class="output-box">{key_points.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

                st.subheader("Generated Questions")
                st.markdown(f'<div class="output-box">{questions.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

                download_text = f"""SUMMARY

{summary}

EXPLANATION

{explanation}

KEY POINTS

{key_points}

QUESTIONS

{questions}
"""

                st.download_button(
                    label="Download Analysis",
                    data=download_text,
                    file_name="document_analysis.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                if show_text:
                    with st.expander("Show extracted text"):
                        st.write(text[:8000])
else:
    st.info("ابدأ برفع ملف PDF لتجربة التطبيق.")