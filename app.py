import os
import re
import sqlite3
import hashlib
from uuid import uuid4
from datetime import datetime
from collections import Counter

import streamlit as st
from PyPDF2 import PdfReader

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


DB_PATH = "app_data.db"


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


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            document_name TEXT NOT NULL,
            language TEXT NOT NULL,
            mode_used TEXT NOT NULL,
            pages_count INTEGER NOT NULL,
            characters_count INTEGER NOT NULL,
            keywords TEXT NOT NULL,
            summary TEXT NOT NULL,
            explanation TEXT NOT NULL,
            key_points TEXT NOT NULL,
            questions TEXT NOT NULL,
            extracted_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(username: str, password: str):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username.strip(), hash_password(password), datetime.utcnow().isoformat())
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "This username already exists."
    finally:
        conn.close()


def authenticate_user(username: str, password: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE username = ? AND password_hash = ?",
        (username.strip(), hash_password(password))
    )
    user = cur.fetchone()
    conn.close()
    return user


def save_analysis(user_id, document_name, language, mode_used, pages_count, characters_count,
                  keywords, summary, explanation, key_points, questions, extracted_text):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO analyses (
            user_id, document_name, language, mode_used, pages_count, characters_count,
            keywords, summary, explanation, key_points, questions, extracted_text, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        document_name,
        language,
        mode_used,
        pages_count,
        characters_count,
        keywords,
        summary,
        explanation,
        key_points,
        questions,
        extracted_text,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()


def get_user_analyses(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM analyses
        WHERE user_id = ?
        ORDER BY id DESC
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()
    return rows


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


def reset_current_document():
    for key in [
        "current_text", "current_lang", "current_name", "current_summary",
        "current_explanation", "current_questions", "current_key_points",
        "current_keywords", "current_mode", "current_pages", "current_characters"
    ]:
        if key in st.session_state:
            del st.session_state[key]


init_db()

st.set_page_config(page_title="AI Document Assistant Pro", page_icon="📄", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px;}
.hero-box {
    padding: 1.3rem 1.4rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(31,41,55,0.95), rgba(17,24,39,0.95));
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
.hero-title {font-size: 2.1rem; font-weight: 800; margin-bottom: 0.35rem;}
.hero-subtitle {color: #cbd5e1; font-size: 1rem;}
.metric-card {
    padding: 0.9rem 1rem; border-radius: 16px; background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08); text-align: center;
}
.metric-number {font-size: 1.4rem; font-weight: 700;}
.metric-label {color: #cbd5e1; font-size: 0.9rem;}
.keyword-chip {
    display: inline-block; padding: 0.35rem 0.7rem; margin: 0.2rem 0.25rem 0.2rem 0;
    border-radius: 999px; background: #1d4ed8; color: white; font-size: 0.88rem; font-weight: 600;
}
.output-box {
    padding: 1rem 1rem; border-radius: 16px; background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;
}
.history-box {
    padding: 0.8rem 0.9rem; border-radius: 14px; background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08); margin-bottom: 0.7rem;
}
</style>
""", unsafe_allow_html=True)

if "user" not in st.session_state:
    st.session_state["user"] = None

if st.session_state["user"] is None:
    st.markdown("""
    <div class="hero-box">
        <div class="hero-title">AI Document Assistant Pro</div>
        <div class="hero-subtitle">
            نسخة متقدمة مع حسابات مستخدمين، حفظ التحليلات، قاعدة بيانات، ورفع عدة ملفات.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Create Account"])

    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True):
            user = authenticate_user(login_username, login_password)
            if user:
                st.session_state["user"] = dict(user)
                st.success("Logged in successfully.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        st.subheader("Create Account")
        signup_username = st.text_input("Choose a username", key="signup_username")
        signup_password = st.text_input("Choose a password", type="password", key="signup_password")
        signup_password2 = st.text_input("Confirm password", type="password", key="signup_password2")

        if st.button("Create Account", use_container_width=True):
            if not signup_username.strip() or not signup_password.strip():
                st.error("Please fill all fields.")
            elif len(signup_password) < 6:
                st.error("Password must be at least 6 characters.")
            elif signup_password != signup_password2:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(signup_username, signup_password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

else:
    user = st.session_state["user"]

    st.markdown(f"""
    <div class="hero-box">
        <div class="hero-title">AI Document Assistant Pro</div>
        <div class="hero-subtitle">
            Welcome, {user["username"]}. حلّل ملفات PDF، احفظ النتائج، وارجع لها لاحقًا.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Analysis mode", ["Smart Local", "OpenAI"], index=0)
        num_sentences = st.slider("Summary sentences", min_value=3, max_value=12, value=5)
        num_questions = st.slider("Questions", min_value=3, max_value=10, value=5)
        num_points = st.slider("Key points", min_value=3, max_value=10, value=5)
        show_text = st.checkbox("Show extracted text", value=False)

        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state["user"] = None
            reset_current_document()
            st.rerun()

    tab_upload, tab_history = st.tabs(["Analyze Documents", "Saved Analyses"])

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded_files:
            file_names = [f.name for f in uploaded_files]
            selected_file_name = st.selectbox("Choose a file to analyze", file_names)

            selected_file = None
            for f in uploaded_files:
                if f.name == selected_file_name:
                    selected_file = f
                    break

            col_a, col_b = st.columns([3, 1])
            with col_a:
                analyze_button = st.button("Analyze Selected Document", use_container_width=True)
            with col_b:
                if st.button("Clear Current Analysis", use_container_width=True):
                    reset_current_document()
                    st.rerun()

            if analyze_button and selected_file is not None:
                with st.spinner("Reading and analyzing the document..."):
                    text, num_pages = extract_text_from_pdf(selected_file)

                    if not text.strip():
                        st.error("No readable text was found in the PDF.")
                    else:
                        lang = detect_language(text)
                        keywords = extract_keywords(text, lang, top_n=8)
                        model_used = "Local"

                        if mode == "OpenAI":
                            summary, _ = ai_summarize_with_openai(text, lang, num_sentences)
                            explanation, _ = ai_explain_document(text, lang)
                            questions, _ = ai_generate_questions(text, lang, num_questions)
                            key_points, _ = ai_key_points(text, lang, num_points)

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

                        st.session_state["current_text"] = text
                        st.session_state["current_lang"] = lang
                        st.session_state["current_name"] = selected_file.name
                        st.session_state["current_summary"] = summary
                        st.session_state["current_explanation"] = explanation
                        st.session_state["current_questions"] = questions
                        st.session_state["current_key_points"] = key_points
                        st.session_state["current_keywords"] = keywords
                        st.session_state["current_mode"] = model_used
                        st.session_state["current_pages"] = num_pages
                        st.session_state["current_characters"] = len(text)

                        save_analysis(
                            user_id=user["id"],
                            document_name=selected_file.name,
                            language=lang,
                            mode_used=model_used,
                            pages_count=num_pages,
                            characters_count=len(text),
                            keywords=", ".join(keywords),
                            summary=summary,
                            explanation=explanation,
                            key_points=key_points,
                            questions=questions,
                            extracted_text=text
                        )

                        st.success("Analysis completed and saved to database.")

        if "current_text" in st.session_state:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{st.session_state["current_pages"]}</div>
                    <div class="metric-label">Pages</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{st.session_state["current_characters"]}</div>
                    <div class="metric-label">Characters</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                language_name = "Arabic" if st.session_state["current_lang"] == "ar" else "English"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{language_name}</div>
                    <div class="metric-label">{st.session_state["current_mode"]}</div>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("Document")
            st.write(st.session_state["current_name"])

            st.subheader("Keywords")
            render_keywords(st.session_state["current_keywords"])

            st.subheader("Summary")
            st.markdown(
                f'<div class="output-box">{st.session_state["current_summary"].replace(chr(10), "<br><br>")}</div>',
                unsafe_allow_html=True
            )

            st.subheader("Explain Document")
            st.markdown(
                f'<div class="output-box">{st.session_state["current_explanation"].replace(chr(10), "<br><br>")}</div>',
                unsafe_allow_html=True
            )

            st.subheader("Key Points")
            st.markdown(
                f'<div class="output-box">{st.session_state["current_key_points"].replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True
            )

            st.subheader("Generated Questions")
            st.markdown(
                f'<div class="output-box">{st.session_state["current_questions"].replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True
            )

            download_text = f"""DOCUMENT
{st.session_state["current_name"]}

SUMMARY
{st.session_state["current_summary"]}

EXPLANATION
{st.session_state["current_explanation"]}

KEY POINTS
{st.session_state["current_key_points"]}

QUESTIONS
{st.session_state["current_questions"]}
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
                    st.write(st.session_state["current_text"][:8000])

    with tab_history:
        st.subheader("Saved Analyses")
        analyses = get_user_analyses(user["id"])

        if not analyses:
            st.info("No saved analyses yet.")
        else:
            selected_history_id = None
            options = {}
            for row in analyses:
                label = f'#{row["id"]} - {row["document_name"]} - {row["created_at"][:19]}'
                options[label] = row["id"]

            chosen_label = st.selectbox("Choose a saved analysis", list(options.keys()))
            selected_history_id = options[chosen_label]

            selected_row = None
            for row in analyses:
                if row["id"] == selected_history_id:
                    selected_row = row
                    break

            if selected_row:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{selected_row["pages_count"]}</div>
                        <div class="metric-label">Pages</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{selected_row["characters_count"]}</div>
                        <div class="metric-label">Characters</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    lang_name = "Arabic" if selected_row["language"] == "ar" else "English"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{lang_name}</div>
                        <div class="metric-label">{selected_row["mode_used"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.write(f'**Document:** {selected_row["document_name"]}')
                st.write(f'**Created at:** {selected_row["created_at"][:19]}')

                keywords = [k.strip() for k in selected_row["keywords"].split(",") if k.strip()]
                st.subheader("Keywords")
                render_keywords(keywords)

                st.subheader("Summary")
                st.markdown(
                    f'<div class="output-box">{selected_row["summary"].replace(chr(10), "<br><br>")}</div>',
                    unsafe_allow_html=True
                )

                st.subheader("Explanation")
                st.markdown(
                    f'<div class="output-box">{selected_row["explanation"].replace(chr(10), "<br><br>")}</div>',
                    unsafe_allow_html=True
                )

                st.subheader("Key Points")
                st.markdown(
                    f'<div class="output-box">{selected_row["key_points"].replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True
                )

                st.subheader("Questions")
                st.markdown(
                    f'<div class="output-box">{selected_row["questions"].replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True
                )

                history_download = f"""DOCUMENT
{selected_row["document_name"]}

SUMMARY
{selected_row["summary"]}

EXPLANATION
{selected_row["explanation"]}

KEY POINTS
{selected_row["key_points"]}

QUESTIONS
{selected_row["questions"]}
"""
                st.download_button(
                    label="Download This Saved Analysis",
                    data=history_download,
                    file_name=f'analysis_{selected_row["id"]}.txt',
                    mime="text/plain",
                    use_container_width=True
                )