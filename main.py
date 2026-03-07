from PyPDF2 import PdfReader
import re
from collections import Counter


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "

    return text


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def summarize_text(text, num_sentences=5):
    text = clean_text(text)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= num_sentences:
        return "\n\n".join(sentences)

    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "as", "it", "its", "from", "than",
        "then", "so", "such", "into", "about", "over", "after", "before", "through",
        "you", "he", "she", "they", "we", "i", "his", "her", "their", "our"
    }

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]

    word_freq = Counter(filtered_words)

    sentence_scores = {}
    for sentence in sentences:
        sentence_words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        filtered_sentence_words = [word for word in sentence_words if word not in stop_words]

        if len(filtered_sentence_words) == 0:
            continue

        score = sum(word_freq.get(word, 0) for word in filtered_sentence_words)
        sentence_scores[sentence] = score / len(filtered_sentence_words)

    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    selected_sentences = ranked_sentences[:num_sentences]

    ordered_summary = [sentence for sentence in sentences if sentence in selected_sentences]

    return "\n\n".join(ordered_summary)


def main():
    pdf_path = "sample.pdf"

    print("Reading PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("Summarizing content...")
    summary = summarize_text(text, num_sentences=5)

    print("\n--- SUMMARY ---\n")
    print(summary)


if __name__ == "__main__":
    main()