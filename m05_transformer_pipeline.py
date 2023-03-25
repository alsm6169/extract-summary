from pdf_reader import get_pdf_text
from transformers import pipeline

import os

print(f'TRANSFORMERS_CACHE: {os.getenv("TRANSFORMERS_CACHE")}')
print(f'HUGGINGFACE_HUB_CACHE: {os.getenv("HUGGINGFACE_HUB_CACHE")}')
print(f'HF_HOME: {os.getenv("HF_HOME")}')


if __name__ == '__main__':
    doc_path_name = 'documents/chat_gpt_ubs.pdf'
    doc_text = get_pdf_text(doc_path_name, 1, 1)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # print(summarizer(INPUT, min_length=50, do_sample=False))
    print(summarizer(doc_text, min_length=50, length_penalty=2.0, num_beams=4))
