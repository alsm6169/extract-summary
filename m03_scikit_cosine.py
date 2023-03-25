from pdf_reader import get_pdf_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

if __name__ == '__main__':
    doc_path_name = 'documents/chat_gpt_ubs.pdf'
    doc_text = get_pdf_text(doc_path_name, 1, 1).replace('\n', ' ')
    doc_text = doc_text.replace('!', '.').replace('?', '.')
    sentences = doc_text.split('.')
    print('num sentences: ', len(sentences), ', num words:', len(doc_text.split(' ')))
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    print('type X:', type(X), ', shape X:', X.shape)
    sentence_similarity_matrix = cosine_similarity(X)
    print('type sentence_similarity_matrix:', type(sentence_similarity_matrix),
          ', shape sentence_similarity_matrix:', sentence_similarity_matrix.shape)
    # print(sentence_similarity_matrix)
    sentence_similarity_scores = np.sum(sentence_similarity_matrix, axis=1)
    print('shape sentence_similarity_scores: ', sentence_similarity_scores.shape,
          ', sum sentence_similarity_scores: ', sentence_similarity_scores.sum())
    sentence_similarity_scores = sentence_similarity_scores / sentence_similarity_scores.sum()
    top_n = 5
    top_sentences = sorted(range(len(sentence_similarity_scores)), key=lambda i: sentence_similarity_scores[i])[-top_n:]
    for i in top_sentences:
        print(sentences[i])

