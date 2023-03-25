from pdf_reader import get_pdf_text  # import function to extract text from PDF
from sklearn.feature_extraction.text import TfidfVectorizer  # import vectorizer
from sklearn.metrics.pairwise import cosine_similarity  # import cosine similarity function
import numpy as np  # import numpy for numerical operations

if __name__ == '__main__':
    doc_path_name = 'documents/chat_gpt_ubs.pdf'  # path to the PDF file
    # extract text from the first page of the PDF file and replace line breaks with spaces
    doc_text = get_pdf_text(doc_path_name, 1, 1).replace('\n', ' ')
    # replace exclamation marks and question marks with periods
    doc_text = doc_text.replace('!', '.').replace('?', '.')
    sentences = doc_text.split('.')  # split text into sentences
    print('num sentences: ', len(sentences), ', num words:', len(doc_text.split(' ')))
    vectorizer = TfidfVectorizer()  # create a vectorizer
    X = vectorizer.fit_transform(sentences)  # fit the vectorizer to the sentences and transform them into vectors
    print('type X:', type(X), ', shape X:', X.shape)
    sentence_similarity_matrix = cosine_similarity(X)  # calculate the cosine similarity matrix between sentences
    print('type sentence_similarity_matrix:', type(sentence_similarity_matrix),
          ', shape sentence_similarity_matrix:', sentence_similarity_matrix.shape)
    sentence_similarity_scores = np.sum(sentence_similarity_matrix, axis=1)  # sum the similarity scores for each sentence
    print('shape sentence_similarity_scores: ', sentence_similarity_scores.shape,
          ', sum sentence_similarity_scores: ', sentence_similarity_scores.sum())
    sentence_similarity_scores = sentence_similarity_scores / sentence_similarity_scores.sum()  # normalize the similarity scores
    top_n = 5  # choose the number of top sentences to display
    # find the indices of the top sentences based on their similarity scores
    top_sentences = sorted(range(len(sentence_similarity_scores)), key=lambda i: sentence_similarity_scores[i])[-top_n:]
    for i in top_sentences:  # print the top sentences
        print(sentences[i])
