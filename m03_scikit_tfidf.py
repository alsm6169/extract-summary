from pdf_reader import get_pdf_text

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the spacy model
nlp = spacy.load("en_core_web_lg")

if __name__ == '__main__':
    # Specify the path to the PDF file
    doc_path_name = 'documents/chat_gpt_ubs.pdf'
    # Extract the text from the first page of the PDF file
    doc_text = get_pdf_text(doc_path_name, 1, 1)

    # Process the text using the spacy model
    doc = nlp(doc_text.replace('\n', ' '))

    # Extract the lemmas from each sentence and create a dataset of lemmas
    lemma_dataset = []
    lemma_sent = []
    total_words = 0
    for token in doc:
        if not (token.is_punct or token.is_space):
            lemma_sent.append(token.lemma_)
            total_words += 1
        elif token.is_sent_end:
            lemma_dataset.append(' '.join(lemma_sent))
            lemma_sent = []

    # Vectorize the dataset of lemmas using TF-IDF
    tfIdfVectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z0-9]+\\w*\\b', use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(lemma_dataset)
    # print(tfIdfVectorizer.get_feature_names_out().shape, tfIdfVectorizer.get_feature_names_out())
    # print(tfIdf[0].T.toarray().flatten(order='C').shape, tfIdf[0].T.toarray().flatten(order='C'))
    # print(tfIdf[1].T.toarray().flatten(order='C').shape, tfIdf[1].T.toarray().flatten(order='C'))

    # Calculate the weight of each sentence based on the TF-IDF values of the words in the sentence
    sent_wght_dict = {}
    idx = 0
    for sent in doc.sents:
        tfidf_wght = {k: v for k, v in zip(tfIdfVectorizer.get_feature_names_out(), tfIdf[idx].T.toarray().flatten(order='C'))}
        idx += 1

        sent_wght = 0
        sent_num_words = 0
        for word in sent:
            if not (word.is_punct or word.is_space or word.is_stop or word.is_sent_end):
                sent_wght += tfidf_wght.get(word.lemma_, 0)
                sent_num_words += 1

        sent_wght_dict[sent] = sent_wght

    # Sort the sentences by their weight in descending order and print the top 5 sentences
    summary_by_importance = sorted(sent_wght_dict.items(), key=lambda x: x[1], reverse=True)
    for i in range(5):
        print(f'{i + 1}:{summary_by_importance[i]}')
