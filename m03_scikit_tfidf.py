import numpy as np
import math
from pdf_reader import get_pdf_text

'''$ python -m spacy download en-core-web-lg'''
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


nlp = spacy.load("en_core_web_lg")

if __name__ == '__main__':
    doc_path_name = 'documents/chat_gpt_ubs.pdf'
    doc_text = get_pdf_text(doc_path_name, 1, 1)

    doc = nlp(doc_text.replace('\n', ' '))
    lemma_dataset = []
    lemma_sent = []
    total_words = 0
    for token in doc:
        # print(token.text, token.lemma_, token.is_alpha, token.is_stop, token.is_punct, token.is_sent_end)

        if not (token.is_punct or token.is_space):
            lemma_sent.append(token.lemma_)
            total_words += 1
        elif token.is_sent_end:
            lemma_dataset.append(' '.join(lemma_sent))
            lemma_sent = []

    # print(len(lemma_dataset))
    # for i in range(len(lemma_dataset)):
    #     print(str(i) + ':' + lemma_dataset[i])

    tfIdfVectorizer = TfidfVectorizer(token_pattern=u'(?ui)\\b\\w*[a-z0-9]+\\w*\\b', use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(lemma_dataset)
    # print(tfIdfVectorizer.get_feature_names_out().shape, tfIdfVectorizer.get_feature_names_out())
    # print(tfIdf[0].T.toarray().flatten(order='C').shape, tfIdf[0].T.toarray().flatten(order='C'))
    # print(tfIdf[1].T.toarray().flatten(order='C').shape, tfIdf[1].T.toarray().flatten(order='C'))

    sent_wght_dict = {}
    idx = 0
    for sent in doc.sents:
        # print(f'idx={idx}>[{sent}]')
        # df = pd.DataFrame(tfIdf[idx].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
        tfidf_wght = {k: v for k, v in
                       zip(tfIdfVectorizer.get_feature_names_out(), tfIdf[idx].T.toarray().flatten(order='C'))}
        idx += 1
        # print(type(sent))
        # print(tfIdfVectorizer.get_feature_names_out())
        # print(tfidf_wght)
        # if 'addressable' in sent.text:
        #     print(sent)
        #     print(tfidf_wght)

        sent_wght = 0
        sent_num_words = 0
        for word in sent:
            # print(word, word.lemma_, tfidf_wght.get(word.lemma_,0))
            if not (word.is_punct or word.is_space or word.is_stop or word.is_sent_end):
                sent_wght += tfidf_wght.get(word.lemma_, 0)
                sent_num_words += 1

        # print(sent, sent_wght)
        sent_wght_dict[sent] = sent_wght

    summary_by_importance = sorted(sent_wght_dict.items(), key=lambda x: x[1], reverse=True)
    # print(summary_by_importance[:5])
    for i in range(5):
        print(f'{i + 1}:{summary_by_importance[i]}')
