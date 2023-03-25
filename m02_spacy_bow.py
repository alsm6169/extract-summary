from pdf_reader import get_pdf_text
'''$ python -m spacy download en-core-web-lg'''
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_lg")

if __name__ == '__main__':
    doc_path_name = 'documents/chat_gpt_ubs.pdf'
    doc_text = get_pdf_text(doc_path_name, 1, 1)
    # print(len(doc_text))

    doc = nlp(doc_text.replace('\n', ' '))
    arr = []
    total_words = 0
    for token in doc:
        # print(token.text, token.lemma_, token.is_alpha, token.is_stop, token.is_punct)
        if not(token.is_stop or token.is_punct or token.is_space):
            # print('token.lemma_: ', token.lemma_)
            arr.append(token.lemma_)
            total_words += 1

    freq_words = Counter(arr)
    # print(freq_words.most_common(10))
    print('freq_words: ', freq_words)
    print('total_words: ', total_words)

    sent_wght_dict = {}
    for sent in doc.sents:
        # print(f'[{sent}]')
        wght = 0
        for word in sent:
            # print(word, word.lemma_, freq_words[word.lemma_])
            wght += freq_words[word.lemma_]/total_words

        # print(sent, wght)
        sent_wght_dict[sent] = wght

    srted = sorted(sent_wght_dict.items(), key=lambda x: x[1], reverse=True)
    # print(type(srted))
    # print(summary_by_importance[:5])
    for i in range(5):
        print(f'{i+1}:{srted[i][0]}')