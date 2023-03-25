from pdf_reader import get_pdf_text
import spacy
import pytextrank

nlp = spacy.load("en_core_web_lg")

if __name__ == '__main__':
    doc_path_name = 'documents/chat_gpt_ubs.pdf'
    doc_text = get_pdf_text(doc_path_name, 1, 1)
    # print(len(doc_text))

    nlp.add_pipe("textrank")
    doc = nlp(doc_text.replace('\n', ' '))

    for sent in doc._.textrank.summary(limit_phrases=5, limit_sentences=5):
        print(sent)
# # examine the top-ranked phrases in the document
# for phrase in doc._.phrases:
#     print(phrase.text)
#     print(phrase.rank, phrase.count)
#     print(phrase.chunks)