from pdf_reader import get_pdf_text
import spacy
from collections import Counter

# Load the large English model for Spacy
nlp = spacy.load("en_core_web_lg")

if __name__ == '__main__':
    # Path and name of the PDF document to read
    doc_path_name = 'documents/chat_gpt_ubs.pdf'
    # Extract text from the first page of the document using the get_pdf_text function
    doc_text = get_pdf_text(doc_path_name, 1, 1)

    # Create a Spacy document object from the extracted text
    doc = nlp(doc_text.replace('\n', ' '))

    # Create a list of lemmatized words (excluding stop words and punctuation)
    arr = []
    total_words = 0
    for token in doc:
        if not (token.is_stop or token.is_punct or token.is_space):
            arr.append(token.lemma_)
            total_words += 1

    # Count the frequency of each lemmatized word
    freq_words = Counter(arr)

    # Print the frequency of the top 10 most common words
    print('freq_words: ', freq_words.most_common(10))
    # Print the total number of words
    print('total_words: ', total_words)

    # Calculate the weight of each sentence based on the frequency of its words
    sent_wght_dict = {}
    for sent in doc.sents:
        wght = 0
        for word in sent:
            wght += freq_words[word.lemma_] / total_words
        sent_wght_dict[sent] = wght

    # Sort the sentences by weight in descending order
    srted = sorted(sent_wght_dict.items(), key=lambda x: x[1], reverse=True)

    # Print the top 5 most important sentences
    for i in range(5):
        print(f'{i + 1}:{srted[i][0]}')
