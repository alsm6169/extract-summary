# extract-summary
GPT Alternatives to Text Summarisation

## m02_spacy_bow.py
The code reads a PDF document, extracts the text, and then analyzes the frequency of its words to identify the most important sentences. It uses Spacy, a natural language processing library, to tokenize the text, lemmatize the words, and identify stop words and punctuation. It then calculates the weight of each sentence based on the frequency of its words, and sorts the sentences by weight to identify the top 5 most important sentences.

## m03_scikit_cosine.py
The code uses a TfidfVectorizer from the Scikit-learn library to transform the sentences into vectors, which are then used to calculate the cosine similarity between all pairs of sentences. The sum of the similarity scores for each sentence is then computed and normalized to obtain a probability distribution over the sentences.

The top N sentences with the highest probability scores are then selected and printed to the console, where N is a parameter that can be set in the code.

## m03_scikit_tfidf.py
The code extracts text from the first page of a PDF file, processes the text using a spaCy language model to extract lemmas, and then vectorizes the dataset of lemmas using TF-IDF. It then calculates the weight of each sentence based on the TF-IDF values of the words in the sentence, and finally sorts the sentences by their weight in descending order and prints the top 5 sentences.

## m04_pytextrank.py
This code reads the text of a PDF document, processes it using the "textrank" algorithm, and prints a summary consisting of the top 5 sentences from the document.

## m05_transformer_generate.py
This code uses a pre-trained BART model to generate a summary of a PDF document. It first extracts the text from the PDF document, then tokenizes the text using the BART tokenizer and generates a summary using the BART model. Finally, it decodes the output tokens to generate the summary string and prints the summary along with the number of input words and output words. Overall, this code demonstrates how to use a pre-trained model to generate a summary using the transformers library.

## m05_transformer_pipeline.py
The main function of the script reads a PDF document using get_pdf_text() function from the pdf_reader module, and then summarizes the text using the BART model of the transformers module through the pipeline() function with specific settings. Finally, it prints the summarized text to the console.

## pdf_reader.py
reads the input provided pdf and retuns the text

## tfidf_example.py
Just a sample code to understand the usage of TfidfVectorizer
