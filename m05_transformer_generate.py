from pdf_reader import get_pdf_text
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

doc_path_name = 'documents/chat_gpt_ubs.pdf'
doc_text = get_pdf_text(doc_path_name, 1, 1)
# print(f'words doc_text = {doc_text.split(" ")}')
print(f'num input words = {len(doc_text.split(" "))}')

checkpoint = 'facebook/bart-large-cnn'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# only for understanding how tokenizer works,
# here we are not setting return_tensors="pt" so it returns list of python integers
tokens = tokenizer(doc_text)
# print('tokens = ', type(tokens), tokens)
print(f'num input tokens = {len(tokens["input_ids"])}')

# again only for understanding purpose the above step can be further split of above steps to see the intermediate values
# tokenized_text = tokenizer.tokenize(doc_text)
# print('tokenized text = ', type(tokenized_text), tokenized_text)
# print(f'num tokens = {len(tokenized_text)}')
# token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
# print('convert_tokens_to_ids', token_ids)
# decoded_string = tokenizer.decode(token_ids)
# print('decoded_string = ', decoded_string)



# Load the pre-trained BART model
# model = BartForConditionalGeneration.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

tokenized_text = tokenizer(doc_text, return_tensors="pt")

# note below: max tokens and not words
outputs = model.generate(tokenized_text['input_ids'], min_new_tokens=100, max_new_tokens=200,
                         length_penalty=2.0, num_beams=4, do_sample=True)
print(f'num output token {outputs.shape}') # , outputs)
output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f'num output words = {len(output_str.split(" "))}')
print(f'summary = {output_str}')
