#extract just a part of text together, both EN and ZH up to the same period. 
#sliding window approach until one of the 2 sentences starts with a dot, then restart.
#STOP: when one of them reaches last char (?)

#Check wether in the process the quality of the alignment of splitted sentences

#When the start of one of them reaches the period, meaning the other has more characters (slower), the second one should jump to that point.

file_path_EN = "/home/pol/EN_ZH_comparison/data/text_EN/alternateithicatom_EN.txt"
file_path_ZH = "/home/pol/EN_ZH_comparison/data/text_ZH/alternateithicatom_ZH.txt"

#FANCIER: the shorter one moves until it reaches the longer one end. Based on sentence length. 
#How to correlate different comparison of the same sentence? Should test


#The same features is going to be used in the regression model.

#Start by testing the second one.

def extract_sentence_list(file_path):
    if "_EN.txt" in file_path:
        split_char = "."
    elif "_ZH.txt" in file_path:
        split_char = "。"
    
    with open(file_path, 'r') as file:
        content = file.read().strip()
        #print(content)
        words = content.split(split_char)
        words.pop()
    return words

sentences_en = extract_sentence_list(file_path_EN)
sentences_zh = extract_sentence_list(file_path_ZH)

def split_sentences(sentences_list):
    split_list = []
    for sentence in sentences_list:
        sentence_list = sentence.split()
        split_list.append(sentence_list)
    return split_list


#Convert sentences to list of words. (Matching with previoud code)

split_en = split_sentences(sentences_en)
split_zh = split_sentences(sentences_zh)

for i in range(3):
    print(split_en[i])
    print(split_zh[i])

print("\n\n")
from transformers import BertTokenizer, BertModel

#config = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True, output_attention=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased", output_attentions=True)

tokens_list_en = []
tokens_list_zh = []

for i in range(3):

    print("_____________________________________\n")
    tokens_en = tokenizer(sentences_en[i], return_tensors='pt', padding=True)
    tokens_zh = tokenizer(sentences_zh[i], return_tensors='pt', padding=True)
    
    print(sentences_en[i])
    print(tokens_en['input_ids'])
    print(tokenizer.convert_ids_to_tokens(tokens_en['input_ids'][0]))
    print("\n")

    print(sentences_zh[i])
    print(tokens_zh['input_ids'])
    print(tokenizer.convert_ids_to_tokens(tokens_zh['input_ids'][0]))
    
    #tokens_list_en.append(tokens_en)
    #tokens_list_zh.append(tokens_zh)

    length_en = tokens_en['input_ids'].shape[1]
    length_zh = tokens_zh['input_ids'].shape[1]
    
    print(type(tokens_en['input_ids']))
    attention_size = min(length_en, length_zh)
    
    if length_en == length_zh:
        shorter_lang = "same"
    elif length_en > length_zh:
        shorter_lang = "zh"
    elif length_en < length_zh:
        shorter_lang = "en"

    print("SHORTER LANG: ", shorter_lang)

    if shorter_lang == "en":
        sliding_windows = [tokens_zh["input_ids"][0][i:i + attention_size] for i in range(len(tokens_zh["input_ids"][0]) - attention_size + 1)]
        for i in range(len(sliding_windows)):
            print(sliding_windows[i])

    if shorter_lang == "zh":
        print(type(tokens_en))
        sliding_windows = [tokens_en["input_ids"][0][i:i + attention_size] for i in range(len(tokens_en["input_ids"][0]) - attention_size + 1)]
        for i in range(len(sliding_windows)):
            print(sliding_windows[i])

            ouputs_en = model(**tokens_en)
            attention = ouputs_en.attentions
            print("\n","EN: ", attention[0].shape)

            outputs_en = model(sliding_windows[i])

    '''
    ouputs_en = model(**tokens_en)
    attention = ouputs_en.attentions
    print("\n","EN: ", attention[0].shape)

    ouputs_zh = model(**tokens_zh)
    attention = ouputs_zh.attentions
    print("ZH: ", attention[0].shape)

    print("\n\n")
    '''

"""
print(tokenizer.convert_ids_to_tokens(tokens_en["input_ids"][0]))
print(tokenizer.convert_ids_to_tokens(tokens_zh["input_ids"][0]))

"""



'''
outputs_en = model(**tokens_en)
outputs_zh = model(**tokens_zh)

attention_en = outputs_en.attentions
attention_zh = outputs_zh.attentions


#Each element in tuple is a layer (use it for average value)
print(attention_en[0].shape)
print(attention_zh[0].shape)

from feature_utils import token_to_word_attention_maps

word_map_en = token_to_word_attention_maps(attention_en[0].detach().numpy(), sentences_EN[0], tokenizer=tokenizer)
word_map_zh = token_to_word_attention_maps(attention_zh[0].detach().numpy(), sentences_ZH[0], tokenizer=tokenizer)'''