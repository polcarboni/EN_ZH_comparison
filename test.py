from extract_dicts import import_split_stories_dict
from typing import Dict, List
from transformers import BertTokenizer, BertModel
from feature_utils import token_to_word_attention_maps

path_EN = "data/text_EN"
path_ZH = "data/text_ZH"

split_text_EN = import_split_stories_dict(path_EN)
split_text_ZH = import_split_stories_dict(path_ZH)

print("\n", split_text_EN.keys(), "\n")

print(len(split_text_EN['life']))
print(len(split_text_ZH['life']))

print(split_text_EN['life'])
print(split_text_ZH['life'])


def split_context(stories: Dict[str, List[str]], context_amount: int) -> Dict[str, List[str]]:
    context_dict = {}

    for key, string_list in stories.items():
        sublists = [
            " ".join(string_list[i:i+context_amount])
            for i in range(0, len(string_list) - context_amount + 1)
        ]

        context_dict[key] = sublists

    return context_dict


context_list = [1, 5, 10, 15, 20, 25, 30, 35, 40]


#Split context ZH
for i in range(len(context_list)):
    var_name = f"ZH_context_{context_list[i]:02}"
    globals()[var_name] = split_context(split_text_ZH, context_amount=context_list[i])

#Split context EN
for i in range(len(context_list)):
    var_name = f"EN_context_{context_list[i]:02}"
    globals()[var_name] = split_context(split_text_EN, context_amount=context_list[i])



def build_bling_dict(en_dict, zh_dict):
    bling_dict = {}
    for key in en_dict:
        if key in zh_dict:
            bling_dict[key] = {"EN": en_dict[key], "ZH": zh_dict[key]}
    return bling_dict


context_01_bling_stimuli = build_bling_dict(EN_context_01, ZH_context_01)
context_05_bling_stimuli = build_bling_dict(EN_context_05, ZH_context_05)
context_10_bling_stimuli = build_bling_dict(EN_context_10, ZH_context_10)
context_15_bling_stimuli = build_bling_dict(EN_context_15, ZH_context_15)
context_20_bling_stimuli = build_bling_dict(EN_context_20, ZH_context_20)
context_25_bling_stimuli = build_bling_dict(EN_context_25, ZH_context_25)
context_30_bling_stimuli = build_bling_dict(EN_context_30, ZH_context_30)
context_35_bling_stimuli = build_bling_dict(EN_context_35, ZH_context_35)
context_40_bling_stimuli = build_bling_dict(EN_context_40, ZH_context_40)


#_______________TEST_TRASNFORMERS__________________

souls_40_EN = context_40_bling_stimuli['alternateithicatom']['EN']
souls_40_ZH = context_40_bling_stimuli['alternateithicatom']['ZH']

print("EN len: ", len(souls_40_EN))
print("ZH len: ", len(souls_40_ZH))

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased', output_attentions=True)



print("Tokenizing...")
inputs_EN = tokenizer(souls_40_EN, return_tensors='pt', padding=True, truncation=True)
#inputs_ZH = tokenizer(souls_20_ZH, return_tensors='pt', padding=True, truncation=True)


print(inputs_EN)

print("Computing outputs...")
outputs_EN = model(**inputs_EN)
#outputs_ZH = model(**inputs_ZH)

attention_EN = outputs_EN.attentions
#attention_ZH = outputs_ZH.attentions





print("\n\n", type(attention_EN))
print(attention_EN[0].shape)
#print(attention_ZH[0].shape)
print(attention_EN[0][0].shape)
#print(attention_ZH[0][0].shape)

att_0 = attention_EN[0][0][0]
print("A before token_to_word: ", att_0.shape)

#token_to_word_attention_maps(att_0, )

