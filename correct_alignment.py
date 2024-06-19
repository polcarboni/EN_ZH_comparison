en_path = "/home/pol/EN_ZH_comparison/data/text_EN/alternateithicatom_EN.txt"
zh_path = "/home/pol/EN_ZH_comparison/data/text_ZH/alternateithicatom_ZH.txt"

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

en_sentences = extract_sentence_list(en_path)
zh_sentences = extract_sentence_list(zh_path)

print(len(en_sentences))
print(len(zh_sentences))

from deep_translator import GoogleTranslator
translator = GoogleTranslator("auto", "en")

for i in range(48, len(zh_sentences)):
    print(en_sentences[i], "\n")
    print(zh_sentences[i])
    print(translator.translate(zh_sentences[i]), "\n", "    ---", "\n")
