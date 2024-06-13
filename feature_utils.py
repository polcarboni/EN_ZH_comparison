import numpy as np
from typing import List

def token_to_word_attention_maps(token_attention: np.ndarray,
                                    input_sequence: List[str],
                                    tokenizer,
                                    gpt2 = False):
    '''Converts an array of token-to-token attention map to an array of word-to-word attention map.
    (as in section 4.1 of Clark et al. 2019)
    
    Sum of attention weights to a word and mean of attention weights from a word.
    '''
    word_attention = np.array(token_attention)

    words_to_tokens = []
    not_word_starts = []
    embedding_index = 0
    notfirstword = False  # Because GPT-2 tokenizes differently if there is a space or not before the word.
    for word in input_sequence:
        if gpt2:
            num_word_tokens = len(tokenizer.encode(word, add_prefix_space=notfirstword))
        else:
            num_word_tokens = len(tokenizer.tokenize(word))
        words_to_tokens.append([i for i in range(embedding_index, embedding_index + num_word_tokens)])
        embedding_index += num_word_tokens
        notfirstword = True

    for word in words_to_tokens:
        not_word_starts += word[1:]

    # Sum the attentions to all tokens for a word that has been split
    for word in words_to_tokens:
        word_attention[:, word[0]] = word_attention[:, word].sum(axis=-1)
    word_attention = np.delete(word_attention, not_word_starts, axis=-1)

    # Do the average of attentions from tokens for word a word that has been split 
    for word in words_to_tokens:
        word_attention[word[0], :] = np.mean(word_attention[word, :], axis=0)
    word_attention = np.delete(word_attention, not_word_starts, axis=0)
    
    return word_attention



def token_to_word():
    #for each word in the input sequence
        #number_of_word_tokens

    pass