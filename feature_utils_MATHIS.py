import cottoncandy as cc
import epitran
import fasttext
import io
import jieba
import logging
import numpy as np
import os
import torch
import wget
import networkx as nx

import random

from gensim.models import KeyedVectors
from typing import Dict, List
from nltk.stem.snowball import SnowballStemmer

from bling.config import DATA_DIRECTORY
from bling.data_loading import hard_coded_things
from bling.data_loading.DataSequence import DataSequence
from bling.utils import get_s3_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger('features')
logger.setLevel('INFO')

try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer, GPT2Model, GPT2Tokenizer, GPT2Config
except Exception:
    logger.warn('Could not import BertModel, BertTokenizer from transformers')

try:
    import benepar
except Exception:
    logger.warn('Could not import benepar.')


# Sublexical features.

def histogram_phonemes2(ds, phonemeset=hard_coded_things.phonemes):
    '''Histograms the phonemes in the DataSequence [ds].'''
    olddata = np.array([ph.upper().strip('0123456789') for ph in ds.data])
    newdata = np.vstack([olddata == ph for ph in phonemeset]).T
    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)


def word_length(ds):
    '''Number of letters in the DataSequence [ds].'''
    newdata = np.vstack([len(k) for k in ds.data])
    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)


def histogram_letters(ds, letter_set=hard_coded_things.letters):
    '''Count of each letter in the DataSequence [ds].'''
    olddata = ds.data
    N = len(ds.data)
    newdata = np.zeros((N, len(letter_set)))
    letters_dict = {k: v for k, v in zip(letter_set, list(range(0, len(letter_set))))}

    for wordnr, word in enumerate(olddata):
        for letter in word.upper():
            if letter in letter_set:
                newdata[wordnr, letters_dict[letter]] += 1
    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)


def get_ipa_symbols(ds: DataSequence,
                    language: str,
                    return_type: str,
                    cedict_file: str = '/home/jlg/cathy/projects/bling/data/pronounciation/cedict_ts.u8'):
    '''Return IPA symbols for words in the DataSequence [ds].

    args:
        ds: DataSequence containing words to extract IPA symbols for.
        language: identifier for the language (en, es, or zh).
        return_type: the type of information to extract. Options:
            counts_all_symbols: extracts the counts of each IPA symbol.
            counts_alphabetic_symbols: extracts the counts of each alphabetic IPA symbol.
            num_all_symbols: extracts the number of IPA symbols per word.
            num_alphabetic_symbols: extracts the number of alphabetic IPA symbols per word.
        cedict_file: Filepath for CC-CEDICT. If hard coded file does not exist, download into local
            DATA_DIRECTORY/feature_files

    Notes:
        Requires lex_lookup installation and CC-CEDict download (see
        https://github.com/dmort27/epitran for details).

    '''

    if not os.path.exists(cedict_file):
        filepath_local = os.path.join(DATA_DIRECTORY, 'feature_files')
        if not os.path.exists(filepath_local):
            os.makedirs(filepath_local)
        print('CC-CEDict File is not found. Downloading into {}'.format(filepath_local))
        cedict_url = 'https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz'
        wget.download(cedict_url, filepath_local)
        os.system('gunzip {}'.format(os.path.join(filepath_local, cedict_url.split('/')[-1])))
        new_cedict_file = os.path.join(filepath_local, 'cedict_ts.u8')
        os.rename(os.path.join(filepath_local, cedict_url.split('/')[-1][:-3]), new_cedict_file)
        cedict_file = new_cedict_file

    # From: https://iso639-3.sil.org/code_tables/639/data
    iso_codes_dict = {'en': 'eng-Latn',
                      'es': 'spa-Latn',
                      'zh': 'cmn-Hans'}

    assert return_type in ['counts_all_symbols',
            'counts_alphabetic_symbols',
            'num_all_symbols',
            'num_alphabetic_symbols'], +\
            'Invalid return_type {return_type} for get_ipa_symbols()'

    if 'alphabetic' in return_type:
        ipa_symbols = hard_coded_things.ipa_alphabetic_symbols
    else:
        ipa_symbols = hard_coded_things.ipa_symbols

    new_data = []
    text = np.array(ds.data)
    epi = epitran.Epitran(iso_codes_dict[language], cedict_file=cedict_file)
    for word in text:
        word_ipa_vector = np.zeros((1, len(ipa_symbols)))
        word_ipa_symbols = epi.transliterate(clean_punctuation(word))
        for ipa_symbol in word_ipa_symbols:
            word_ipa_vector[0, ipa_symbols.index(ipa_symbol)] += 1
        new_data.append(word_ipa_vector)

    if 'num' in return_type:
        new_data = np.vstack([np.sum(ipa_vector) for ipa_vector in new_data])
    else:
        new_data = np.squeeze(new_data)

    stimulus_ds = DataSequence(new_data, ds.split_inds, ds.data_times, ds.tr_times)

    return stimulus_ds


# Lexical and super-lexical features.
def get_word_embedding(word: str,
                       embedding: Dict,
                       default_embedding: np.ndarray):
    '''Return the lexical embedding of a given word.

    Parameters:
    ----------
    word : str
        The word for which to get an embedding.
    embedding : dict
        A word:embedding dictionary of word embeddings.
    default_embedding : array_like
        The embedding to use if the word is not in the embedding dictionary.
    '''
    try:
        return embedding[word]
    except Exception:
        logger.warn(f'{word} missing from embedding dict.')
        return default_embedding


def get_lexical_embeddings(ds: DataSequence,
                           embedding: int,
                           default_embedding: np.ndarray):
                        #    stemming: bool = False):
    new_data = []
    text = np.array(ds.data)
    # stemming = False
    # if stemming:
        # stemmer = SnowballStemmer("english")
    # covered_sum = 0
    for word in text:
        # if stemming:
            # word = stemmer.stem(word)
        # word_embedded, covered = get_word_embedding(word, embedding, default_embedding)
        # covered_sum += covered
        new_data.append(get_word_embedding(word, embedding, default_embedding))
    random.shuffle(new_data)

    # print(f'Stemming is {stemming}.\nThe word coverage is {covered_sum/len(text)*100:.2f}%')
    embedding_ds = DataSequence(np.array(new_data), ds.split_inds, ds.data_times, ds.tr_times)
    # embedding_ds = DataSequence(np.squeeze(np.array(new_data)), ds.split_inds, ds.data_times, ds.tr_times)
    return embedding_ds


def get_contextual_embeddings(ds: DataSequence,
                              model_name: str,
                              context_length: int,
                              layer_num: int,
                              add_special_tokens: bool = True,
                              avg_tokens:bool = True,
                              pretrained: bool = True,
                            #   use_attention: str = False,
                              shuffle: bool = False,
                              sequencing: str = 'sliding_window',
                              input_embeddings: bool = False,
                              bad_words: List[str] = hard_coded_things.bad_words_with_sentence_boundaries):
    '''Returns the embeddings from multilingual BERT corresponding to the values in ds.

    args:
        ds: A DataSequence containing stimuli for which to retrieve embeddings.
        context_length: The number of words preceeding the embedded word to feed as context.
        layer_num: The layer from which to extract embeddings.
        bad_words: A list of words to ignore.
    '''

    torch.manual_seed(0)

    def token_to_word_embeddings(layer_embedding: np.ndarray,
                                 input_sequence: List[str],
                                 tokenizer):
        '''Converts an array of token embeddings to an array of word embeddings.

        Uses the last token of a word as its embedding.
        '''
        word_embeddings = []
        embedding_index = 0
        for word in input_sequence:
            num_word_tokens = len(tokenizer.tokenize(word))
            embedding_index += num_word_tokens
            word_embeddings.append(np.expand_dims(layer_embedding[embedding_index - 1], axis=0))
        return word_embeddings
    
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

    def pad_to_constant_size(matrix, expected_size):
        assert(matrix.shape[0] == matrix.shape[1])
        if matrix.shape[0] < expected_size:
            return np.pad(matrix, (0, expected_size - matrix.shape[0]))
        elif matrix.shape[0] == expected_size:
            return matrix
        else:
            logger.warning(f'The attention matrix has unexpected size {matrix.shape[0]}')
            return matrix[:expected_size, :expected_size]

    use_attention = False
    if '_Att_' in model_name:
        idx = model_name.find('_Att_')
        use_attention = model_name[idx+5:]
        model_name = model_name[:idx]

    if 'bert-base-multilingual-cased-finetuned-' in model_name:
        config = AutoConfig.from_pretrained('bert-base-multilingual-cased')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased') 
    elif 'gpt2' in model_name:
        config = GPT2Config.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    config.output_hidden_states = True
    if use_attention:
        config.output_attentions = True
    else:
        config.output_attentions = False
    
    if 'gpt2' in model_name:
        model = GPT2Model.from_pretrained(model_name, config=config)
    else:
        if pretrained:
            model = AutoModel.from_pretrained(model_name, config=config)
        else:
            model = AutoModel.from_config(config)
            
    new_data = []
    text = np.array(ds.data)
    if 'sentence' in sequencing:
        sentence_starts = [i for i, word in enumerate(text) if word == 'sentence_start']
        sentence_ends = [i for i, word in enumerate(text) if word == 'sentence_end']
        sentence_index = 0
    input_sequences = []
    for word_index, word in enumerate(text):

        if word == 'sentence_start' or word == 'sentence_end':
            continue
        
        if sequencing == 'sliding_window':
            input_sequences.append(text[max(0, word_index - context_length): word_index + 1])
        
        elif sequencing == 'whole_sentence':
            if word_index > sentence_ends[sentence_index]:
                sentence_index += 1
            input_sequences.append(text[sentence_starts[sentence_index] + 1: sentence_ends[sentence_index]])
            
        elif sequencing == 'left_sentence':
            if word_index > sentence_ends[sentence_index]:
                sentence_index += 1
            input_sequences.append(text[sentence_starts[sentence_index] + 1: word_index + 1])
            
        else:
            logger.error(f'Sequencing: {sequencing} is not supported!')
            assert(False)
            
    for input_sequence in input_sequences:
        input_sequence_bad_words_indices = np.where(np.isin(input_sequence, bad_words))[0]
        input_sequence_cleaned = np.delete(input_sequence, input_sequence_bad_words_indices)
        if shuffle:
            random.shuffle(input_sequence_cleaned)
        input_sequence_cleaned = ' '.join(input_sequence_cleaned)
        if 'bert' in model_name:
            encoded_input_sequence = tokenizer.encode_plus(text=input_sequence_cleaned,
                                                            add_special_tokens=add_special_tokens,
                                                            return_special_tokens_mask=add_special_tokens,
                                                            return_tensors='pt')
        elif 'gpt2' in model_name:
            encoded_input_sequence = tokenizer.encode_plus(text=input_sequence_cleaned,
                                                            return_tensors='pt')
        tokens_tensor = encoded_input_sequence['input_ids']
        
        with torch.no_grad():
            outputs = model(tokens_tensor)

            # Compute raw attention, rollout and flow
            if use_attention:
                all_attentions = outputs[-1]
                _attentions = [att.detach().numpy() for att in all_attentions]

                # raw
                attentions_mat = np.asarray(_attentions)[:,0]

                if use_attention == 'avg_words':

                    # print(attentions_mat.shape)
                    # input_sequence_with_spetok = tokenizer.decode(tokens_tensor[0]).split()
                    # w2w_FULL = np.zeros(shape=(attentions_mat.shape[0],attentions_mat.shape[0],context_length + 3,context_length + 3))
                    # for i in range(attentions_mat.shape[0]):
                    #     for j in range(attentions_mat.shape[1]):
                    #         word_attention_map = token_to_word_attention_maps(attentions_mat[i,j], input_sequence_with_spetok, tokenizer)
                    #         word_attention_map_padded = pad_to_constant_size(word_attention_map, context_length + 3)
                    #         w2w_FULL[i,j,:,:] = word_attention_map_padded
                    
                    new_data.append(np.mean(attentions_mat[:,:,:,1:-1], axis=(-1,-2)).flatten())

                elif use_attention.startswith('OLDlayer'):
                    layer = int(use_attention[5:])
                    attention_layer = attentions_mat[layer,:,:,:]
                    new_data.append(np.mean(attention_layer[:,:,-1], axis = 1))
                
                elif use_attention.startswith('head'):
                    head = int(use_attention[4:])
                    attention_head = attentions_mat[:,head,:,:]
                    new_data.append(np.mean(attention_head[:,:,-1], axis = 1))

                elif use_attention.startswith('layerSEP'):
                    layer = int(use_attention[8:])
                    attention_layer = attentions_mat[layer,:,:,:]
                    input_sequence_with_spetok = tokenizer.decode(tokens_tensor[0]).split()
                    avg_att_SEP = []
                    for head in range(attention_layer.shape[0]):
                        word_attention_map = token_to_word_attention_maps(attention_layer[head,:,:], input_sequence_with_spetok, tokenizer)
                        avg_att_SEP.append(np.mean(word_attention_map[:,-1]))
                    new_data.append(np.asarray(avg_att_SEP))

                elif use_attention.startswith('layerLastWord'):
                    layer = int(use_attention[13:])
                    attention_layer = attentions_mat[layer,:,:,:]
                    input_sequence_with_spetok = tokenizer.decode(tokens_tensor[0]).split()
                    avg_att_LastWord = []
                    for head in range(attention_layer.shape[0]):
                        word_attention_map = token_to_word_attention_maps(attention_layer[head,:,:], input_sequence_with_spetok, tokenizer)
                        avg_att_LastWord.append(np.mean(word_attention_map[:,-2]))
                    new_data.append(np.asarray(avg_att_LastWord))

                elif use_attention.startswith('linghead'):
                    ling = use_attention[8:]
                    HEADS_LINGUISTIC = {
                        "Dir" : (7, 9),
                        "Det" : (7, 10),
                        "Pos" : (6, 5),
                        "Pas" : (3, 9),
                        "Obj" : (8, 5),
                        "Cor" : (4, 3),
                        "Bro" : (0, 0),
                        "Nxt" : (2, 0),
                        "Sep" : (7, 6)
                    }
                    head = HEADS_LINGUISTIC[ling]
                    input_sequence_with_spetok = tokenizer.decode(tokens_tensor[0]).split()
                    word_attention_map = token_to_word_attention_maps(attentions_mat[head], input_sequence_with_spetok, tokenizer)
                    word_attention_map_padded = pad_to_constant_size(word_attention_map, context_length + 3)
                    new_data.append(word_attention_map_padded.flatten())

                elif use_attention.startswith('wordhead'):
                    ling = use_attention[8:]
                    HEADS_LINGUISTIC = {
                        "Dir" : (7, 9),
                        "Det" : (7, 10),
                        "Pos" : (6, 5),
                        "Pas" : (3, 9),
                        "Obj" : (8, 5),
                        "Cor" : (4, 3),
                        "Bro" : (0, 0),
                        "Nxt" : (2, 0),
                        "Sep" : (7, 6)
                    }
                    head = HEADS_LINGUISTIC[ling]
                    input_sequence_with_spetok = tokenizer.decode(tokens_tensor[0]).split()
                    word_attention_map = token_to_word_attention_maps(attentions_mat[head], input_sequence_with_spetok, tokenizer)
                    word_attention_map_padded = pad_to_constant_size(word_attention_map, context_length + 3)
                    new_data.append(word_attention_map_padded[1:-1,1:-1].flatten())

                elif use_attention.startswith('lastwordhead'):
                    ling = use_attention[12:]
                    HEADS_LINGUISTIC = {
                        "Dir" : (7, 9),
                        "Det" : (7, 10),
                        "Pos" : (6, 5),
                        "Pas" : (3, 9),
                        "Obj" : (8, 5),
                        "Cor" : (4, 3),
                        "Bro" : (0, 0),
                        "Nxt" : (2, 0),
                        "Sep" : (7, 6)
                    }
                    head = HEADS_LINGUISTIC[ling]
                    input_sequence_with_spetok = tokenizer.decode(tokens_tensor[0]).split()
                    word_attention_map = token_to_word_attention_maps(attentions_mat[head], input_sequence_with_spetok, tokenizer)
                    word_attention_map_padded = pad_to_constant_size(word_attention_map, context_length + 3)
                    new_data.append(word_attention_map_padded[1:-1,-2].flatten())

                elif use_attention.startswith('sixling'):
                    HEADS_LINGUISTIC = {
                        "Dir" : (7, 9),
                        "Det" : (7, 10),
                        "Pos" : (6, 5),
                        "Pas" : (3, 9),
                        "Obj" : (8, 5),
                        "Cor" : (4, 3),
                        # "Bro" : (0, 0),
                        # "Nxt" : (2, 0),
                        # "Sep" : (7, 6)
                    }
                    # head = HEADS_LINGUISTIC[ling]
                    input_sequence_with_spetok = tokenizer.decode(tokens_tensor[0]).split()
                    attention_maps_list = []
                    for head in HEADS_LINGUISTIC.values():
                        word_attention_map = token_to_word_attention_maps(attentions_mat[head], input_sequence_with_spetok, tokenizer)
                        attention_maps_list.append(pad_to_constant_size(word_attention_map, context_length + 3))
                    new_data.append(np.asarray(attention_maps_list).flatten())

                elif use_attention.startswith('sixonlywords'):
                    HEADS_LINGUISTIC = {
                        "Dir" : (7, 9),
                        "Det" : (7, 10),
                        "Pos" : (6, 5),
                        "Pas" : (3, 9),
                        "Obj" : (8, 5),
                        "Cor" : (4, 3),
                        # "Bro" : (0, 0),
                        # "Nxt" : (2, 0),
                        # "Sep" : (7, 6)
                    }
                    # head = HEADS_LINGUISTIC[ling]
                    input_sequence_with_spetok = tokenizer.decode(tokens_tensor[0]).split()
                    attention_maps_list = []
                    for head in HEADS_LINGUISTIC.values():
                        word_attention_map = token_to_word_attention_maps(attentions_mat[head], input_sequence_with_spetok, tokenizer)
                        attention_maps_list.append(pad_to_constant_size(word_attention_map, context_length + 3)[1:-1,1:-1])
                    new_data.append(np.asarray(attention_maps_list).flatten())

                elif use_attention.startswith('sixbroadheads'):
                    HEADS_BROADLY = {
                        "Bro1" : (0, 0),
                        "Bro2" : (0, 4),
                        "Bro3" : (0, 9),
                        "Bro4" : (1, 0),
                        "Bro5" : (1, 8),
                        "Bro6" : (3, 1)
                    }
                    # head = HEADS_LINGUISTIC[ling]
                    input_sequence_with_spetok = tokenizer.decode(tokens_tensor[0]).split()
                    attention_maps_list = []
                    for head in HEADS_BROADLY.values():
                        word_attention_map = token_to_word_attention_maps(attentions_mat[head], input_sequence_with_spetok, tokenizer)
                        attention_maps_list.append(pad_to_constant_size(word_attention_map, context_length + 3))
                    new_data.append(np.asarray(attention_maps_list).flatten())

                elif use_attention.startswith('sixlastword'):
                    HEADS_LINGUISTIC = {
                        "Dir" : (7, 9),
                        "Det" : (7, 10),
                        "Pos" : (6, 5),
                        "Pas" : (3, 9),
                        "Obj" : (8, 5),
                        "Cor" : (4, 3),
                        # "Bro" : (0, 0),
                        # "Nxt" : (2, 0),
                        # "Sep" : (7, 6)
                    }
                    # head = HEADS_LINGUISTIC[ling]
                    input_sequence_with_spetok = tokenizer.decode(tokens_tensor[0]).split()
                    attention_maps_list = []
                    for head in HEADS_LINGUISTIC.values():
                        word_attention_map = token_to_word_attention_maps(attentions_mat[head], input_sequence_with_spetok, tokenizer)
                        attention_maps_list.append(pad_to_constant_size(word_attention_map, context_length + 3)[1:-1,-2])
                    new_data.append(np.asarray(attention_maps_list).flatten())

                elif use_attention == 'sum_last_word':
                    input_sequence_with_spetok = tokenizer.decode(tokens_tensor[0]).split()
                    sumed_att = []
                    for i in range(attentions_mat.shape[0]):
                        for j in range(attentions_mat.shape[1]):
                            word_attention_map = token_to_word_attention_maps(attentions_mat[i,j], input_sequence_with_spetok, tokenizer)
                            sumed_att.append(np.sum(word_attention_map[:,-2]))
                    new_data.append(np.asarray(sumed_att))

                elif use_attention == 'save_w2wFULL':
                    input_sequence_with_spetok = tokenizer.decode(tokens_tensor[0]).split()
                    if 'bert' in model_name:
                        sentence_length = context_length + 3
                        gpt2 = False
                    elif 'gpt2' in model_name:
                        sentence_length = context_length + 1
                        gpt2 = True
                    else:
                        logger.error(f'{model_name} is not supported!')

                    w2w_FULL = np.zeros(shape=(attentions_mat.shape[0],attentions_mat.shape[0], sentence_length, sentence_length))
                    for i in range(attentions_mat.shape[0]):
                        for j in range(attentions_mat.shape[1]):
                            word_attention_map = token_to_word_attention_maps(attentions_mat[i,j], input_sequence_with_spetok, tokenizer, gpt2)
                            word_attention_map_padded = pad_to_constant_size(word_attention_map, sentence_length)
                            w2w_FULL[i,j,:,:] = word_attention_map_padded
                    new_data.append(w2w_FULL.flatten())

                elif use_attention == 'one_value':
                    new_data.append(np.asarray([np.mean(attentions_mat)]))
                    
                elif use_attention == 'all_heads':
                    final_att = attentions_mat
                    new_data.append(np.mean(final_att[:,:,:,-1], axis=2).flatten())

                elif use_attention == 'all_broad':
                    final_att = []
                    HEADS_BROAD = {(0, 0): 'Broad',
                                    (0, 1): 'Broad',
                                    (0, 4): 'Broad',
                                    (0, 5): 'Broad',
                                    (0, 6): 'Broad',
                                    (0, 7): 'Broad',
                                    (0, 9): 'Broad',
                                    (1, 0): 'Broad',
                                    (1, 8): 'Broad',
                                    (1, 9): 'Broad',
                                    (3, 1): 'Broad'}
                    for head in HEADS_BROAD.keys():
                        final_att.append(np.mean(attentions_mat[head[0],head[1],:,-1], axis=0))
                    new_data.append(final_att)

                elif use_attention == 'all_position':
                    final_att = []
                    HEADS_POSITION = {(0, 10): 'Next',
                                    (1, 4): 'Previous',
                                    (2, 0): 'Next',
                                    (2, 9): 'Next',
                                    (3, 5): 'Previous',
                                    (5, 9): 'Next',
                                    (6, 11): 'Previous',
                                    (7, 4): 'Previous'}
                    for head in HEADS_POSITION.keys():
                        final_att.append(np.mean(attentions_mat[head[0],head[1],:,-1], axis=0))
                    new_data.append(final_att)

                elif use_attention == 'all_sep':
                    final_att = []
                    HEADS_SEP = {(4, 3): '[SEP]',
                                (4, 7): '[SEP]',
                                (5, 1): '[SEP]',
                                (5, 5): '[SEP]',
                                (5, 6): '[SEP]',
                                (5, 7): '[SEP]',
                                (5, 11): '[SEP]',
                                (6, 0): '[SEP]',
                                (6, 1): '[SEP]',
                                (6, 3): '[SEP]',
                                (6, 10): '[SEP]',
                                (7, 3): '[SEP]',
                                (7, 6): '[SEP]',
                                (7, 7): '[SEP]',
                                (7, 11): '[SEP]',
                                (8, 0): '[SEP]',
                                (8, 2): '[SEP]',
                                (8, 4): '[SEP]',
                                (8, 5): '[SEP]',
                                (8, 6): '[SEP]',
                                (9, 4): '[SEP]',
                                (9, 9): '[SEP]'}
                    for head in HEADS_SEP.keys():
                        final_att.append(np.mean(attentions_mat[head[0],head[1],:,-1], axis=0))
                    new_data.append(final_att)

                elif use_attention == 'all_ling':
                    final_att = []
                    HEADS_LING = {(4, 3): 'Coreference',
                                (7, 10): 'Determiner',
                                (7, 9): 'Direct object',
                                (8, 5): 'Object of prep.',
                                (3, 9): 'Passive auxiliary',
                                (6, 5): 'Possesive'}
                    for head in HEADS_LING.keys():
                        final_att.append(np.mean(attentions_mat[head[0],head[1],:,-1], axis=0))
                    new_data.append(final_att)

                elif use_attention == 'all_firsttok':
                    final_att = attentions_mat
                    new_data.append(np.mean(final_att[:,:,:,0], axis=2).flatten())

                elif use_attention == 'all_secondtok':
                    final_att = attentions_mat
                    new_data.append(np.mean(final_att[:,:,:,1], axis=2).flatten())

                elif use_attention == 'all_lastword':
                    final_att = attentions_mat
                    new_data.append(np.mean(final_att[:,:,:,-2], axis=2).flatten())

                elif use_attention == 'all_wordstolast':
                    final_att = attentions_mat
                    new_data.append(np.mean(final_att[:,:,1:-1,-2], axis=2).flatten())

                elif use_attention == 'all_source':
                    final_att = attentions_mat
                    new_data.append(np.mean(final_att[:,:,-1,:], axis=2).flatten())

                elif use_attention == 'all_concat':
                    final_att = attentions_mat
                    new_data.append(np.concatenate((np.mean(final_att[:,:,-1,:], axis=2).flatten(), np.mean(final_att[:,:,:,-1], axis=2).flatten())))
                    # new_data.append(np.concatenate(np.mean(final_att[:,:,-1,:], axis=2).flatten(),np.mean(final_att[:,:,:,-1], axis=2).flatten()))

                else:

                    raw_att_mat = attentions_mat.sum(axis=1)/attentions_mat.shape[1]

                    # res
                    res_att_mat = raw_att_mat
                    res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None,...]
                    res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[...,None]

                    if use_attention == 'lateral':
                        final_att = attentions_mat.sum(axis=0)/attentions_mat.shape[0]

                    elif use_attention == 'raw':
                        final_att = raw_att_mat

                    elif use_attention == 'res':
                        final_att = res_att_mat
                                        
                    elif use_attention == 'wordstoroll':
                        # rollout
                        joint_attentions = compute_joint_attention(res_att_mat, add_residual=False)
                        final_att = joint_attentions

                    elif use_attention == 'flow':
                        res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=input_sequence_cleaned)
                        res_G = draw_attention_graph(res_adj_mat,res_labels_to_index, n_layers=res_att_mat.shape[0], length=res_att_mat.shape[-1])
                        
                        # flow
                        output_nodes = []
                        input_nodes = []
                        for key in res_labels_to_index:
                            if 'L24' in key:
                                output_nodes.append(key)
                            if res_labels_to_index[key] < attentions_mat.shape[-1]:
                                input_nodes.append(key)

                        flow_values = compute_flows(res_G, res_labels_to_index, input_nodes, length=attentions_mat.shape[-1])
                        flow_att_mat = convert_adjmat_tomats(flow_values, n_layers=attentions_mat.shape[0], l=attentions_mat.shape[-1])

                        final_att = flow_att_mat

                    else:
                        print(f'{use_attention} is not supported!')
                        assert(False)

                    # print(input_sequence_cleaned)
                    # print(attentions_mat.shape, res_att_mat.shape, 
                    #       joint_attentions.shape, flow_att_mat.shape)
                    # att2 = attentions_mat.sum(axis=1)/attentions_mat.shape[1]
                    # print(att2.shape)

                    # new_data.append(final_att[:,:,-1].flatten())
                    new_data.append(np.mean(final_att[:,1:-1,-2], axis=1))
                    # TODO trying attention to
                    # new_data.append(np.mean(final_att[:,-1,:], axis=1))
                    # TODO trying attention concat
                    # new_data.append(np.concatenate((np.mean(final_att[:,:,-1], axis=1),np.mean(final_att[:,-1,:], axis=1))))


            else:
                if input_embeddings:
                    embeddings = model.get_input_embeddings().weight[tokens_tensor[0]]  # [num_tokens x hidden_size]
                    mean_pool = torch.mean(embeddings, axis=0)
                    new_data.append(np.expand_dims(mean_pool, axis=0))
                else:
                    layer_embedding = outputs[-1][layer_num][0]
                    if avg_tokens:
                        if add_special_tokens:
                            # Discard the special tokens (logical not because they are marked as 1 and normal tokens as 0)
                            # special_mask = np.array(encoded_input_sequence['special_tokens_mask'])
                            # new_data.append(np.expand_dims(torch.mean(layer_embedding[np.logical_not(special_mask)], dim=0), axis=0))
                            
                            # try with all the tokens (CLS)
                            new_data.append(np.expand_dims(torch.mean(layer_embedding, dim=0), axis=0))
                        else:
                            new_data.append(np.expand_dims(torch.mean(layer_embedding, dim=0), axis=0))
                    else:
                        if add_special_tokens:
                            # Take the last token after discarding the special ones so that it corresponds to the last word
                            special_mask = np.array(encoded_input_sequence['special_tokens_mask'])
                            new_data.append(np.expand_dims(layer_embedding[np.logical_not(special_mask)][-1], axis=0))
                        else:
                            new_data.append(np.expand_dims(layer_embedding[-1], axis=0))

    bad_words_indices = np.where(np.isin(text, bad_words))[0]
    text_times = ds.data_times
    text_times_cleaned = np.delete(text_times, bad_words_indices)
    split_inds_array = np.array(ds.split_inds)
    for index in bad_words_indices[::-1]:
        split_inds_array[split_inds_array > index] = split_inds_array[split_inds_array > index] - 1
    embedding_ds = DataSequence(np.squeeze(np.array(new_data)), split_inds_array, text_times_cleaned, ds.tr_times)
    return embedding_ds


# Helper Functions.
def load_fasttext_aligned_vectors(fname, skip_first_line=True):
    '''Return dictionary of word embeddings from file saved in fasttext format.

    Parameters:
    -----------
    fname : str
        Name of file containing word embeddings.
    skip_first_line : bool
        If True, skip first line of file. Should do this if first line of file
        contains metadata.

    Returns:
    --------
    data : dict
        Dictionary of word embeddings.
    '''
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    if skip_first_line:
        n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(token) for token in tokens[1:]])
    return data


def get_embedding(embedding_name: str,
        word_vector_dir_s3: str = '/bling_features/word_vectors/',
        word_vector_dir_local: str = os.path.join(DATA_DIRECTORY, 's3_files', 'word_vectors'),
        cci_bucket_name: str = 'glab-bling-shared'):
    '''Returns dictionary of embeddings and default embedding for a given type of embedding.

    Parameters:
    -----------
        embedding_name: str
            Name of embedding to retrieve (e.g. fastText_en).
        word_vector_dir_s3: str
            Directory containing word vectors on s3
        word_vector_dir_local: str
            Directory containing word vectors on local folder
        cci_bucket_name: str
            Name of bucket containing embeddings.

    Notes:
    ------
        fastText embeddings are from https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
        word2vec_en is from https://code.google.com/archive/p/word2vec/
        word2vec_es is from https://crscardellino.github.io/SBWCE/
        word2vec_zh is from https://github.com/Kyubyong/wordvectors  ==> requires gensim version 3.0
            FIXME: would be better to use this maybe?
            https://github.com/Embedding/Chinese-Word-Vectors, SGNS with Word and Mixed corpora
            but downloaded file is a text file and not gensim format.

        word embeddings are also locally stored here (for now): /auto/k1/fatma/projects/Language/Multilingual/bling_save_20200212/data/language_models/
    '''
    if embedding_name == 'fastText_en':
        filename = 'cc.en.300.bin'
    elif embedding_name == 'fastText_zh':
        filename = 'cc.zh.300.bin'
    elif embedding_name == 'fastText_es':
        filename = 'cc.es.300.bin'
    elif embedding_name == 'fastText_zh_RCSLS':
        filename = 'wiki.zh.align.vec'
    elif embedding_name == 'fastText_en_RCSLS':
        filename = 'wiki.en.align.vec'
    elif embedding_name == 'word2vec_en':
        filename = 'GoogleNews-vectors-negative300.bin.gz'
    elif embedding_name == 'word2vec_es':
        filename = 'SBW-vectors-300-min5.bin'
    elif embedding_name == 'word2vec_zh':
        filename = 'kyubyong_wikipedia_zh.bin'
    elif embedding_name == 'word2vec_en_cca':
        filename = 'word2vec_en_cca.txt'
    elif embedding_name == 'word2vec_zh_cca':
        filename = 'word2vec_zh_cca.txt'
    elif 'english1000' in embedding_name:
        filename = f'{embedding_name}.npz'
    elif embedding_name == 'AffectVec':
        filename = 'AffectVec-data.txt'
    elif embedding_name == 'brysbaert2014':
        filename = f'{embedding_name}.npz'

    filepath_local = os.path.join(word_vector_dir_local, filename)
    tub_server_path = '/mnt/raid'  # Used to check whether to use S3 (not available at TU Berlin)

    if embedding_name not in ['word2vec_zh'] and tub_server_path not in word_vector_dir_local:
        cci = cc.get_interface(cci_bucket_name)
        get_s3_file(s3_dir=word_vector_dir_s3,
            local_dir=word_vector_dir_local,
            filename=filename,
            cci=cci)

    if 'fastText' in embedding_name:
        if 'RCSLS' in embedding_name:
            model = load_fasttext_aligned_vectors(filepath_local)
            default_embedding = np.zeros(list(model.values())[0].shape)
        else:
            model = fasttext.load_model(filepath_local)
            default_embedding = np.mean(model.get_input_matrix(), axis=0)  # Default is mean of 4M subword and word embeddings.
    elif embedding_name in ['word2vec_en', 'word2vec_zh']:
        # FIXME: This local path should be removed if it is a good performing embedding space
        if 'word2vec_zh' in embedding_name:
            filepath_local = '/auto/k1/fatma/projects/Language/Multilingual/bling_save_20200212/data/language_models/word2vec_zh/kyubyong/zh.bin'
            # This word2vec is trained on wikipedia-dump
            # filepath_local = '/auto/k1/fatma/projects/Language/Multilingual/bling_save_20200212/data/language_models/word2vec_zh/wikipediavocab/chinese-word2vec'

        if embedding_name in 'word2vec_en':
            model = KeyedVectors.load_word2vec_format(filepath_local, binary=True)
        else:
            model = KeyedVectors.load(filepath_local)
        if 'word2vec_zh' in embedding_name:
            default_embedding = model.syn0.mean(axis=0)  # Default is mean of all raw word embeddings
        else:
            default_embedding = model.vectors.mean(axis=0)  # Default is mean of all word embeddings
    elif embedding_name in ['word2vec_en_cca', 'word2vec_zh_cca']:
        model = load_fasttext_aligned_vectors(filepath_local, skip_first_line=False)
        default_embedding = np.zeros(list(model.values())[0].shape)
    elif 'english1000' in embedding_name or 'brysbaert2014' in embedding_name:
        english1000_dict = np.load(filepath_local)
        english1000_keys = english1000_dict['keys']
        english1000_values = english1000_dict['values']
        model = {k: v for k, v in zip(english1000_keys, english1000_values)}
        default_embedding = np.zeros(english1000_values[0].shape)
    elif 'AffectVec' in embedding_name:
        model = KeyedVectors.load_word2vec_format(filepath_local) # Should be Word2Vec format
        default_embedding = model.vectors.mean(axis=0)  # Default is mean of all word embeddings
    return model, default_embedding


def get_berkeley_parser_tags(ds: DataSequence,
                             language: str,
                             tag_categories: List[str] = []):
    '''
    Returns pos tag features using Berkeley Parser. If return_type=depth, provides depth of each node.

    Note: Benepar provides a full parse tree, but this function only looks at POS tags.

    args:
    ====
    ds: DataSequence containing the stimulus.
    language: Language identifier (currently supports 'en' for English and 'zh' for Chinese).
    tag_categories: Uses tag subsets specified in bling.data_loading.hard_coded_things.pos_labels_categories.
        For instance, if tag_categories == ['noun', 'verb'] then returns a 2-dimension feature space with tags
            collapsed into pos_labels_categories[language]['noun'] and pos_labels_categories[language]['verb'] categories.
        Includes all categories if tag_categories == [].
        TODO: We should run an analysis to see if the groupings we hypothesize (eg noun vs verb) actually reflect
              weights in the brain.

    '''
    def tokenize_chinese(sentence: str):
        result = jieba.tokenize(sentence)
        tokenized_sentence = [r[0] for r in result]
        return tokenized_sentence

    # Get the POS tags, sentence delimiters, and parser corresponding to the specified language.
    if language == 'en':
        parser = benepar.Parser('benepar_en3')
        sentence_end_delimiter = '.'
        pos_labels = hard_coded_things.benepar_pos_labels_en
    elif language == 'zh':
        parser = benepar.Parser('benepar_zh2')
        sentence_end_delimiter = '。'
        pos_labels = hard_coded_things.benepar_pos_labels_zh

    # Get parse tree labels.
    newdata = []
    text = np.array(ds.data)
    sentence_ends = np.array([word_index for word_index, word in enumerate(ds.data) if sentence_end_delimiter in word])
    if len(sentence_ends) == 0:
        sentence_ends = [len(ds.data) - 1]
    sentence_starts = np.array([-1] + list(sentence_ends[: -1])) + 1
    for sentence_start, sentence_end in zip(sentence_starts, sentence_ends):
        sentence = list(text[sentence_start: sentence_end + 1])
        if language == 'en':
            tree = parser.parse(sentence)
        elif language == 'zh':
            tree = parser.parse(sentence)
        sentence_length = len(sentence)
        assert(len(tree.leaves()) == sentence_length)
        for word_index in range(sentence_length):
            leaf_position = tree.leaf_treeposition(word_index)
            word_parse_path = np.zeros((1, len(pos_labels)))
            subtree = tree
            leaf_positions = leaf_position[:-1]
            for position_index, position in enumerate(leaf_positions):
                subtree = subtree[position]
                label = subtree.label()
                if position_index == len(leaf_positions) - 1:
                    word_parse_path[0, pos_labels.index(label)] += 1
            newdata.append(word_parse_path)
    newdata = np.concatenate(newdata, axis=0)

    # If using tag subsets, collapse across specified subsets.
    if len(tag_categories) > 0:
        newdata_tagsubset = np.zeros((newdata.shape[0], len(tag_categories)))
        for category_index, category in enumerate(tag_categories):
            category_tags = hard_coded_things.pos_labels_categories[language][category]
            category_tag_indices = [pos_labels.index(tag) for tag in category_tags]
            newdata_tagsubset[:, category_index] = np.sum(newdata[:, category_tag_indices], axis=-1)
        newdata = newdata_tagsubset
    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)


def clean_punctuation(word: str,
                      punctuation_characters: List = hard_coded_things.punctuation_characters):
    '''Returns the word with punctuation characters removed.'''
    cleaned_word = ''.join(character for character in word if character not in punctuation_characters)
    return cleaned_word


# Attention analysis

def get_adjmat(mat, input_tokens):
    n_layers, length, _ = mat.shape
    adj_mat = np.zeros(((n_layers+1)*length, (n_layers+1)*length))
    labels_to_index = {}
    for k in np.arange(length):
        labels_to_index[str(k)+"_"+input_tokens[k]] = k

    for i in np.arange(1,n_layers+1):
        for k_f in np.arange(length):
            index_from = (i)*length+k_f
            label = "L"+str(i)+"_"+str(k_f)
            labels_to_index[label] = index_from
            for k_t in np.arange(length):
                index_to = (i-1)*length+k_t
                adj_mat[index_from][index_to] = mat[i-1][k_f][k_t]
                
    return adj_mat, labels_to_index 


def draw_attention_graph(adjmat, labels_to_index, n_layers, length):
    A = adjmat
    G=nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(G, {(i,j): A[i,j]}, 'capacity')

    pos = {}
    label_pos = {}
    for i in np.arange(n_layers+1):
        for k_f in np.arange(length):
            pos[i*length+k_f] = ((i+0.5)*2, length - k_f)
            label_pos[i*length+k_f] = (i*2, length - k_f)

    index_to_labels = {}
    for key in labels_to_index:
        index_to_labels[labels_to_index[key]] = key.split("_")[-1]
        if labels_to_index[key] >= length:
            index_to_labels[labels_to_index[key]] = ''

    #plt.figure(1,figsize=(20,12))

    # nx.draw_networkx_nodes(G,pos,node_color='green', node_size=50)
    # nx.draw_networkx_labels(G,pos=label_pos, labels=index_to_labels, font_size=10)

    all_weights = []
    #4 a. Iterate through the graph nodes to gather all the weights
    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight']) #we'll use this when determining edge thickness

    #4 b. Get unique weights
    unique_weights = list(set(all_weights))

    #4 c. Plot the edges - one by one!
    for weight in unique_weights:
        #4 d. Form a filtered list with just the weight you want to draw
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
        #4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
        
        w = weight #(weight - min(all_weights))/(max(all_weights) - min(all_weights))
        width = w
        # nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width, edge_color='darkblue')
    
    return G

def get_attention_graph(adjmat, labels_to_index, n_layers, length):
    A = adjmat
    G=nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(G, {(i,j): A[i,j]}, 'capacity')

    pos = {}
    label_pos = {}
    for i in np.arange(n_layers+1):
        for k_f in np.arange(length):
            pos[i*length+k_f] = ((i+0.5)*2, length - k_f)
            label_pos[i*length+k_f] = (i*2, length - k_f)

    index_to_labels = {}
    for key in labels_to_index:
        index_to_labels[labels_to_index[key]] = key.split("_")[-1]
        if labels_to_index[key] >= length:
            index_to_labels[labels_to_index[key]] = ''

    #plt.figure(1,figsize=(20,12))

    nx.draw_networkx_nodes(G,pos,node_color='green', node_size=50)
    nx.draw_networkx_labels(G,pos=label_pos, labels=index_to_labels, font_size=10)

    all_weights = []
    #4 a. Iterate through the graph nodes to gather all the weights
    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight']) #we'll use this when determining edge thickness

    #4 b. Get unique weights
    unique_weights = list(set(all_weights))

    #4 c. Plot the edges - one by one!
    for weight in unique_weights:
        #4 d. Form a filtered list with just the weight you want to draw
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
        #4 e. I think multiplying by [num_nodes/sum(all_weights)] makes the graphs edges look cleaner
        
        w = weight #(weight - min(all_weights))/(max(all_weights) - min(all_weights))
        width = w
        nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width, edge_color='darkblue')
    
    return G

def compute_flows(G, labels_to_index, input_nodes, length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in labels_to_index:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer*length+v ] = flow_value
            flow_values[u] /= flow_values[u].sum()
            
    return flow_values

def compute_node_flow(G, labels_to_index, input_nodes, output_nodes,length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in output_nodes:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v, flow_func=nx.algorithms.flow.edmonds_karp)
                flow_values[u][pre_layer*length+v ] = flow_value
            flow_values[u] /= flow_values[u].sum()
            
    return flow_values

def compute_joint_attention(att_mat, add_residual=True):
    if add_residual:
        residual_att = np.eye(att_mat.shape[1])[None,...]
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[...,None]
    else:
        aug_att_mat =  att_mat
    
    joint_attentions = np.zeros(aug_att_mat.shape)

    layers = joint_attentions.shape[0]
    joint_attentions[0] = aug_att_mat[0]
    for i in np.arange(1,layers):
        joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1])
        
    return joint_attentions

def plot_attention_heatmap(att, s_position, t_positions, sentence):

    cls_att = np.flip(att[:,s_position, t_positions], axis=0)
    xticklb = input_tokens= list(itertools.compress(['<cls>']+sentence.split(), [i in t_positions for i in np.arange(len(sentence)+1)]))
    yticklb = [str(i) if i%2 ==0 else '' for i in np.arange(att.shape[0],0, -1)]
    ax = sns.heatmap(cls_att, xticklabels=xticklb, yticklabels=yticklb, cmap="YlOrRd")
    return ax


def convert_adjmat_tomats(adjmat, n_layers, l):
    mats = np.zeros((n_layers,l,l))
   
    for i in np.arange(n_layers):
        mats[i] = adjmat[(i+1)*l:(i+2)*l,i*l:(i+1)*l]
       
    return mats