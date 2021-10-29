# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See LICENSE for more details.
#
# Copyright: Red Hat Inc. 2021
# Author: Songyu Liu <sonliu@redhat.com>

import re
import gensim
import gensim.downloader
from gensim.corpora import Dictionary
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import nltk
from pandarallel import pandarallel

# Initialization
pandarallel.initialize()

try:
    from nltk import word_tokenize
except FileNotFoundError as fnf_error:
    # In Python 3.x, using as is required to assign an exception to a variable.

    print(fnf_error)
    nltk.download("punkt")
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
    remove_stopwords,
)
from nltk.stem import WordNetLemmatizer


def preprocess_gensim(string):
    """
    a wrapper around the library functions.
    Return type    list of str
    """
    nltk.download("wordnet")
    lemmatizer = WordNetLemmatizer()
    # need to provide the context in which you want to lemmatize that is the parts-of-speech (POS).
    filters = [
        strip_tags,
        lambda x: x.lower(),
        strip_punctuation,
        strip_multiple_whitespaces,
        strip_numeric,
        remove_stopwords,
        lemmatizer.lemmatize,
    ]
    return preprocess_string(string, filters=filters)


def clean_numbers(x):
    if bool(re.search(r"\d", x)):  # find a digit
        x = re.sub("[0-9]{5,}", "#####", x)
        #  a digit repeats more than 5 times
        x = re.sub("[0-9]{4}", "####", x)
        x = re.sub("[0-9]{3}", "###", x)
        x = re.sub("[0-9]{2}", "##", x)
    return x


def preprocess(string):
    # lower the text
    string = string.lower()

    # Clean the text. similar to isalnum
    string = re.sub(r"[^\w\s]", " ", string)
    # \w If the ASCII flag is used, only [a-zA-Z0-9_] is matched.
    string = string.replace("_", " ")
    # underscores are replaced. other symbols have been removed

    # Clean numbers
    string = clean_numbers(string)

    # Tokenize the sentences. now we get a list

    return word_tokenize(string)


def training_words_in_word2vector(word_to_vec_map, word_to_index):
    """
    input:
        word_to_vec_map: a word2vec model loaded using gensim.
        word_to_index: word to index mapping from training set
    """

    vocab_size = len(word_to_index)
    count = 0
    # Set each row "idx" of the embedding matrix to be
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        if word in word_to_vec_map:
            count += 1

    return print(
        "Found {} words present from {} training vocabulary in the set of pre-trained word vector".format(
            count, vocab_size
        )
    )


def pretrained_embedding_matrix(word_to_vec_map, word_to_index, emb_mean, emb_std):
    """
    input:
        word_to_vec_map: a word2vec model loaded using gensim.models
        word_to_index: word to index mapping from training set
    """

    # adding 1 to fit Keras embedding (requirement)
    vocab_size = len(word_to_index)
    # hard code dimensionality of pre-trained word vectors 100

    # initialize the matrix with generic normal distribution values
    embed_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, 100))

    # Set each row "idx" of the embedding matrix to be
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        if word in word_to_vec_map:
            embed_matrix[idx] = word_to_vec_map.get_vector(word)
    return embed_matrix


def create_embedding(train_x):
    """the embedding corresponds to a word to index mapping. both are saved.
    embedding should only have access to the training data"""
    wv = gensim.downloader.load("glove-wiki-gigaword-100")
    dct = Dictionary(train_x.tolist())  # initialize a Dictionary
    word_to_index = dct.token2id  # a python built in dict that maps keywords to int
    training_words_in_word2vector(wv, word_to_index)
    emb_mean = wv.vectors.mean()
    emb_std = wv.vectors.std()
    embedding_matrix = pretrained_embedding_matrix(wv, word_to_index, emb_mean, emb_std)
    word_to_tensor = {
        "word_to_index": word_to_index,
        "embedding_matrix": embedding_matrix,
    }
    torch.save(word_to_tensor, "data/word_to_tensor.pt")
    return word_to_tensor


def series_to_tensor(series, train=0):
    """
    input: the text column of the pandas dataframe.
    returns padded int sequences.
    """
    maxlen = series.apply(lambda x: len(x)).quantile(0.9)
    # the longest commit message should be considered as an outlier
    print(f"max number of words in a commit to use {maxlen}")
    series = series.apply(lambda x: x[: int(maxlen)])

    if train:
        d = create_embedding(series)
    else:
        d = torch.load("data/word_to_tensor.pt")
    word_to_index = d["word_to_index"]

    def word_to_tensor(word):
        """
        maps a word to a tensor representing its index.
        """
        # Dictionary data types represent the implementation of hash tables.
        index = word_to_index[word]
        return torch.tensor(index)

    def sent_to_tensor(sent):
        """maps a sentence to a stacked tensor. each row of it is one index."""
        text = list(filter(lambda word: word in word_to_index, sent))
        if len(text) == 0:
            print("commit contains too few words")
            breakpoint()
        sentence = list(map(word_to_tensor, text))
        return torch.stack(sentence)

    X_list = series.parallel_apply(sent_to_tensor).tolist()
    pad = pad_sequence((X_list), batch_first=True)
    # This function returns a Tensor of size T x B x * or B x T x * where T is the length of the longest sequence. B is the number of data points. This function assumes trailing dimensions and type of all the Tensors in sequences are same.
    return pad


"""
def parallel(big_df, process_apply):

    cpu_count = mp.cpu_count()
    with mp.Pool(processes=cpu_count) as p:
        split_dfs = np.array_split(big_df,cpu_count)    # a list of small arrays
        breakpoint()
        pool_results = p.map(process_apply, split_dfs)

    # merging parts processed by different processes
    parts = pd.concat(pool_results, axis=0)
    return parts
"""
