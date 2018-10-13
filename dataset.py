import glob
import os
import pandas as pd
import pickle
import sys
from time import time

from preprocess import pipe, get_stopwords

def get_pandas_df(all_csv=False):
    """Returns dataframe with all files
    :returns: TODO

    NOTE: these return dataframes that look different
    to each other. so. yeah.
    """
    if all_csv:
        frame = pd.read_csv('./data/all.csv', encoding="ISO-8859-1", header=0)
        return frame
    path = r'./data'  # use your path
    allFiles = glob.glob(path + "/*_facebook_statuses.csv")
    frame = pd.DataFrame()
    list_ = []
    for csv_file in allFiles:
        df = pd.read_csv(csv_file, index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    return frame


def prepare(df, cat=False):
    res = df.dropna(subset=['status_message'])
    res = res.drop_duplicates(subset=['status_id'])
    res['link_name'] = res['link_name'].fillna('')
    if cat:
        res['🙀'] = res['status_message'] + ' ' + res['link_name']
    res['message'] = res['status_message'] + ' ' + res['link_name']
    res = res[res['status_message'].str.len() < 1000]
    res = res[res['num_reactions'] - res['num_likes'] > 11]
    return res


def extract_bag_of_words(df, use_cached=True):
    if use_cached and os.path.isfile('./cache/_cached_bag_of_words.pkl'):
        with open('./cache/_cached_bag_of_words.pkl', 'rb') as infile:
            tokens = pickle.load(infile)
            return tokens
    stopwords = get_stopwords()
    tokens = {}
    for n in range(df.shape[0]):
        if not n % 1000:
            print(n)
        tokens[df.iloc[n]['status_id']] = pipe(df.iloc[n]['message'], stopwords)
    with open('./cache/_cached_bag_of_words.pkl', 'wb') as outfile:
        pickle.dump(tokens, outfile)
    return tokens


def extract_ttf(tokens_df):
    ttf = {}
    for status_id, tokens in tokens_df.items():
        for token in tokens:
            token = token.encode('ascii', 'ignore').decode('utf-8')
            if not token:
                continue
            if token not in ttf:
                ttf[token] = 0
            ttf[token] += 1
    return ttf

def filter_ttf(ttf, threshold = 4):
    return dict([(term, freq) for term, freq in ttf.items() if freq >= threshold])

def create_vocab_dict_from_ttf(ttf):
    vocab_dict = {}
    for key in ttf:
        vocab_dict[key] = 0
    return vocab_dict

def one_hot_encode(tokens_df, ttf, use_cached=True):
    if use_cached and os.path.isfile('./cache/_cached_one_hot_encoded.pkl'):
        with open('./cache/_cached_one_hot_encoded.pkl', 'rb') as infile:
            tokens_one_hot = pickle.load(infile)
            return tokens_one_hot
    tokens_one_hot = {}
    for status_id, tokens in tokens_df.items():
        tokens_one_hot[status_id] = set()
        for token in tokens:
            if token in ttf:
                tokens_one_hot[status_id].add(token)
    with open('./cache/_cached_one_hot_encoded.pkl', 'wb') as outfile:
        pickle.dump(tokens_one_hot, outfile)
    return tokens_one_hot

def main():
    # Get Facebook posts data frame
    df = get_pandas_df()
    df = prepare(df)

    # Create one-hot bag of words structure
    tokens_df = extract_bag_of_words(df)
    ttf = extract_ttf(tokens_df)
    ttf = filter_ttf(ttf, threshold=4)
    vocab_dict = create_vocab_dict_from_ttf(ttf)
    tokens_one_hot = one_hot_encode(tokens_df, ttf, vocab_dict)

    print(tokens_one_hot)
main()