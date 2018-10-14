import glob
import os
import pickle

import pandas as pd
import numpy as np
import gensim
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from preprocess import pipe, get_stopwords
from sentence_vector import get_vector_from_sentence


VOCAB_SIZE = 28648
CORPUS_SIZE = 178843


def get_pandas_df(all_csv=True):
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
    res = res.reset_index()
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


def filter_ttf(ttf, threshold=4):
    return dict([(term, freq) for term, freq in ttf.items() if freq >= threshold])


def create_vocab_to_idx_map(ttf):
    vocab_dict = {}
    for i, key in enumerate(ttf):
        vocab_dict[key] = i
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


def get_crude_sentence_vector(sentence, vocab_dict):
    """ Get sentence vector for 1 sentence

    :tokens_df: list of words in sentence
    :vocab_dict: map from word to index
    """
    vec = np.zeros(VOCAB_SIZE)
    for word in sentence:
        word = word.encode('ascii', 'ignore').decode('utf-8')
        if word in vocab_dict:
            vec[vocab_dict[word]] += 1
    return vec


def sentiment(df, use_cached=True):
    if use_cached and os.path.isfile('./cache/_cached_sentiment_dicts.pkl'):
        with open('./cache/_cached_sentiment_dicts.pkl', 'rb') as infile:
            sentiment_dicts = pickle.load(infile)
            return sentiment_dicts
    vader = SentimentIntensityAnalyzer()
    df['sentiment'] = df['message'].apply(vader.polarity_scores)
    sentiment_dicts = df.set_index(['status_id'])['sentiment'].to_dict()
    with open('./cache/_cached_sentiment_dicts.pkl', 'wb') as outfile:
        pickle.dump(sentiment_dicts, outfile)
    return sentiment_dicts


def sentence_vector(df, use_cached=True):
    if use_cached and os.path.isfile('./cache/_cached_sentence_vectors.pkl'):
        with open('./cache/_cached_sentence_vectors.pkl', 'rb') as infile:
            vectors = pickle.load(infile)
            return vectors
    model = gensim.models.Doc2Vec.load('doc2vec.model')
    df['vector'] = df['message'].apply(gensim.utils.simple_preprocess).apply(model.infer_vector)
    vectors = df.set_index(['status_id'])['vector']
    with open('./cache/_cached_sentence_vectors.pkl', 'wb') as outfile:
        pickle.dump(vectors, outfile)
    return vectors


def reaction(df, use_cached=True):
    if use_cached and os.path.isfile('./cache/_cached_reaction.pkl'):
        with open('./cache/_cached_reaction.pkl', 'rb') as infile:
            vectors = pickle.load(infile)
            return vectors
    vectors = df[['status_id', 'num_loves', 'num_hahas', 'num_wows', 'num_sads', 'num_angrys']].set_index(['status_id']).T.to_dict('list')
    with open('./cache/_cached_reaction.pkl', 'wb') as outfile:
        pickle.dump(vectors, outfile)
    return vectors


def one_hot(df, key):
    """ Get one hot encoding for given key
    :key: column in df to encode
    """
    one_hot_encoding_of_key = pd.get_dummies(df[key], drop_first=True).join(df.status_id).set_index('status_id')
    one_hot_encoding_of_key['onehot'] = one_hot_encoding_of_key.values.tolist()


def create_samples(n, batch_size=50, use_cache=True, prepared=None):
    """ Returns X, y where X is INPUT_LENx1 input vector and y is OUTPUT_LENx1 output vector
    Features:
        - sentence vector
        - sentiment
        - source
        - length
        - type
    Output:
        - hahas
        - loves
        - all the other ones
    """
    INPUT_LEN = 54
    OUTPUT_LEN = 5
    if prepared is None:
        df = get_pandas_df()
        df = prepare(df)
    else:
        df = prepared
    s = sentiment(df, use_cache)
    # type_dict = one_hot(df, 'status_type')
    # source_dict = one_hot(df, 'source')
    v = sentence_vector(df, use_cache)
    r = reaction(df, use_cache)

    X = np.empty([batch_size, INPUT_LEN])
    y = np.empty([batch_size, OUTPUT_LEN])
    for i, sid in enumerate(df['status_id'][(n * batch_size): (n + 1) * batch_size]):
        if not i % (batch_size // 10):
            print(i)
        in_vector = np.concatenate((np.array(v[sid]), np.array(list(s[sid].values()))))
        X[i] = in_vector
        reaction_array = np.array(r[sid], dtype=np.int64)
        if np.count_nonzero(reaction_array) == 0:
            y[i] = np.ones([OUTPUT_LEN])
        else:
            y[i] = reaction_array / np.sum(reaction_array)
    return X, y


def get_input_vector(sentence):
    v = np.array(get_vector_from_sentence(sentence))
    vader = SentimentIntensityAnalyzer()
    sentiment = np.array(list(vader.polarity_scores(sentence).values()))
    print(v)
    print(sentiment)
    return np.concatenate((v, sentiment))
