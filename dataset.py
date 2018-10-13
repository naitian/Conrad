import glob
import pandas as pd
import pickle
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
    if use_cached:
        with open('_cached_bag_of_words.pkl', 'rb') as infile:
            tokens = pickle.load(infile)
            return tokens
    stopwords = get_stopwords()
    tokens = {}
    for n in range(df.shape[0]):
        if not n % 1000:
            print(n)
        tokens[df.iloc[n]['status_id']] = pipe(df.iloc[n]['message'], stopwords)
    with open('_cached_bag_of_words.pkl', 'wb') as outfile:
        pickle.dump(tokens, outfile)
    return tokens
