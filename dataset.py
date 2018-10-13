import glob
import pandas as pd


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


def preprocess(df, cat=False):
    res = df.dropna(subset=['status_message'])
    res = res.drop_duplicates(subset=['status_id'])
    res['link_name'] = res['link_name'].fillna('')
    if cat:
        res['🙀'] = res['status_message'] + ' ' + res['link_name']
    res['message'] = res['status_message'] + ' ' + res['link_name']
    res = res[res['status_message'].str.len() < 1000]
    res = res[res['num_reactions'] - res['num_likes'] > 11]
    return res
