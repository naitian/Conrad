import glob
import pandas as pd


def get_pandas_df():
    """Returns dataframe with all files
    :returns: TODO

    """
    path = r'./data'  # use your path
    allFiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for csv_file in allFiles:
        df = pd.read_csv(csv_file, index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    return frame
