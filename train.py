import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from dataset import get_pandas_df, prepare, create_samples


df = get_pandas_df()
df = prepare(df)

X, y = create_samples(0, batch_size=178843, prepared=df)

mr = MultiOutputRegressor(RandomForestRegressor(n_estimators=50,
                                                max_depth=30,
                                                random_state=0,
                                                verbose=True))

print("training model")
mr.fit(X[:10000], y[:10000])
with open('random_forest_model.normed.pkl', 'wb') as f:
    pickle.dump(mr, f)
print("finished training")

# mr = pickle.load(open('random_forest_model.normed.pkl', 'rb'))

print(mr.score(X[10000:], y[10000:]))
