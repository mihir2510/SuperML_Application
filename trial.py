from pandas import read_csv
import pickle
from auto_machine_learning.data_preprocessing.preprocessing import get_features
dataset = read_csv('./data/diabetes.csv')
label = 'Outcome'
features = get_features(dataset, label)
X, y = dataset[features], dataset[label]

with open('./model.pickle.sav', 'rb') as f:
    model = pickle.load(f)

print(dataset.iloc[0])
print(model.predict([X.iloc[0]]))

# ypred = model.predict(X)
from auto_machine_learning.metrics.metrics import get_model_metrics
print(get_model_metrics(model, [0, 1], 'classification', X, y))

