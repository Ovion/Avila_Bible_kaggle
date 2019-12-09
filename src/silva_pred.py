import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('input/training_dataset.csv')
test = pd.read_csv('input/test_dataset.csv')

X_train = data.drop(['scribe', 'id'], axis=1)
y_train = data.scribe
X_test = test.drop(['id'], axis=1)

print('Training...')
rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rfc.fit(X_train, y_train)
print('Doing a prediction...')
y_pred = rfc.predict(X_test)

print('Saving...')
submit = pd.DataFrame({
    'id': test['id'],
    'scribe': y_pred
})

submit.to_csv('output/submit_silva.csv', index=False)
print('Done')