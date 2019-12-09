import pandas as pd
from sklearn.metrics import mean_squared_error

import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.random_forest import H2ORandomForestEstimator as rf

h2o.init(nthreads=-1, max_mem_size=12)

data = h2o.import_file('input/training_dataset.csv')
pred = h2o.import_file('input/test_dataset.csv')
data = data.drop(['id'], axis=1)

splits = data.split_frame(ratios=[0.8], seed=1)
train = splits[0]
test = splits[1]

y = 'scribe'
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

X_test = pred.drop(['id'], axis=1)

aml = rf(ntrees = 500, seed=1, nfolds=0)
aml.train(y=y, training_frame=train, validation_frame=test)

y_pred = aml.predict(X_test)

submit = pred['id']
submit['scribe'] = y_pred['predict']
submit = submit.as_data_frame(use_pandas=True)

submit.to_csv('output/submit_aqua_rf.csv', index=False)

print('Saving params in records.txt ...')
with open('output/records_aqua.txt', "a+") as file:
    file.write(
        f'''Models:\n {aml.leaderboard} \n\n'''
    )

h2o.cluster().shutdown()
