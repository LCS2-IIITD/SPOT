import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE

model_name = "phase1"

train_file = './data/pretrainedREP-{}.csv'.format(model_name)
df_train = pd.read_csv(train_file)

dev_file = './data/pretrainedREP-dev-{}.csv'.format(model_name)
df_dev = pd.read_csv(dev_file)

test_file = './data/pretrainedREP-test-{}.csv'.format(model_name)
df_test = pd.read_csv(test_file)

X_train = df_train.drop(columns=['768'])
y_train = df_train['768']

X_dev = df_dev.drop(columns=['768'])
y_dev = df_dev['768']

X_test = df_test.drop(columns=['768'])
y_test = df_test['768']


print('Original dataset shape %s' % Counter(y_train))
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print('Resampled dataset shape %s' % Counter(y_res))

trainup = pd.concat([X_res, y_res], axis=1)

trainup.to_csv("./data/data_iter_train_{}_up.csv".format(model_name),index=None)