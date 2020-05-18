import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#________Loading Data________________
dftrain = pd.read_csv('file:///home/bhavya/Datasets/student/student-por.csv',sep=";")
dftrain=dftrain[["G1","G2","G3","traveltime","studytime","failures","absences","health","goout","schoolsup","internet","higher"]]
x_train, x_test = train_test_split(dftrain, test_size=0.2)
y_train = x_train.pop('G3')
y_test = x_test.pop('G3')

#_____Data Preprocessing_____________
CATEGORICAL_COLUMNS=["schoolsup","internet","higher"]
Column_Names = ["G1","G2","traveltime","studytime","failures","absences","health","goout"]
feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in Column_Names:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
#print(feature_columns)

#________Creating input function_____________
def make_input_fn(data_df, label_df, num_epochs=12, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    print(ds)
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(x_train, y_train)
eval_input_fn = make_input_fn(x_test, y_test, num_epochs=1, shuffle=False)

#________Building the model_____________
estimator = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        model_dir="Evaluate")

#_______Training and testing model____________
estimator.train(train_input_fn)
result = estimator.evaluate(eval_input_fn)
print("Loss : ",result['loss'])

#___Predicting the student performance______________
y = estimator.predict(make_input_fn(x_test, y_test, num_epochs=1, shuffle=False) )

prediction = [p["predictions"] for p in y]
list_pred=[]
for num in prediction:
    list_pred.append(num[0])

d = {'y_test': y_test, 'final_preds': list_pred}
df = pd.DataFrame(data=d)
print(df[:10])

print("Mean Squared error : ",mean_squared_error(y_test,prediction))
print("Mean absolute error : ",mean_absolute_error(y_test,prediction))















