# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 22:03:34 2017

@author: Admin
"""

##### IMPORTS ####
import pandas as pd
import tensorflow as tf
import re
import tempfile
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
from textblob import TextBlob
import numpy as np
##################


df_predict = pd.read_csv('bag_of_words.csv')
#### Select smaller column list than stored model.
df_predict = df_predict.iloc[:, 0:300]
df_predict = df_predict.loc[400:500]

pred_column = 'BELOW_3'
df_predict = df_predict.drop('Unnamed: 0', axis=1)

#############################

base_columns = []

for column in df_predict:
    if column != pred_column:
        if df_predict[column].dtype == 'object':
            feature = tf.feature_column.categorical_column_with_hash_bucket(column, hash_bucket_size=10000)
        elif df_predict[column].dtype == 'int64':
            feature = tf.feature_column.numeric_column(column)
        elif df_predict[column].dtype == 'float64':
            df_predict[column] = df_predict[column].astype(int)
            feature = tf.feature_column.numeric_column(column)
        else:
            feature = tf.feature_column.categorical_column_with_hash_bucket(column, hash_bucket_size=10000)
        base_columns.append(feature)

#crossed columns
crossed_columns = []

########################################





#predict is done only once, without shuffle, since we are writing results nex
#next to data inputs.
predict_input = tf.estimator.inputs.pandas_input_fn(
       x=df_predict,
       batch_size=100,
       num_epochs=1,
       shuffle=False,
       num_threads=1)

model_dir = 'C:\\Users\\Admin\\Documents\\Spyder_Projects\\hotel_reviews_text\\models'
m = tf.estimator.LinearClassifier(
     model_dir=model_dir, feature_columns=base_columns + crossed_columns)


#predict
predictions = m.predict(input_fn=predict_input)

#create list to save predictions to.
predict_list = []
prob_list = []

# For each input row, add binary prediction to new column.
for i, p in enumerate(predictions):
  #print("Prediction %s: %s" % (i + 1, p['classes']))
  #predict_list.append(p['classes'])
  for x in p['classes']:
     predict_list.append(re.sub("\D", "", str(x)))
  #print("Probabilities %s: %s" % (i + 1, p['probabilities']))
  prob_list.append(str(p['probabilities']))

pred_series = pd.Series(predict_list)
df_predict['Prediction'] = pred_series.values

prob_series = pd.Series(prob_list)
df_predict['Probabilities'] = prob_series.values

# Output prediction results to file.
df_predict.to_csv('restored_model_prediction_results.csv', sep=',', encoding='utf-8')
