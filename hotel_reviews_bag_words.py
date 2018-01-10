# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 13:24:26 2017

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


df = pd.read_csv('hotel_review_data.csv')

df['words'] = df['reviews'].apply(lambda x:TextBlob(x).words)

df['words'] = df['words'].apply(lambda x: [item for item in x if item not in stop])

df['words'] = df['words'].apply(lambda x: " ".join(x))

#df['words'] = df['words'].apply(lambda x: x.encode('utf-8').strip())

df['words'] = df['words'].replace(to_replace="([^\s\w]|_)+", value="", regex=True)

df['words'] = df['words'].apply(lambda x: x.encode("ascii", errors="ignore").decode())

df = df.drop('reviews', 1)

df.columns = ['BELOW_3', 'RATING_SCORE', 'WORD_LIST']



words = pd.Series([y for x in df.WORD_LIST.values.flatten() for y in x.split()]).value_counts()
words = words.index.tolist()



df2 = df.join(df.WORD_LIST.str.get_dummies(' ').loc[:, words])

df2 = df2.drop('WORD_LIST', axis=1)
df2 = df2.drop('RATING_SCORE', axis=1)

df2 = df2[df2.columns.drop(list(df.filter(regex="'")))]

########### SELECT TOP 300 ##########
df2 = df2.iloc[:, 0:300]

############################
################################
#input_file = 'bag_of_words.csv'

#file = open(input_file)
#numline = len(file.readlines())
#
#train_size = int(numline * .95)
#test_size = int(numline - numline *.95) - 2



df_train = df2.sample(frac=0.9)

df_test = df2.sample(frac=0.1)
df_predict = df2.sample(frac=0.1)

df_train = df_train.dropna(axis=0, how='any')
df_test = df_test.dropna(axis=0, how='any')
df_predict = df_predict.dropna(axis=0, how='any')

#df_train = df_train[df_train.columns.drop(list(df.filter(regex='/?.')))]
#df_test = df_test[df_test.columns.drop(list(df.filter(regex='/?.')))]
#df_predict = df_predict[df_predict.columns.drop(list(df.filter(regex='/?.')))]
#
#df_train = df_train.drop('WORD_LIST' , axis=1)
#df_test = df_test.drop('WORD_LIST' , axis=1)
#df_predict = df_predict.drop('WORD_LIST' , axis=1)


###################################



pred_column = 'BELOW_3'
pred_label = True

base_columns = []

for column in df_test:
    if column != pred_column:
        if df_test[column].dtype == 'object':
            feature = tf.feature_column.categorical_column_with_hash_bucket(column, hash_bucket_size=10000)
        elif df_test[column].dtype == 'int64':
            feature = tf.feature_column.numeric_column(column)
        elif df_test[column].dtype == 'float64':
            df_test[column] = df_test[column].astype(int)
            feature = tf.feature_column.numeric_column(column)
        else:
            feature = tf.feature_column.categorical_column_with_hash_bucket(column, hash_bucket_size=10000)
        base_columns.append(feature)

#crossed columns
crossed_columns = []



################ SETUP TENSORFLOW INPUTS ##########################
# Establish Labels to generate with model.
df_train['train_labels'] = df_train[pred_column] == pred_label
df_test['test_labels'] = df_test[pred_column] == pred_label
  
train_input = tf.estimator.inputs.pandas_input_fn(
       x=df_train,
       y=df_train["train_labels"],
       batch_size=100,
       num_epochs=None,
       shuffle=True,
       num_threads=5)
  
test_input = tf.estimator.inputs.pandas_input_fn(
       x=df_test,
       y=df_test["test_labels"],
       batch_size=100,
       num_epochs=None,
       shuffle=True,
       num_threads=5)

#predict is done only once, without shuffle, since we are writing results nex
#next to data inputs.
predict_input = tf.estimator.inputs.pandas_input_fn(
       x=df_predict,
       batch_size=100,
       num_epochs=1,
       shuffle=False,
       num_threads=1)

##
# Define where tensorflow model is saved.  This hardcoded version will break.
# files need to be deleted after run.  Use tempfolder to deal with this.
model_dir = tempfile.mkdtemp()
m = tf.estimator.LinearClassifier(
     model_dir=model_dir, feature_columns=base_columns + crossed_columns)


############ Training and Evaluating Our Model #####################
 #train
m.train(input_fn=train_input, steps=1000)
  
 #eval
results = m.evaluate(input_fn=test_input, steps=100)

#Show eval results
print("model directory = %s" % model_dir)
for key in sorted(results):
    tf.summary.scalar(key, results[key])
    print("%s: %s" % (key, results[key]))
print ('######################################')

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
df_predict.to_csv('prediction_results.csv', sep=',', encoding='utf-8')

#### GET COUNTS OF TRUE PREDICTION COLUMNS - TO NARROW IN ON WORDS THAT MATTER


    

        
