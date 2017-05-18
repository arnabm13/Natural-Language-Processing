"""
See README.md
Reference : http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
"""
from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import urllib.request
from nltk.corpus import brown
import gensim
import sys
import tensorflow as tf
from boltons import iterutils
import operator
def download_data():
    url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
    urllib.request.urlretrieve(url, 'train.txt')
    url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'
    urllib.request.urlretrieve(url, 'test.txt')


def read_data(filename):
    input_file = open(filename, 'r')
    lines = input_file.read()
    list_lines = lines.splitlines()
    list_lines.append('')
    list_sentence = []
    list_result = []
    list_ner=[]
    for token in list_lines:
        if not token =='':
            list_sentence.append(token.split()[0])
        else:
            if not '-DOCSTART-' in list_sentence:
                list_result.append(list_sentence.copy())
            list_sentence.clear()
    list_lines_ner=[x for x in list_lines if x not in('','-DOCSTART- -X- -X- O')]
    for items in list_lines_ner:
        list_ner.append(items.split()[3])
    return list_result,list_ner

def input_data(data,w2v_model):
    list_words=[]
    ti=[]
    for sublist in data:
        for item in sublist:
            list_words.append(item)
    for words in list_words:
        ti.append(np.array(iterutils.chunked(list(w2v_model.wv[words]), 1)))
    return ti

def output_data(data):
    list_labels=[]
    tag={'I-LOC':[1,0,0,0,0],
         'I-MISC':[0,1,0,0,0],
         'I-ORG':[0,0,1,0,0],
         'I-PER':[0,0,0,1,0],
         'O':[0,0,0,0,1]}
    for label in data:
        list_labels.append(tag.get(label))
    return  list_labels
def predicted_output(data):
    list_tag=[]
    tag_index={0:'I-LOC',1:'I-MISC',2:'I-ORG',3:'I-PER',4:'O'}
    for items in data:
        index, value = max(enumerate(items), key=operator.itemgetter(1))
        list_tag.append(tag_index.get(index))
    return  list_tag

def confusion(true_labels, pred_labels):
    list_labels=zip(true_labels,pred_labels)
    count_lables=Counter(list_labels)
    ner_labels=sorted(list(set(true_labels)))
    df=pd.DataFrame()
    for i,row in enumerate(ner_labels):
        row_data={}
        for j,column in enumerate(ner_labels):
            row_data[column]=count_lables[row,column]
        df=df.append(pd.DataFrame.from_dict({row:row_data},orient='index'))
    return df
    pass

def evaluate(confusion_matrix):
    row_labels=['precision','recall','f1']
    df=pd.DataFrame()
    for row in row_labels:
        row_data={}
        for column in list(confusion_matrix):
            if row=='precision':
                row_data[column]=(confusion_matrix.get_value(column,column))/(sum(confusion_matrix[column]))
            if row=='recall':
                row_data[column]=(confusion_matrix.get_value(column,column))/(sum(confusion_matrix.loc[column]))
            if row=='f1':
                p=df.get_value('precision',column)
                r=df.get_value('recall',column)
                row_data[column]=(2*p*r)/(p+r)
        df = df.append(pd.DataFrame.from_dict({row: row_data}, orient='index'))
    return df
    pass

def average_f1s(evaluation_matrix):
    ###TODO
    df=evaluation_matrix.drop('O',1)
    avg_f1=sum(df.loc['f1'])/len(list(df))
    return avg_f1
    pass


if __name__ == '__main__':

    download_data()
    train_data,train_labels = read_data('train.txt')
    test_data, test_labels = read_data('test.txt')
    w2v_model = gensim.models.Word2Vec(train_data+test_data, min_count=1, size=50, window=5)
    train_input=input_data(train_data,w2v_model)
    train_output=output_data(train_labels)
    test_input=input_data(test_data,w2v_model)
    test_output=output_data(test_labels)
    print('test and training data loaded')

    data = tf.placeholder(tf.float32, [None, 50,1])  # Number of examples, number of input, dimension of each input
    target = tf.placeholder(tf.float32, [None, 5])
    num_hidden = 20
    cell = tf.contrib.rnn.LSTMCell(num_hidden,forget_bias=1.0,activation=tf.sigmoid)
    val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)
    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)
    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    init_op = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init_op)

    batch_size = 1
    no_of_batches = int((len(train_input)) / batch_size)
    epoch = 15
    for i in range(epoch):
        ptr = 0
        o_count = 0
        for j in range(no_of_batches):
            inp, out = train_input[ptr:ptr + batch_size], train_output[ptr:ptr + batch_size]
            ptr += batch_size
            if out[0] == [0,0,0,0,1]:
                o_count = o_count + 1
                if o_count < 1100:
                    sess.run(minimize, {data: inp, target: out})

            else:
                sess.run(minimize, {data: inp, target: out})

        print("Epoch ", str(i))
    incorrect = sess.run(error, {data: test_input, target: test_output})

    list_prediction=[]
    for input in test_input:
        list_prediction.append(sess.run(prediction, {data: [input]}).tolist()[0])
    predicted_tag=predicted_output(list_prediction)
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    sess.close()
    with open('output2.txt', 'a') as f:
        confusion_matrix = confusion(test_labels, predicted_tag)
        print('confusion matrix:\n%s\n' % str(confusion_matrix),file=f)

        evaluation_matrix = evaluate(confusion_matrix)
        print('evaluation matrix:\n%s\n' % str(evaluation_matrix),file=f)

        print('average f1s: %f\n' % average_f1s(evaluation_matrix),file=f)