import simplejson as json
import collections
import csv
import itertools
import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from collections import Counter, defaultdict


def review_check(data, lis):
    if data['type'] == 'review' and data['stars'] < 3:
        if data['business_id'] in lis:
            return True

    return False


def Reader(inp, op, colomn_list, lis):
    rstrnt_cntr = {}
    review_cnt = 0
    with open(op, 'w', newline='') as wr:
        csvw = csv.writer(wr)
        csvw.writerow(list(colomn_list))
        cnt = 0
        with open(inp, encoding='utf-8') as r:
            for l in r:
                tmp = []
                data = json.loads(l)
                if review_check(data, lis):
                    if data['business_id'] not in rstrnt_cntr.keys():
                        rstrnt_cntr[data['business_id']] = 1
                    review_cnt = review_cnt + 1
                    for m in colomn_list:
                        line_value = data[m]
                        if isinstance(line_value, str):
                            tmp.append('{0}'.format(line_value.encode('utf-8')))
                        elif line_value is not None:
                            tmp.append('{0}'.format(line_value))
                        else:
                            tmp.append('')
                    csvw.writerow(tmp)
    print("Number of Reviews:")
    print(review_cnt)
    print("Number of Restaurants with bad reviews:")
    print(len(rstrnt_cntr))


def restrnt_reader(inp):
    dic = {}
    with open('yelp_academic_dataset_business.json', encoding='utf-8') as r:
        for l in r:
            data = json.loads(l)
            if data['categories']:
                if 'Restaurants' in data['categories']:
                    dic[data['business_id']] = data['name']
    return dic


def labl(filename):
    label = {}
    with open(filename) as f:
        for l in f:
            label[l.split(':')[0]] = l.split(':')[1].strip()
    return label


def count_classifier(path):
    file_list = []
    k = os.listdir(path)
    num_list = []
    for t in k:
        if t.endswith('.txt'):
            file_list.append(t)
    for k in file_list:
        with open(path + os.sep + k) as f:
            p = f.read().count('<end>')

            num_list.append(p)

    return num_list, file_list


def tokenize(txt):
    """
    This tokenizing function removes all the mentions,urls and punctuations
    Params:
    txt: A string to be tokenized
    Returns
    A lower case txt with all mentions,urls and punctuations removed
    """
    l = ''
    for m in (txt.split('<end>')[0]).split():
        if (m[0:5] != 'https') and m[0] != '@':
            l = l + m + ' '
    return re.sub('\W+', ' ', l.lower())


def review_list(file_list, num_list, path):
    lis = []
    for k in file_list:
        with open(path + os.sep + k) as f:
            for m in range(min(num_list)):
                lis.append(tokenize(f.readline()))
    return lis


def classify(X_train, Y_train, min_df=1,
             max_df=1., binary=True, tfidf=True, clas='OVR'):
    """Constructs a classifier according to the specifications given
    Params:
    X_train:The training set X values
    Y_train:The training set Y values
    tfidf:If this is true, TfidfTransformer will be used
    clas=if this parameter is OVR then OneVsRestClassifier will be used and if the value is
         LOG then LogisticRegression will be used
    Returns:
    classifier:The constructed classifier will be returned
    """
    if (tfidf == True and clas == 'OVR'):
        classifier = Pipeline([
            ('vectorizer', CountVectorizer(min_df=min_df, max_df=max_df, binary=binary)),
            ('tfidf', TfidfTransformer()),
            ('clf', OneVsRestClassifier(LinearSVC()))])
    if (tfidf == False and clas == 'OVR'):
        classifier = Pipeline([
            ('vectorizer', CountVectorizer(min_df=min_df, max_df=max_df, binary=binary)),
            ('clf', OneVsRestClassifier(LinearSVC()))])
    if (tfidf == True and clas == 'LOG'):
        classifier = Pipeline([
            ('vectorizer', CountVectorizer(min_df=min_df, max_df=max_df, binary=binary)),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression())])
    if (tfidf == False and clas == 'LOG'):
        classifier = Pipeline([
            ('vectorizer', CountVectorizer(min_df=min_df, max_df=max_df, binary=binary)),
            ('clf', LogisticRegression())])
    classifier.fit(X_train, Y_train)
    return classifier


def do_cross_validation(X, y, verbose=False, fld=5, min_df=1,
                        max_df=1., binary=True, tfidf=True, clas='OVR'):
    """
    Perform n-fold cross validation, calling get_clf() to train n
    different classifiers. Use sklearn's KFold class: http://goo.gl/wmyFhi
    Be sure not to shuffle the data, otherwise your output will differ.
    Params:
        X.........a csr_matrix of feature vectors
        y.........the true labels of each document
        n_folds...the number of folds of cross-validation to do
        verbose...If true, report the testing accuracy for each fold.
    Return:
        the average testing accuracy across all folds.
    """
    ###TODO
    ###
    cv = KFold(len(Y_train), fld)
    accuracies = []
    for train_idx, test_idx in cv:
        # print(train_idx,test_idx)
        XT = []
        X = []
        for p in train_idx:
            X.append(X_train[p])
        for p in test_idx:
            XT.append(X_train[p])
        clf = classify(X, Y_train[train_idx], min_df=min_df,
                       max_df=max_df, binary=binary, tfidf=tfidf, clas=clas)
        predicted = clf.predict(XT)
        acc = accuracy_score(Y_train[test_idx], predicted)
        accuracies.append(acc)
    if verbose == True:
        k = 0
        for m in accuracies:
            print('fold %d accuracy=%.4f' % (k, m))
            k = k + 1
    avg = np.mean(accuracies)
    return avg


def get_true_labels(file_list, num_list):
    a = np.empty(len(file_list) * min(num_list), dtype=int)
    n = 0
    for k in file_list:
        for p in range(min(num_list)):
            a[n] = k.split('.txt')[0]
            n = n + 1

    return a


def classifier_details(clas, tfidf):
    if clas == 'OVR':
        print('OneVsRestClassifier')

    else:
        print('LogisticRegression')
    if tfidf == True:
        print('TfidfTransformer enabled')
    else:
        print('TfidfTransformer disabled')


csvf = 'reviews.csv'
rstrnt = restrnt_reader('yelp_academic_dataset_business.json')
print("Total number of Restaurants")
print(len(rstrnt))
if os.path.isfile(csvf):
    print("CSV file already exists")

else:

    Reader('yelp_academic_dataset_review.json', csvf, ['business_id', 'type', 'stars', 'text'], rstrnt.keys())

path = 'data'
lab = labl('label.txt')
num_list, file_list = count_classifier(path)
for m in range(len(num_list)):
    print(lab[file_list[m].split('.txt')[0]] + ' : ' + '%d' % num_list[m])
print('We will be training our classifier using %d sentences from each class' % min(num_list))
ac_dic = {}
Y_train = get_true_labels(file_list, num_list)
X_train = review_list(file_list, num_list, path)
for clas in ('OVR', 'LOG'):
    for tfidf in (True, False):
        classifier_details(clas, tfidf)

        classifier = classify(X_train, Y_train, clas=clas, tfidf=tfidf)
        val = do_cross_validation(X_train, Y_train, tfidf=tfidf, clas=clas)
        print(val)
        ac_dic[do_cross_validation(X_train, Y_train, tfidf=tfidf, clas=clas)] = [tfidf, clas]
mx_ac = max([m for m in ac_dic.keys()])
print("The classifier with the highest accuracy is")
classifier_details(ac_dic[mx_ac][1], ac_dic[mx_ac][0])
print("Lets try different combinations of min_df,max_df,binary in our classifier")
acc = []
mindf = []
maxdf = []
binry = []
for binary in [True, False]:
    for min_df in range(1, 11):
        for max_df in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
            classifier = classify(X_train, Y_train, min_df=min_df, max_df=max_df, binary=binary, clas=ac_dic[mx_ac][1],
                                  tfidf=ac_dic[mx_ac][0])
            acc.append(do_cross_validation(X_train, Y_train, verbose=False, fld=5, min_df=min_df, max_df=max_df,
                                           binary=binary))
            mindf.append(min_df)
            maxdf.append(max_df)
            binry.append(binary)

val = 0
ind = 0
for m in range(len(acc)):
    if val < acc[m]:
        val = acc[m]
        ind = m
print("the highest accuracy is:")
print(val)
print("with min_df,max_df,binary as follows")
print(mindf[ind])
print(maxdf[ind])
print(binry[ind])

ac_classifier = classify(X_train, Y_train, clas=ac_dic[mx_ac][1], tfidf=ac_dic[mx_ac][0], min_df=mindf[ind],
                         max_df=maxdf[ind], binary=binry[ind])

