# coding: utf-8
"""CS585: Assignment 3

In this assignment, you will build a named-entity classifier
using LogisticRegression.

We'll use the labeled data from the CoNLL 2003 Shared Task:
http://www.cnts.ua.ac.be/conll2003/ner/

This is downloaded by the download_data method below.

The main goals of this assignment are to have you:
1- Implement different feature sets for the classifier.
2- Compute evaluation metrics for the classifier on the test set.
3- Enumerate over various settings of the features to determine
   which features result in the highest accuracy.

See Log.txt for the expected output of running the main method.
(Subject to minor variants based on computing environment.)
"""

### DO NOT ADD TO THESE IMPORTS. ####
from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import urllib.request
#####################################


def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/bqitsnhk911ndqs/train.txt?dl=1'
    urllib.request.urlretrieve(url, 'train.txt')
    url = 'https://www.dropbox.com/s/s4gdb9fjex2afxs/test.txt?dl=1'
    urllib.request.urlretrieve(url, 'test.txt')
    
    
def read_data(filename):
    """
    Read the data file into a list of lists of tuples.
    
    Each sentence is a list of tuples.
    Each tuple contains four entries:
    - the token
    - the part of speech
    - the phrase chunking tag
    - the named entity tag
    
    For example, the first two entries in the
    returned result for 'train.txt' are:
    
    > train_data = read_data('train.txt')
    > train_data[:2]
    [[('EU', 'NNP', 'I-NP', 'I-ORG'),
      ('rejects', 'VBZ', 'I-VP', 'O'),
      ('German', 'JJ', 'I-NP', 'I-MISC'),
      ('call', 'NN', 'I-NP', 'O'),
      ('to', 'TO', 'I-VP', 'O'),
      ('boycott', 'VB', 'I-VP', 'O'),
      ('British', 'JJ', 'I-NP', 'I-MISC'),
      ('lamb', 'NN', 'I-NP', 'O'),
      ('.', '.', 'O', 'O')],
     [('Peter', 'NNP', 'I-NP', 'I-PER'), ('Blackburn', 'NNP', 'I-NP', 'I-PER')]]
    """
    ###TODO
    input_file=open(filename,'r')
    lines=input_file.read()
    list_lines=lines.splitlines()
    list_sentence=[]
    list_result=[]
    for token in list_lines:
        list_sentence.append(tuple(token.split()))
        if token=='':
            list_result.append(list_sentence[:-1])
            list_sentence = []
    list_result.append(list_sentence)
    for element in list_result:
        for item in element:
            a,b,c,d=item
            if a=='-DOCSTART-':
                list_result.remove(element)
    return list_result
    pass

def make_feature_dicts(data,
                       token=True,
                       caps=True,
                       pos=True,
                       chunk=True,
                       context=True):
    """
    Create feature dictionaries, one per token. Each entry in the dict consists of a key (a string)
    and a value of 1.
    Also returns a numpy array of NER tags (strings), one per token.

    See a3_test.

    The parameter flags determine which features to compute.
    Params:
    data.......the data returned by read_data
    token......If True, create a feature with key 'tok=X', where X is the *lower case* string for this token.
    caps.......If True, create a feature 'is_caps' that is 1 if this token begins with a capital letter.
               If the token does not begin with a capital letter, do not add the feature.
    pos........If True, add a feature 'pos=X', where X is the part of speech tag for this token.
    chunk......If True, add a feature 'chunk=X', where X is the chunk tag for this token
    context....If True, add features that combine all the features for the previous and subsequent token.
               E.g., if the prior token has features 'is_caps' and 'tok=a', then the features for the
               current token will be augmented with 'prev_is_caps' and 'prev_tok=a'.
               Similarly, if the subsequent token has features 'is_caps', then the features for the
               current token will also include 'next_is_caps'.
    Returns:
    - A list of dicts, one per token, containing the features for that token.
    - A numpy array, one per token, containing the NER tag for that token.
    """
    ###TODO
    list_ner=[]
    dicts=[]
    for elements in data:
        list_ner.append([x[3] for x in elements])
    list_ner_final=[item for sublist in list_ner for item in sublist]
    labels=np.array(list_ner_final)
    for elements in data:
        s_len = len(elements) - 1
        for index,list_tuple in enumerate(elements):
            line_dict = {}
            if token==True:
                line_dict['tok=' + list_tuple[0].lower()] = 1
            if caps==True:
                if list_tuple[0][0].isupper():
                    line_dict['is_caps'] = 1
            if pos==True:
                line_dict['pos=' + list_tuple[1]] = 1
            if chunk==True:
                line_dict['chunk=' + list_tuple[2]] = 1
            if context==True:
                prev=index-1
                next=index+1
                if index > 0 :
                    if token==True:
                        line_dict['prev_tok=' + elements[prev][0].lower()] = 1
                    if pos==True:
                        line_dict['prev_pos=' + elements[prev][1]] = 1
                    if chunk==True:
                        line_dict['prev_chunk=' + elements[prev][2]] = 1
                    if caps == True:
                        if elements[prev][0][0].isupper():
                            line_dict['prev_is_caps'] = 1
                if index < s_len:
                    if token==True:
                        line_dict['next_tok=' + elements[next][0].lower()] = 1
                    if pos==True:
                        line_dict['next_pos=' + elements[next][1]] = 1
                    if chunk==True:
                        line_dict['next_chunk=' + elements[next][2]] = 1
                    if caps==True:
                        if elements[next][0][0].isupper():
                            line_dict['next_is_caps'] = 1
            dicts.append(line_dict)
    return dicts,labels
    pass


def confusion(true_labels, pred_labels):
    """
    Create a confusion matrix, where cell (i,j)
    is the number of tokens with true label i and predicted label j.

    Params:
      true_labels....numpy array of true NER labels, one per token
      pred_labels....numpy array of predicted NER labels, one per token
    Returns:
    A Pandas DataFrame (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)
    See Log.txt for an example.
    """
    ###TODO
    list_true_labels=true_labels.tolist()
    list_pred_labels=pred_labels.tolist()
    list_labels=zip(list_true_labels,list_pred_labels)
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
    """
    Compute precision, recall, f1 for each NER label.
    The table should be sorted in ascending order of label name.
    If the denominator needed for any computation is 0,
    use 0 as the result.  (E.g., replace NaNs with 0s).

    NOTE: you should implement this on your own, not using
          any external libraries (other than Pandas for creating
          the output.)
    Params:
      confusion_matrix...output of confusion function above.
    Returns:
      A Pandas DataFrame. See Log.txt for an example.
    """
    ###TODO
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
    """
    Returns:
    The average F1 score for all NER tags,
    EXCLUDING the O tag.
    """
    ###TODO
    df=evaluation_matrix.drop('O',1)
    avg_f1=sum(df.loc['f1'])/len(list(df))
    return avg_f1
    pass

def evaluate_combinations(train_data, test_data):
    """
    Run 16 different settings of the classifier,
    corresponding to the 16 different assignments to the
    parameters to make_feature_dicts:
    caps, pos, chunk, context
    That is, for one setting, we'll use
    token=True, caps=False, pos=False, chunk=False, context=False
    and for the next setting we'll use
    token=True, caps=False, pos=False, chunk=False, context=True

    For each setting, create the feature vectors for the training
    and testing set, fit a LogisticRegression classifier, and compute
    the average f1 (using the above functions).

    Returns:
    A Pandas DataFrame containing the F1 score for each setting,
    along with the total number of parameters in the resulting
    classifier. This should be sorted in descending order of F1.
    (See Log.txt).

    Note1: You may find itertools.product helpful for iterating over
    combinations.

    Note2: You may find it helpful to read the main method to see
    how to run the full analysis pipeline.
    """
    ###TODO
    def evaluate_combinations_average_f1s(train_data,
                                          test_data,
                                          token=True,
                                          caps=True,
                                          pos=True,
                                          chunk=True,
                                          context=True):

        dicts, labels = make_feature_dicts(train_data,
                                           token=True,
                                           caps=caps,
                                           pos=pos,
                                           chunk=chunk,
                                           context=context)
        vec = DictVectorizer()
        X = vec.fit_transform(dicts)
        n_params=len(vec.get_feature_names())*5
        clf = LogisticRegression()
        clf.fit(X, labels)
        test_dicts, test_labels = make_feature_dicts(test_data,
                                                     token=True,
                                                     caps=caps,
                                                     pos=pos,
                                                     chunk=chunk,
                                                     context=context)
        X_test = vec.transform(test_dicts)
        preds = clf.predict(X_test)
        confusion_matrix = confusion(test_labels, preds)
        evaluation_matrix = evaluate(confusion_matrix)
        return average_f1s(evaluation_matrix),n_params

    features_values = [[True,False],[True,False],[True,False],[True,False]]
    list_features_values = list(product(*features_values))
    df = pd.DataFrame(columns=('f1', 'n_params', 'caps','pos','chunk','context'))
    for i,element in enumerate(list_features_values):
        ca_value, p_value, ch_value, co_value = element
        combinations_average_f1s,n_params=evaluate_combinations_average_f1s(train_data,test_data,token=True,caps=ca_value,pos=p_value,chunk=ch_value,context=co_value)
        df.loc[i]=[combinations_average_f1s,n_params,ca_value,p_value,ch_value,co_value]
        df.n_params = df.n_params.astype(int)
        sorted_df=df.sort_values(['f1'],ascending=[False])
    return sorted_df
    pass

if __name__ == '__main__':
    """
    This method is done for you.
    # See Log.txt for expected output.
    """
    download_data()
    train_data = read_data('train.txt')
    dicts, labels = make_feature_dicts(train_data,
                                   token=True,
                                   caps=True,
                                   pos=True,
                                   chunk=True,
                                   context=True)

    vec = DictVectorizer()
    X = vec.fit_transform(dicts)

    print('training data shape: %s\n' % str(X.shape))
    clf = LogisticRegression()
    clf.fit(X, labels)


    test_data = read_data('test.txt')
    test_dicts, test_labels = make_feature_dicts(test_data,
                                                 token=True,
                                                 caps=True,
                                                 pos=True,
                                                 chunk=True,
                                                 context=True)
    X_test = vec.transform(test_dicts)

    print('testing data shape: %s\n' % str(X_test.shape))

    preds = clf.predict(X_test)

    confusion_matrix = confusion(test_labels, preds)
    print('confusion matrix:\n%s\n' % str(confusion_matrix))

    evaluation_matrix = evaluate(confusion_matrix)
    print('evaluation matrix:\n%s\n' % str(evaluation_matrix))

    print('average f1s: %f\n' % average_f1s(evaluation_matrix))

    combo_results = evaluate_combinations(train_data, test_data)
    print('combination results:\n%s' % str(combo_results))
