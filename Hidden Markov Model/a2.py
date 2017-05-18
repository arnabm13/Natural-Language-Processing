# coding: utf-8
"""CS585: Assignment 2

In this assignment, you will complete an implementation of
a Hidden Markov Model and use it to fit a part-of-speech tagger.
"""

from collections import Counter, defaultdict
import math
import numpy as np
import os.path
import urllib.request


class HMM:
    def __init__(self, smoothing=0):
        """
        Construct an HMM model with add-k smoothing.
        Params:
          smoothing...the add-k smoothing value
        
        This is DONE.
        """
        self.smoothing = smoothing
 
    def fit_transition_probas(self, tags):
        """
        Estimate the HMM state transition probabilities from the provided data.

        
        Creates a new instance variable called `transition_probas` that is a 
        dict from a string ('state') to a dict from string to float. E.g.
        {'N': {'N': .1, 'V': .7, 'D': 2},
         'V': {'N': .3, 'V': .5, 'D': 2},
         ...
        }
        See test_hmm_fit_transition.
        
        Params:
          tags...a list of lists of strings representing the tags for one sentence.
        Returns:
            None
        """
        ###TODO
      
        my_list=[]
        my_final2={}
        my_dict1={}
        my_dict2={}
        my_dict3={}
        my_dict4={}
        
        for element in tags:
            for item in element:
                my_list.append(item)
        
        self.states=list(set(my_list))
        
        for state in self.states:
            for s in self.states:
                above=0
                below=0
                for tag in tags:
                    if state in tag:
                        length = len(tag)
                        for i in range(length):
                            if (state == tag[i] and i<length-1):
                                below +=1
                                t1=tag[i]
                                i+=1
                                t2 = tag[i]
                                tpair = t1+t2
                                spair = state+s
                                if (spair == tpair):
                                    above+=1
                                    i-=1
                                    #print(spair+" = "+tpair)
                total_types=len(self.states)
                num=above+self.smoothing
                denom=below+self.smoothing*total_types
                prob = num/denom
                my_dict1[s]=prob
            #print(state)
            #print(my_dict1)
            #my_dict2[state]=my_dict1
            my_dict2.update({state:my_dict1.copy()})
        self.transition_probas=my_dict2
        #print(my_dict2)
        pass

    def fit_emission_probas(self, sentences, tags):
        """
        Estimate the HMM emission probabilities from the provided data. 

        Creates a new instance variable called `emission_probas` that is a 
        dict from a string ('state') to a dict from string to float. E.g.
        {'N': {'dog': .1, 'cat': .7, 'mouse': 2},
         'V': {'run': .3, 'go': .5, 'jump': 2},
         ...
        }

        Params:
          sentences...a list of lists of strings, representing the tokens in each sentence.
          tags........a list of lists of strings, representing the tags for one sentence.
        Returns:
            None          

        See test_hmm_fit_emission.
        """
        ###TODO
       

        my_list=[]
        for element in tags:
            for item in element:
                my_list.append(item)
        
        count=Counter(my_list)
        #print(count)
        list_tags=[]
        my_dict={}
        dict_final1={}
        dict_final2={}
        list_sentences=[]
        list_final=[]
        for tag in tags:
            for t in tag:
                list_tags.append(t)
        #print(list_tags)
        
        for sentence in sentences:
            for word in sentence:
                list_sentences.append(word)
        #print(list_sentences)
        #dictionary = dict(zip(list_tags,list_sentences))
        #l=[j for i in zip(list_tags,list_sentences) for j in i]
        index=-1
        for element in list_tags:
            index+=1
            list_final.append(tuple([element]+[list_sentences[index]]))
        count1=Counter(list_final)
        #print(count1)
        total_types=len(set(list_sentences))
        #total_types= 7
        #print("total types====",total_types)
        
        my_dict={}
        for state in self.states:
            my_dict[state] = {}
            for word in list_sentences:
                for key, value in count1.items():
                    a, b = key
                    if (a == state) and (b == word):
                        num=value+self.smoothing
                        denom=count[state]+self.smoothing*total_types
                        prob=num/denom
                        my_dict[state][word] = prob
                        break
                    else:
                        num=self.smoothing
                        denom=count[state]+self.smoothing*total_types
                        my_dict[state][word] = num/denom
        self.emission_probas=my_dict
                       
                     
        
                              
        pass            
                
       
    
    def fit_start_probas(self, tags):
        """
        Estimate the HMM start probabilities form the provided data.

        Creates a new instance variable called `start_probas` that is a 
        dict from string (state) to float indicating the probability of that
        state starting a sentence. E.g.:
        {
            'N': .4,
            'D': .5,
            'V': .1        
        }

        Params:
          tags...a list of lists of strings representing the tags for one sentence.
        Returns:
            None

        See test_hmm_fit_start
        """
        ###TODO
      
        my_dict={}
        for state in self.states:
            above=0
            below=0
            for tag in tags:
                if(state==tag[0]):
                    above+=1
                below+=1 
            #print(state)
            #print(above)
            #print(below)
            #self.smoothing=1
            total_types=len(self.states)
            num=above+self.smoothing
            denom=below+self.smoothing*total_types
            prob=num/denom
            #print(prob)
            my_dict[state]=prob
        #print(my_dict)
        self.start_probas=my_dict
        
        pass

    def fit(self, sentences, tags):
        """
        Fit the parameters of this HMM from the provided data.

        Params:
          sentences...a list of lists of strings, representing the tokens in each sentence.
          tags........a list of lists of strings, representing the tags for one sentence.
        Returns:
            None          

        DONE. This just calls the three fit_ methods above.
        """
        self.fit_transition_probas(tags)
        self.fit_emission_probas(sentences, tags)
        self.fit_start_probas(tags)


    def viterbi(self, sentence):
        """
        Perform Viterbi search to identify the most probable set of hidden states for
        the provided input sentence.

        Params:
          sentence...a lists of strings, representing the tokens in a single sentence.

        Returns:
          path....a list of strings indicating the most probable path of POS tags for
                    this sentence.
          proba...a float indicating the probability of this path.
        """
        ###TODO
        pass


def read_labeled_data(filename):
    
    """
    Read in the training data, consisting of sentences and their POS tags.

    Each line has the format:
    <token> <tag>

    New sentences are indicated by a newline. E.g. two sentences may look like this:
    <token1> <tag1>
    <token2> <tag2>

    <token1> <tag1>
    <token2> <tag2>
    ...

    See data.txt for example data.

    Params:
      filename...a string storing the path to the labeled data file.
    Returns:
      sentences...a list of lists of strings, representing the tokens in each sentence.
      tags........a lists of lists of strings, representing the POS tags for each sentence.
    """
    ###TODO
      
    list_parts1=list()
    list_parts2=list()
    sentences=list()
    tags=list()
    fname = 'data.txt'
    text_file = open(fname, 'r')
    lines = text_file.read()
    list1=lines.splitlines()
    for element in list1:
        if element=='':
            sentences.append(list_parts1)
            list_parts1=list()
            tags.append(list_parts2)
            list_parts2=list()
        else:
            for item in [element]:
                a=item.split()
                list_parts1.append(a[0])
                list_parts2.append(a[1])
    return sentences,tags
    pass

def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/ty7cclxiob3ajog/data.txt?dl=1'
    urllib.request.urlretrieve(url, 'data.txt')

if __name__ == '__main__':
    """
    Read the labeled data, fit an HMM, and predict the POS tags for the sentence
    'Look at what happened'

    DONE - please do not modify this method.

    The expected output is below. (Note that the probability may differ slightly due
    to different computing environments.)

    $ python3 a2.py  
    model has 34 states
        ['$', "''", ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', '``']
    predicted parts of speech for the sentence ['Look', 'at', 'what', 'happened']
    (['VB', 'IN', 'WP', 'VBD'], 2.751820088075314e-10)
    """
    fname = 'data.txt'
    if not os.path.isfile(fname):
          download_data()     
    sentences, tags = read_labeled_data(fname)
    'print(sentences)'
    'print(tags)'

    model = HMM(.001)
    model.fit(sentences, tags)

    print('model has %d states' % len(model.states))

    print(model.states)

    sentence = ['Look', 'at', 'what', 'happened']
    print('predicted parts of speech for the sentence %s' % str(sentence))
    print(model.viterbi(sentence))
