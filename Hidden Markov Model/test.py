# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:26:44 2017

@author: Arnab
"""
list_parts1=list()
list_parts2=list()
sentences=list()
tags=list()
fname = 'data.txt'
text_file = open(fname, 'r')
lines = text_file.read()
a=lines.replace(', ,','')
b=a.replace('$ $','')
c=b.replace('; :','')
d="".join([s for s in c.strip().splitlines(True) if s.strip()])

list1=lines.splitlines()


for element in list1: 
    
    if element=='':
        'print(count)'
        
        sentences.append(list_parts1)
        list_parts1=list()
        tags.append(list_parts2)
        list_parts2=list()
    else:
        for item in [element]:
            a=item.split()
            'print(a[0])'
            list_parts1.append(a[0])
            list_parts2.append(a[1])
    
       
#print(tags)
dict2={'D': {'V': 0.0, 'N': 0.75, 'D': 0.25}}
dict1={'a':1}
dict3={'m':4}
dict2.update({'N': {'V': 0.75, 'N': 0.25, 'D': 0.0}})
dict1['b']=2
print(dict2)
dict1.update(dict3)
print(dict1)



 





'''
    parts = element.split(',,')
    print (parts)
'''
'''

for line in final:
    if len(line.strip()) == 0 :
        print('a') 

        list_lines.append(final.splitlines())
'''

    















        
'''
            for key,value in my_dict2.items():
                
                
                if key==state:
                    
                    my_dict3[key]=value
                    my_tuple=tuple([key]+[value])
                    my_final.append(my_dict3)
                    
                    #my_dict4.update({my_dict3.keys():my_dict3.values()})
                    print(state)
                    print("-------")
                    print(key)
                    print(value)
            
                    my_dict3.clear()
        print(my_final)
                    #print(my_dict3)
                    #print(my_final)
            #print(my_dict1)
            #print(my_dict2)
            #print(my_dict3)
            
                #print(my_dict.items())
                
                #for k,v in my_dict.items(): 
                   # res[k].append(v)
        #print(res)
                #my_dict[state][s]=prob
                #my_final.append(my_dict)
                
      
                                    
            #print("Probability of "+state+" = "+prob)                    
                            
   
'''        

'''
        emission_probas = {'D': {'the': 1.0, 'boy': 0.0, 'jumped': 0.0},
								'N': {'the': 0.0, 'boy': 0.4, 'jumped': 0.0},
                                              'V': {'the': 0.0, 'boy': 0.0, 'jumped': 0.5}

								 }
'''

 for state in states:
            for tag in tags:
                index=-1
                for t in tag:
                    index+=1
                    if state == t:
                        word_count=0
                        words = []
                        for sent in sentences:
                            word = sent[index]
                            #print(word)
                            #print(index)
                            if words is None:
                                words.append(word)
                                word_count+=1
                                print(words)
                                print(word_count)
                            else:
                                print(index)
                                word = sent[index]
                                if word in words:
                                    word_count+=1

