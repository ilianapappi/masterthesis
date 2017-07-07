import numpy as np
import pandas as pd
import ast
import collections
from sklearn.feature_extraction.text import CountVectorizer

DIR1 = '/home/11394153/bagofwords/aftermanualprepro/'
dir2 = '/home/11394153/bagofwords/bow_results/'
vocabulary = []
vocabulary_cl = []


def bag_of_words(docs):
    
    vectorizer = CountVectorizer(analyzer='word',
                                vocabulary=voc) #custom vocabulary
    bow_repr = vectorizer.fit_transform(docs)
    
    
    return bow_repr.toarray() 





#make the vocabulary
with open(DIR1+'z.txt') as f:
    content = f.readlines()

for efn in content:

    input_filename = efn.strip('\n')
    if input_filename == 'wendys2_clean.xlsx':
        column_name = 'imageCaption'
    else:
        column_name = 'title'
    

    try:
        df_words = pd.read_excel(DIR1+input_filename,index_col=0)[column_name]
        print ('t',input_filename)
    except:
        print ('e',input_filename)


  
    for i in df_words.index:
        df_words.loc[i] = ast.literal_eval(df_words.loc[i])


        
        for word in df_words.loc[i]:
            vocabulary.append(word)

vocabulary_cl = sorted(set(vocabulary))

print len(vocabulary_cl)
print len(vocabulary)

voc = {}

for i, i_el in enumerate(vocabulary_cl):
    voc[i_el] = i

    

with open(DIR1+'z.txt') as f:
    content = f.readlines()

for efn in content:

    input_filename = efn.strip('\n')
    output_filename = input_filename.split('.')[0]+'_bow.csv'

    if input_filename == 'wendys2_clean.xlsx':
        column_name = 'imageCaption'
    else:
        column_name = 'title'

    df_words = pd.read_excel(DIR1+input_filename)[column_name]
    bow_array = np.zeros((df_words.shape[0],len(voc.keys())))
    
    for i in df_words.index:

        if not ast.literal_eval(df_words.loc[i]): #in case the list is empty!
            print(input_filename,i)

        else:
            df_words.loc[i] = ' '.join(ast.literal_eval(df_words.loc[i])) 
            
            bow_array[i,:] = bag_of_words([df_words.loc[i]])
            
    print (bow_array.shape, input_filename)
    np.savetxt(dir2+output_filename,bow_array,delimiter=",")