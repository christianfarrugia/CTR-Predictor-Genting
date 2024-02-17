import pandas as pd
import urllib.request
import urllib.parse
import re
import requests
from sklearn.feature_extraction.text import CountVectorizer
import unicodedata
import nltk
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
import pattern
from pattern.en import lemma, lexeme
import spacy
from sklearn.feature_extraction.text import TfidfTransformer
import feather

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import utils
import math
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score##
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

spacy.load('en')

lemmatizer = spacy.lang.en.English()



# tokenize the doc and lemmatize its tokens
def my_tokenizer(doc):
    tokens = lemmatizer(doc)
    return([token.lemma_ for token in tokens])
    
def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values
    from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    '''
    res = {}
    for cl in le.classes_:
        res.update({cl:le.transform([cl])[0]})

    return res


#excel_file = pd.read_excel(r'\\sbirstafil001\users\CFarrugia\email_clicks_udc_project_this.xlsm', sheet_name=1) # can also index sheet by name or fetch all sheets


"""Insert file name HERE"""
path = r'.\file name.xlsm'
excel_file=pd.read_excel(path,sheet_name=1)

#run the below line if you want to save as feather file
#feather.write_dataframe(excel_file, path)

excel_file=feather.read_dataframe(path)#to read feather file
mylist2 = excel_file['htmlviewonline'].tolist()#list of urls
clicks_sends=excel_file['clicks_sends'].tolist()#list of CTR's
y=clicks_sends
udc_sends=excel_file['udc_clicks_sends'].tolist()
#vip=excel_file['vip'].tolist() 
product=excel_file['prod'].tolist()
promotions=excel_file['prom_type'].tolist()
lifecycles=excel_file['life_cycle'].tolist() 

distinct_products=['Sportsbook','Casino','Other']
distinct_promotions=list(set(promotions))
distinct_lifecycles=list(set(lifecycles))

y=clicks_sends
####changing promotion list from alphabetic to numeric
le = preprocessing.LabelEncoder()
le.fit(promotions)
diff_proms=le.classes_
integerMapping = get_integer_mapping(le)
prom_mappings=list()
#integerMapping = get_integer_mapping(le)#mapping from type of promotion to number

for dp in diff_proms:
    prom_mappings.append(integerMapping[dp])    
    

#list(le.classes_)
promotions=le.transform(promotions)    
promotions=promotions.tolist()
####


####changing lifecycle list from alphsbtic to numeric
le1=preprocessing.LabelEncoder()
le1.fit(lifecycles)
diff_lifec=le1.classes_
int_mapping_life=get_integer_mapping(le1)
life_mappings=list()
int_mapping_life=get_integer_mapping(le1)

for l in diff_lifec:
    life_mappings.append(int_mapping_life[l])

#list(le1.classes_)
lifecycles=le1.transform(lifecycles)
lifecycles=lifecycles.tolist()
###

le2=preprocessing.LabelEncoder()
le2.fit(product)
diff_prod=le2.classes_
int_mapping_prod=get_integer_mapping(le2)
prod_mappings=list()
int_mapping_prod=get_integer_mapping(le2)

for l in diff_prod:
    prod_mappings.append(int_mapping_prod[l])

#list(le2.classes_)
product=le2.transform(product)
product=product.tolist()
####



#to obtain new corpus
corpus2=list()
for j in mylist2:          
    url = mylist2[mylist2.index(j)]  
    url_txt= urllib.request.urlopen(url)
    text_to_analyse=url_txt.read()
    replaces5= text_cleaning(text_to_analyse)
    corpus2.append(replaces5)        
    #analyzer = CountVectorizer().build_analyzer()


    
word_count=list()


for c in corpus2:
    word_count.append(len(c.split()) )    

"""    
word_count

outliers=[643] #remove any outliers such as high click through rate

for o in sorted(outliers,reverse=True):
    #del word_count[o]
    del product[o]
    del clicks_sends[o]
    del promotions[o]
    del lifecycles[o]
    #del corpus2[o]
    del mylist2[o]
    #del corpus2_list[o]
"""    


import pickle
#to save new corpus as pickle file
#with open('corpus2_pickle.pkl', 'wb') as pickle_out:
#   pickle.dump(corpus2, pickle_out)

#to extract corpus    
with open(r'.\corpus2_pickle.pkl', 'rb') as pickle_in:
    corpus2 = pickle.load(pickle_in)


vectorizer = CountVectorizer(tokenizer=my_tokenizer,ngram_range=(1, 4))
result = vectorizer.fit_transform(corpus2).todense()
cols = vectorizer.get_feature_names()
res_df3 = pd.DataFrame(result, columns = cols)

##transforming CountVectorizer matrix to Tf-IDF
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(result)

tf_idf_vector=tfidf_transformer.transform(result)


df = pd.DataFrame(tf_idf_vector.todense(),columns=cols)#TF-IDF matrix
##df is the new TF-IDF matrix

path = r'.\TF-IDF_dataset.feather'
feather.write_dataframe(df, path)

#TO READ DF RUN "df = feather.read_dataframe(path)"


path = r'\\sbirstafil001\users\CFarrugia\Count_Vectorizer_dataset.feather'
feather.write_dataframe(res_df3, path)

res_df2 = feather.read_dataframe(path)#Count Vectorizer matrix

#df_up = pd.DataFrame(tf_idf_vector.todense(),columns=cols) 
#analyzer = CountVectorizer().build_analyzer()
#tokenizer=CountVectorizer().tokenizer()
#vectorizer = CountVectorizer(analyzer=stemmed_words,ngrams =range(1,3))
#vectorizer = CountVectorizer(analyzer=stemmed_words)



#import gensim
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument


"""  NOT USED
tagged_corpus2 = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(corpus2)]

corpus2_tokens=list()

for cor in corpus2:
    corpus2_tokens.append(cor.split())
    
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_corpus2)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")



model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
test_data = word_tokenize("I love chatbots".lower())
v1 = model.infer_vector(corpus2_tokens)
print("V1_infer", v1)
model['bet']

model[corpus2[0]]
"""




