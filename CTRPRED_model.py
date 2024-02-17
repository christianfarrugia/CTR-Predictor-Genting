"""
Created on Mon Sep  9 14:34:17 2019

@author: CFarrugia
"""
"""
Email predictor
"""


"""Email Analysis function"""
import pandas as pd
import nltk
import re
#import urllib.request
#import requests
from bs4 import BeautifulSoup
#from collections import Counter
import string
#from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
#from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import EnglishStemmer
#ps = PorterStemmer()
#import stem
#import datefinder
#from nltk import word_tokenize

#import urllib.parse
#import unicodedata
#from nltk.stem import WordNetLemmatizer 
#nltk.download('wordnet')
#import pattern
#from pattern.en import lemma, lexeme
import spacy
from sklearn.feature_extraction.text import TfidfTransformer
import feather
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys, os 

#import unicodedata


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import make_scorer
import xgboost as xgb

#from gensim.models import Word2Vec
import gensim
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#from nltk.tokenize import word_tokenize
import pickle


#the function below is used to make sure that each file referenced in the program is stored in the same directory as the product
def resource_path(relative_path):
    
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

#to clean html tags
def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True


analyzer = CountVectorizer().build_analyzer()


#stemmer
def stemmed_words(replaces5):
          stemmer = EnglishStemmer()
          return (stemmer.stem(w) for w in analyzer(replaces5))
  


    
def text_cleaning(text_to_clean):
        
    
        
        stop = set(stopwords.words('english'))##stopwords
        rex1 =re.compile('\d{2}/\d{2}/\d{4}')#dates such as 28/08/2019
        rex2=re.compile('\d\d:\d\d')#time such as 08:30
        rex3=re.compile('\d{2}/\d{2}/\d{2}')#dates such as 21/08/19
        rex4=re.compile('((\d)(st|nd|rd|th)|(\d{2})(st|nd|rd|th))')#remove 1st/2nd/23rd etc..
        rex5=re.compile('((\d.\d{2})(am|pm|a.m.|a.m|p.m|p.m.)|(\d{2}.\d{2})(am|pm|a.m.|a.m|p.m|p.m.))')#remove any time format such as 8.00 am
        rex6=re.compile('\d/\d')#odds such as 5/1
        rexx=re.compile('\d{2},000x')#10,000x more
        rexthousand=re.compile('\d*,000')#numbers such as 10,000
        #rexpt1=re.compile('\d.\d')
        #rexpt2=re.compile('\d+.00')
        rexpt1=re.compile('\d\.\d')#odds such as 1.7
        rexpt2=re.compile('£\d+\.00|£[0-9]+\.*[0-9]*')#currency amounts
        rextr=re.compile("<tr>\s(.*) </tr>",re.DOTALL)#remove html script between <tr> and </tr>
        rexatr=re.compile("<tr>|</tr>")#remove <tr>|</tr> 
        rextimes1=re.compile("[0-9]+,[0-9]+x")#any number (including a comma) directly followed by an x such as 100,000x
        rextimes2=re.compile("[0-9]+x")#any number (without) a comma followed by an x such as 10x 
                             
        
        soup = BeautifulSoup(text_to_clean,features='lxml')
        soup
        """ The list below contains words and phrases that are to be removed from the text"""
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 'September', 'October', 'November', 'December','january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
        months_abbrevs=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul','Aug', 'Sep', 'Oct', 'Nov', 'Dec','jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul','aug', 'sep', 'oct', 'nov', 'dec']
        char_list = ['BST','GMT','My Preferences','Help', 'Forgot Password','Forgotten Password','MOBILE','TABLET','DESKTOP','GMT','gmt','UK Casinos','Online Casino','Sports Betting', 'My Preferences', 'View Online','IF YOU ARE NOT ABLE TO VIEW THIS EMAIL CLICK HERE','LOGIN','Username:','USERNAME','Username','Username;','Update','|']
        days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        char_list.extend(months)
        char_list.extend(months_abbrevs) 
        char_list.extend(days)
        
        data=soup.findAll(text=True)#extracting text using beautiful soup
        result = filter(visible, data)#filter text using function "visible"
        instring =list(result)#text in list format

        
        str1 = ''.join(instring)#transform text from list format to string 
        replaces = str1.replace("\n", "")#replace character \n with empty space
        """ this section focuses on removing custom phrases with empty space"""
        replaces=replaces.replace("Can't read this email? View online\xa0 Header links Username: Forgotten Password? Help  Main banner   Main offer   Copy block Live Casino"," ")
        replaces=replaces.replace("Can't read this email? View online"," ")####
        replaces=replaces.replace("Header links"," ")#####
        replaces=replaces.replace("View on Web"," ")###
        replaces=replaces.replace("View on web"," ")###
        replaces=replaces.replace("Main banner   Main offer  Play block   Copy block"," ")####
        replaces=replaces.replace("Signoff Logo   Sign off"," ")####
        replaces=replaces.replace("LIVE CASINO"," LIVE CASINO ")####
        replaces=replaces.replace("Cashback Bonus payment date confirmation"," ")
        replaces=replaces.replace("Genting Black Members Only"," ")
        replaces=replaces.replace("(find out how to join here)"," ")
 
        pt1=rexpt1.findall(replaces)#find all decimal odds
        pt2=rexpt2.findall(replaces)#find all currency amounts
        t1=rextimes1.findall(replaces)#find numbers followed by an x (and including comma)
        t2=rextimes2.findall(replaces)#find numbers followed by an x(and not including comma)
        
        replaces=re.sub("|".join(char_list), "",replaces)#replace all words and phrases found in char_list
        replaces=replaces.replace("|","")
        replaces=replaces.split('This email is sent on behalf of')[0]#remove everything that follows 'this email is sent  on behalf of'
        replaces=replaces.replace("*This email is powered by Acteol CRM* ","")#replace text in first part of bracket
        
        replaces=replaces.lower()#decapitalise text
        
        for x in t1:
            replaces=re.sub(x,'  XTIMES  ',replaces)
        
        for y in t2:
            replaces=re.sub(y,'  XTIMES  ',replaces)
    
        
        for p in pt1:
            replaces=re.sub(p,'  ODD  ',replaces)
        
        for pp in pt2:
            replaces=re.sub(pp,'  MONEY  ',replaces)
        
        
        
        """removing terms and conditions"""
        
        re_not_between=re.compile('t&c(.*)(key terms|valid until|valid from)',re.DOTALL)
        matches = re.search(re_not_between, replaces)
       
        if type(matches)==re.Match:
            text_between_t_c=matches.group(1)
            #replaces=re.sub(replaces,replaces + 'OOOOOOOOOOO '+ text_between_t_c,replaces)
            replaces=replaces+text_between_t_c
            #replaces = replaces.replace(replaces, matches.group(1))
       
         #the 5 lines of code above are used to extract text found between two terms and conditions paragraphs,for example:
        #Key terms .......... full t&c's.     Bet now on man united .........   Key terms ........ full t&c's.
        #the code above extracts "Bet now on man united ........."
        
         
        rex = re.compile("key.*general (terms|t&c|t&cs|t&c’s|t& c|t& c’s){0,1}  apply", re.DOTALL)#. means any character,* means 0 or more
        replaces = rex.sub("", replaces)
        
        rexxtr=re.compile("key.*full (terms|t&c|t&cs|t&c’s|t& c|t& c’s)",re.DOTALL)
        replaces=rexxtr.sub("",replaces)
        
        rexterms=re.compile("key.*(t&c|t&cs|t&c’s|t& c|t& c’s)", re.DOTALL)
        replaces=rexterms.sub("",replaces)
        
        rexvalid=re.compile("valid until.*(t&c|t&cs|t&c’s|t& c|t& c’s)",re.DOTALL)
        replaces=rexvalid.sub("",replaces)
        
        rexvalidf=re.compile("valid (from|to).*(t&c|t&cs|t&c’s|t& c|t& c’s)",re.DOTALL)
        replaces=rexvalidf.sub("",replaces)
        
       
        rexvalidb=re.compile("valid between.*(t&c|t&cs|t&c’s|t& c|t& c’s)",re.DOTALL)
        replaces=rexvalidb.sub("",replaces)
       
        rexvalidp=re.compile("promotion runs.*(t&c|t&cs|t&c’s|t& c’s|t& c)",re.DOTALL)
        replaces=rexvalidp.sub("",replaces)

        rexda=re.compile('\d{4}[a-zA-Z]{2,} ')#removing words such as 2019genting
        dtw=rexda.findall(replaces)
        
        rexd=re.compile('\d{2}[a-zA-Z]{2,} ')#removing words such as "10win"
        d1=rexd.findall(replaces)
        
        for dw in dtw:#list of words such as 2019genting
            #replaces=re.sub(dw,dw.replace(dw,' '+dw,1),replaces)
             replaces=re.sub(dw,dw.replace(dw,' '+dw,1),replaces)
            
           
        replaces2=replaces
        replaces2.translate(str.maketrans('', '', string.punctuation))
        
      
        #replaces2=" ".join([i for i in replaces2.split() if i not in stop])#remove stopwords
        
        #search and remove/modify custom words/numbers/dates compiled by regex compilers
        dates1=rex1.findall(replaces2)
        dates2=rex3.findall(replaces2)
        times=rex2.findall(replaces2)
        day_abbrevs=rex4.findall(replaces2)
        day_abbrevs1=[lis[0] for lis in day_abbrevs]
        times_eng=rex5.findall(replaces2)
        times_eng1=[lis[0] for lis in times_eng]
        slash_odds=rex6.findall(replaces2)
        dx=rexx.findall(replaces2)
        th=rexthousand.findall(replaces2)
        pt1=rexpt1.findall(replaces2)
        pt2=rexpt2.findall(replaces2)
        d1=rexd.findall(replaces2)
        
        
        replaces2=re.sub("|".join(dates1),"",replaces2)
        replaces2=re.sub("|".join(dates2),"",replaces2)
        replaces2=re.sub("|".join(times),"",replaces2)
        replaces2=re.sub("|".join(day_abbrevs1),"",replaces2)
        replaces2=re.sub("|".join(times_eng1),"",replaces2)
        
        for odd in slash_odds:#replace 5/1 with ODD
            replaces2=re.sub(odd,' ODD' + " ",replaces2)
        
        for d in dx:
            replaces2=re.sub(d, d.replace(',','',1),replaces2)
            
        for t in th:
            replaces2=re.sub(t,t.replace(',','',1),replaces2)
        
        
        for ddd in d1:
             replaces2=re.sub(ddd,ddd.replace(ddd[2:],' '+ddd[2:],1),replaces2)
             
        rexcup=re.compile('cup')
        cups=rexcup.findall(replaces2)
        
        for cup in cups:
            replaces2=re.sub(cup,cup+' ',replaces2)
        
        
        tr=rextr.findall(replaces2)#removing text between tr tags 
        
        for tt in tr:
            replaces2=re.sub(tt," ",replaces2)

        
        atr=rexatr.findall(replaces2)
        
        
        for at in atr:
            replaces2=re.sub(at," ",replaces2)
           
        #remove custom words
        
        #replaces2=re.sub("|".join(char_list), "",replaces2)
        #replaces2=replaces2.replace("|","")
     
        
        ###remove stopwords
        replaces2=" ".join([i for i in replaces2.split() if i not in stop])
        ###
        
        renumbers=re.compile('[0-9]+')#remove numbers such as "20 times"#1 or more numbers 
        replaces2=renumbers.sub(" ",replaces2)
        
        rexif=re.compile("\[if(.+)\[endif\]",re.DOTALL)#removing text between specific html code  
        replaces2 = rexif.sub(" ANIMATION ", replaces2)
            
            
        #replaces2 = re.sub(r'[^\w\s]',' ',replaces2)  
        replaces2=re.sub('[^a-zA-Z&]+',' ',replaces2)
        replaces2=re.sub(r"\b[a-zA-Z]\b", " ", replaces2)
        replaces2=re.sub('[full]* t&c.*s',' ',replaces2)   
        replaces2=re.sub('mobile desktop tablet',' ',replaces2)
        replaces2=re.sub('apply',' ',replaces2) 
        #find regex code that removs multiple whitespace
        replaces2=re.sub(' & ',' ',replaces2) #removing &
        replaces2=re.sub('&(c|cs|c’s) ',' ',replaces2)
        rewhite=re.compile('\s{2,}')#2 or more whitespaces
        replaces2=rewhite.sub(" ",replaces2)
        replaces2=replaces2.replace("terms conditions"," ")
        replaces2=replaces2.strip()
        
        #extra_chars_to_remove=['desktop','mobile','stphone',' ll ']
        #rexhtml=re.compile("\[(.*)\]",re.DOTALL)#to remove some html script
        #replaces2=rexhtml.sub("",replaces)
        
        return replaces2



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

#path = r'.\Email_source_file.feather'
path=resource_path('Email_source_file.feather')
#feather.write_dataframe(excel_file, path)

excel_file=feather.read_dataframe(path)#Original data set
mylist2 = excel_file['htmlviewonline'].tolist()#list of urls
clicks_sends=excel_file['clicks_sends'].tolist()#list of click through rates
#udc_sends=excel_file['udc_clicks_sends'].tolist()#list of udc/sends (not used)
#vip=excel_file['vip'].tolist() 
product=excel_file['prod'].tolist()#list of products
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


####changing lifecycle list from alphabtic to numeric
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
#global corpus2

#with open(r'.\corpus2_pickle.pkl', 'rb') as pickle_in:
#    corpus2 = pickle.load(pickle_in)

with open(resource_path('corpus2_pickle.pkl'), 'rb') as pickle_in:
    corpus2 = pickle.load(pickle_in)



    
word_count=list()


for c in corpus2:
    word_count.append(len(c.split()) )  
    
    
#path = r'.\TF-IDF_dataset.feather'
path=resource_path('TF-IDF_dataset.feather')
df = feather.read_dataframe(path)



""" Different predictor functions """#rand f on tf-idf
def email_clicks_predictor (text,prod,promotion_type,lifecycle):#random forest on tf-idf with pca
    
    cln=text_cleaning(text)
    
    corpus=list()
    
    for corp in corpus2:#copyig corpus 2
        corpus.append(corp)
    
    corpus.append(cln)#add new text to corpus
    
    #vectorizer = CountVectorizer(tokenizer=my_tokenizer,ngram_range=(1, 3))
    vectorizer = CountVectorizer(tokenizer=my_tokenizer,ngram_range=(1, 4))
    result_1 = vectorizer.fit_transform(corpus).todense()
    cols_1 = vectorizer.get_feature_names()
    res_df_2 = pd.DataFrame(result_1, columns = cols_1)
    
    ##transforming CountVectorizer matrix to Tf-IDF
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(result_1)
    
    tf_idf_vector=tfidf_transformer.transform(result_1)
    
    df_2 = pd.DataFrame(tf_idf_vector.todense(),columns=cols_1)##tf-idf with new row  and more columns(possibly)
    
    data_to_predict =df_2[-1:]#last row (corresponds to the text the user entered)
    #df_2=df_2.drop(df_2.index[outliers])   
    df_2=df_2.drop(df_2.index[-1])#remove test data(row corresponding to text user entered) from training data
    """
    df_2['PRODUCT']=product 
    df_2['WORD COUNT']=word_count
    df_2['PROMOTION TYPE']=promotions
    df_2['LIFECYCLE']=lifecycles  
    """
    #word length,product,promotion type and lifecycle of new email
    l=len(cln.split())#word count
    w_c=[l]#list
    pr=[prod]#product
    lf=[lifecycle]#lifecycle
    pt=[promotion_type]
    
    """
    data_to_predict['WORD']=w_c
    data_to_predict['PRODUCT']=pr
    data_to_predict['LIFECYCLE']=lf
    data_to_predict['PROMOTION TYPE']=pt
    
    """
    
    #df_2=df_2.drop(df_2.tail(1).index,inplace=True)
    X_train=df_2#.drop(df_2.tail(1).index,inplace=True)#declaring training data set
    
    
    
    sc = StandardScaler()
    pca = PCA(n_components=500)
    X_train = sc.fit_transform(X_train)
    X_train = pca.fit_transform(X_train)
    X_train=pd.DataFrame(X_train)
    X_train['WORD COUNT']=word_count
    X_train['PRODUCT']=product 
    X_train['LIFECYCLE']=lifecycles 
    X_train['PROMOTION TYPE']=promotions
     
    
    data_to_predict=sc.transform(data_to_predict)
    data_to_predict=pca.transform(data_to_predict)
    data_to_predict=pd.DataFrame(data_to_predict)
    data_to_predict['WORD COUNT']=w_c
    data_to_predict['PRODUCT']=pr
    data_to_predict['LIFECYCLE']=lf
    data_to_predict['PROMOTION TYPE']=pt
    
    regressor = RandomForestRegressor(n_estimators = 500, max_depth = 10, min_samples_leaf = 20, random_state = 0)#tuned parameters
    regressor.fit(X_train,y)
    #y=clicks_sends
    
    
    clicks_sends_predicted1=regressor.predict(data_to_predict)
    
    return clicks_sends_predicted1






def email_clicks_predictor_xgb (text,prod,promotion_type,lifecycle):
    
    cln=text_cleaning(text)
    
    corpus=list()
    
    for corp in corpus2:
        corpus.append(corp)
    
    corpus.append(cln)#add new text to corpus
    
    vectorizer = CountVectorizer(tokenizer=my_tokenizer,ngram_range=(1, 3))
    result_1 = vectorizer.fit_transform(corpus).todense()
    cols_1 = vectorizer.get_feature_names()
    res_df_2 = pd.DataFrame(result_1, columns = cols_1)
    
    ##transforming CountVectorizer matrix to Tf-IDF
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(result_1)
    
    tf_idf_vector=tfidf_transformer.transform(result_1)
    
    df_2 = pd.DataFrame(tf_idf_vector.todense(),columns=cols_1)##tf-idf with new row  and more columns(possibly)
    
    data_to_predict =df_2[-1:]
    #df_2=df_2.drop(df_2.index[outliers])   
    df_2=df_2.drop(df_2.index[-1])
    """
    df_2['PRODUCT']=product 
    df_2['WORD COUNT']=word_count
    df_2['PROMOTION TYPE']=promotions
    df_2['LIFECYCLE']=lifecycles  
    """
    #word length,product,promotion type and lifecycle of new email
    l=len(cln.split())
    w_c=[l]
    pr=[prod]
    lf=[lifecycle]
    pt=[promotion_type]
    
    """
    data_to_predict['WORD']=w_c
    data_to_predict['PRODUCT']=pr
    data_to_predict['LIFECYCLE']=lf
    data_to_predict['PROMOTION TYPE']=pt
    
    """
    
    #df_2=df_2.drop(df_2.tail(1).index,inplace=True)
    X_train=df_2#.drop(df_2.tail(1).index,inplace=True)
    
    
    
    sc = StandardScaler()
    pca = PCA(n_components=500)
    X_train = sc.fit_transform(X_train)
    X_train = pca.fit_transform(X_train)
    X_train=pd.DataFrame(X_train)
    X_train['PRODUCT']=product 
    X_train['WORD COUNT']=word_count
    X_train['PROMOTION TYPE']=promotions
    X_train['LIFECYCLE']=lifecycles  
    
  
    data_to_predict=sc.transform(data_to_predict)
    data_to_predict=pca.transform(data_to_predict)
    data_to_predict=pd.DataFrame(data_to_predict)
    data_to_predict['PRODUCT']=pr
    data_to_predict['WORD COUNT']=w_c
    data_to_predict['PROMOTION TYPE']=pt
    data_to_predict['LIFECYCLE']=lf

    
    regressor=xgb.XGBRegressor(random_state=0,colsample_bytree=0.7,learning_rate=0.01,max_depth=7,min_child_weight=4,n_estimators=250,nthread=4,objective='reg:linear',silent=1,subsample=0.8)


    regressor.fit(X_train, y)

    
    
    clicks_sends_predicted2=regressor.predict(data_to_predict)
    
    return clicks_sends_predicted2



def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in model.vocab]
    return np.mean(model[doc], axis=0)

# Our earlier preprocessing was done when we were dealing only with word vectors
# Here, we need each document to remain a document 
def preprocess(text):
    #text = text.lower()
    doc = word_tokenize(text)
    #doc = [word for word in doc if word not in stop_words]
    #doc = [word for word in doc if word.isalpha()] 
    return doc

# Function that will help us drop documents that have no word vectors in word2vec
def has_vector_representation(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)

# Filter out documents
def filter_docs(corpus, texts, condition_on_doc):
    """
    Filter corpus and texts given the function condition_on_doc which takes a doc. The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    corpus = [doc for doc in corpus if condition_on_doc(doc)]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts)


#mod_path=r'.\GoogleNews-vectors-negative300.bin'
mod_path=resource_path('GoogleNews-vectors-negative300.bin')
model = gensim.models.KeyedVectors.load_word2vec_format(mod_path, binary=True)  
#model.vector_size

#this  model uses a random forest regressor on a Word 2 vec dataset    
def email_clicks_predictor_randf_w2v(text):
    
    #model = gensim.models.KeyedVectors.load_word2vec_format(r'C:\Users\CFarrugia\Downloads\GoogleNews-vectors-negative300.bin.gz', binary=True)  
    #mod_path=r'.\GoogleNews-vectors-negative300.bin'
    #model = gensim.models.KeyedVectors.load_word2vec_format(mod_path, binary=True)  
    #model.vector_size
    #bet=model['casino']
    
    corpus2_list=list()
    # Create a list of strings, one for each email
    for corp in corpus2:
        corpus2_list.append(corp)
    
    corpus_new=list()
    
    for corp in corpus2:
        corpus_new.append(corp)
    
    
    # Collapse the list of strings into a single long string for processing
    big_title_string = ' '.join(corpus2_list)
    
    
    
    # Tokenize the string into words
    tokens = word_tokenize(big_title_string)
    
    # Filter the list of vectors to include only those that Word2Vec has a vector for
    """stores vector for each word found in corpus and in model """
    
    vector_list = [model[word] for word in tokens if word in model.vocab]#stores vector for each word found in corpus and in model
    model.vocab
    # Create a list of the words corresponding to these vectors
    words_filtered = [word for word in tokens if word in model.vocab]#"""each word found in corpus and in model """
    
    # Zip the words together with their vector representations
    word_vec_zip = zip(words_filtered, vector_list)
    
    # Cast to a dict so we can turn it into a DataFrame
    word_vec_dict = dict(word_vec_zip)#""" each word together with its vector representation """
    df_w2v = pd.DataFrame.from_dict(word_vec_dict, orient='index')#"""each word and its vector representation  """
    
    corpus_new, corpus2_list = filter_docs(corpus2, corpus2_list, lambda doc: has_vector_representation(model, doc))


    x = []
    for doc in corpus_new: # append the vector for each document
        x.append(document_vector(model, doc))
    X = np.array(x) # list to array
    """ X contains the 300 dimensional DOCUMENT vector for each email in the corpus. X has the email index on the left and the columns form a 300 dimensional vector """ 

    X_train=pd.DataFrame(X)
    
    #######turning text user inputted into a vector
    #text='Welcome to Genting. Bet on football now to win £10 and free spins'
    text=text_cleaning(text)
    text_list=[text]
    text=[text]
    #text=text.tolist()
    
    big_string=' '.join(text_list)

    tokens_text=word_tokenize(big_string)
    vector_list_text=[model[word] for word in tokens_text if word in model.vocab]
    words_filtered_text=[word for word in tokens_text if word in model.vocab]
    
    
    
    word_vec_zip_text=zip(words_filtered_text,vector_list_text)
    
    word_vec_dict_text=dict(word_vec_zip_text)
    df_w2v_text=pd.DataFrame.from_dict(word_vec_dict_text, orient='index')
    
    
    text,text_list=filter_docs(text,text_list,lambda doc:has_vector_representation(model,doc))
    
    x1=[]
    
    for doc1 in text:
        x1.append(document_vector(model,doc1))
        
    data_to_predict=np.array(x1)
    
    
    regressor = RandomForestRegressor(n_estimators = 700, max_depth = 3, min_samples_leaf = 10, random_state = 0)
    regressor.fit(X_train,y)
    
     
    clicks_sends_predicted3=regressor.predict(data_to_predict)
    
    return clicks_sends_predicted3


"""
GUI
"""
import time
from threading import Thread
import wx
#from wx.lib.pubsub import setuparg1
#from wx.lib.pubsub import pub as Publisher
import pyperclip
#from pubsub import publisher

#import pubsub as Publisher
#from flask import Flask
#from multiprocessing import Process

app=wx.App()

class TestThread(Thread):
    """Test Worker Thread Class."""
 
    #----------------------------------------------------------------------
    def __init__(self):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.start()    # start the thread
 
    #----------------------------------------------------------------------
    def run(self):
        """Run Worker Thread."""
        # This is the code executing in the new thread.
        #  this function is used to make the program not go into "not responding mode"
        #this function needs some tuning
        for i in range(60):
            time.sleep(1)
            wx.CallAfter(self.postTime, i)
        #time.sleep(1)
        #wx.CallAfter(Publisher.sendMessage, "update", "Prediction done!")
        #wx.CallAfter(Publisher.sendMessage, "update", "Prediction done!")
 
    #----------------------------------------------------------------------
    #def postTime(self, amt):
        #"""
        #Send time to GUI
        #"""
        #amtOfTime = (amt + 1) * 10
        #Publisher.sendMessage("update", amtOfTime)
 

class MyFrame(wx.Frame):    
    
    def __init__(self):
        
        
        super().__init__(parent=None, title='CTR predictor')
        
        self.panel = wx.Panel(self)    
        self.instruction1=wx.StaticText(self.panel,label='First choose the correct promotion type, product and lifecycle of the email ',pos=(0,40))
        self.instruction2=wx.StaticText(self.panel,label='Now please enter the content of the email and press the "Predict" button to predict its click through rate',pos=(775,40))
        self.result = wx.StaticText(self.panel, label="",pos=(910,415))
        self.prom_choice_label=wx.StaticText(self.panel,label='Promotion type',pos=(8,80))
        self.prom_label=wx.StaticText(self.panel,label="",pos=(0,110))
        self.prod_choice_label=wx.StaticText(self.panel,label='Product',pos=(130,80))
        self.prod_label=wx.StaticText(self.panel,label="",pos=(130,110))
        self.life_choice_label=wx.StaticText(self.panel,label='Lifecycle',pos=(230,80))
        self.life_label=wx.StaticText(self.panel,label="",pos=(230,110))
        self.mod_label=wx.StaticText(self.panel,label="Prediction model",pos=(0,130))
        
        self.displayLbl = wx.StaticText(self.panel, label="",pos=(780,350))
        self.processLbl=wx.StaticText(self.panel,label="",pos=(920,400))
        
        my_sizer = wx.BoxSizer(wx.VERTICAL)        
        self.text_ctrl = wx.TextCtrl(self.panel,size=(1000,200),style=wx.TE_MULTILINE|wx.VSCROLL,pos=(500,100))
        
        #my_sizer.Add(self.text_ctrl, 1, wx.ALL | wx.EXPAND, 15)        
        my_btn = wx.Button(self.panel, label='Predict',pos=(750,100))
        my_btn.Bind(wx.EVT_BUTTON, self.on_press)
        my_sizer.Add(my_btn, 0, wx.ALL | wx.CENTER, 300) 
        
        self.prom_choice = wx.Choice(self.panel,choices = distinct_promotions,name='Promotion type',pos=(0,95))
        #self.prom_choice.SetStringSelection(string='Promotion') 
        self.prom_choice.Bind(wx.EVT_CHOICE, self.OnChoice_prom)
        self.prod_choice=wx.Choice(self.panel,choices=distinct_products,name='Product',pos=(130,95))
        self.prod_choice.Bind(wx.EVT_CHOICE, self.OnChoice_prod)
        self.life_choice=wx.Choice(self.panel,choices=distinct_lifecycles,id=-1,name='Lifecycle',pos=(230,95))
        self.life_choice.Bind(wx.EVT_CHOICE, self.OnChoice_life)
        #self.png = wx.Image(r'\\sbirstafil001\users\CFarrugia\CTR_hist.png', wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        #wx.StaticBitmap(self.panel, -1, self.png, (650, 500), (self.png.GetWidth(), self.png.GetHeight()))
        models=['Model 1', 'Model 2','Model 3']
        self.mod_choice=wx.Choice(self.panel,choices =models,name='Model',pos=(0,150))
        self.mod_choice.Bind(wx.EVT_CHOICE,self.OnChoice_mod)
        
        self.index=0
        
        self.email_ctr_dict={}
        self.list_ctrl = wx.ListCtrl(
        self.panel, size=(800,450),pos=(600, 500), 
        style=wx.LC_REPORT | wx.BORDER_SUNKEN
        )
        self.list_ctrl.InsertColumn(0, 'Content', width=700)
        self.list_ctrl.InsertColumn(1, 'CTR', width=100)
        self.list_ctrl.Bind(wx.EVT_RIGHT_UP, self.ShowPopup)

        #Publisher.subscribe(self.updateDisplay, "update")
        
        self.SetBackgroundColour((100, 179, 179))
        self.panel.SetSizer(my_sizer)        
        self.Show()
        #wx.MessageBox('You need to insert promotion type, product and lifecycle of the email', 'Warning',wx.OK | wx.ICON_WARNING)
        #prm_ind=self.prom_choice.GetSelection()

    
    def OnChoice_prom(self,event): 
      #prm_ind=self.prom_choice.GetSelection()
      self.prm=self.prom_choice.GetString(self.prom_choice.GetSelection())
      self.mapped_prm=integerMapping[self.prm]
      #self.prom_label.SetLabel("You selected "+ prm +" from Choice")  

    def OnChoice_prod(self,event): 
      
      self.prd=self.prod_choice.GetString(self.prod_choice.GetSelection())  
      self.mapped_prd=int_mapping_prod[self.prd]
      #self.prod_label.SetLabel("You selected "+ prd +" from Choice")
      
    def OnChoice_life(self,event):
        
      self.lcy=self.life_choice.GetString(self.life_choice.GetSelection())
      self.mapped_life=int_mapping_life[self.lcy]
      #self.life_label.SetLabel("You selected "+lcy+" from Choice")      
     
    def OnChoice_mod(self,event):
        self.modl=self.mod_choice.GetString(self.mod_choice.GetSelection())
            
    def ShowPopup(self, event):
        menu = wx.Menu()
        menu.Append(1, "Copy selected items")
        menu.Bind(wx.EVT_MENU, self.CopyItems, id=1)
        self.PopupMenu(menu)

    

    def CopyItems(self, event):
    
        listSelectedLines =[]
        index = self.list_ctrl.GetFirstSelected()  
    
        while index is not -1:
            listSelectedLines.append(self.list_ctrl.GetItem(index, 0).GetText())
            index = self.list_ctrl.GetNextSelected(index)             
    
        pyperclip.copy(''.join(listSelectedLines))
        
        
    def on_press(self, event):
        
        
        self.value = self.text_ctrl.GetValue()
        #self.list_ctrl.InsertStringItem(self.index, self.text_ctrl.GetValue() )
        #self.list_ctrl.SetStringItem(self.index, 1, "2010")
       
        #self.index += 1
        
        if not self.value:
            wx.MessageBox("You didn't input any text", 'Warning',wx.OK | wx.ICON_WARNING)
            #self.result.SetLabel("You didn't enter anything")
            #self.quote = wx.StaticText(self.panel, label="Prediction:",pos=(910,400))
            
        elif self.prom_choice.GetSelection()==-1 or self.prod_choice.GetSelection()==-1 or self.life_choice.GetSelection()==-1 or self.mod_choice.GetSelection()==-1: #or prd or prm:
            wx.MessageBox('You need to insert promotion type, product and lifecycle of the email, as well as which prediction model you want to use', 'Warning',wx.OK | wx.ICON_WARNING)
         
        elif self.modl =='Model 1':
            
            self.processLbl.SetLabel("Predicting..")
            TestThread()
            self.prediction=email_clicks_predictor(self.value,self.mapped_prd,self.mapped_prm,self.mapped_life)
            
            self.processLbl.SetLabel("Prediction done!")
            #self.result.SetLabel('The predicted click rate for the email you entered is'+ prediction)
            self.list_ctrl.InsertStringItem(self.index,self.value)
            #self.list_ctrl.SetStringItem(self.index, 1, self.prediction)
            self.list_ctrl.SetStringItem(self.index, 1,str(self.prediction[0]))
            #wx.MessageBox('Click rate:' + self.prediction[0], 'Warning',wx.OK | wx.ICON_WARNING)
         
            #self.index += 1
            #self.result.SetLabel('The predicted click rate for the email you entered is')
        
        elif self.modl == 'Model 2':
            
            self.processLbl.SetLabel("Predicting..")
            TestThread()
            self.prediction=email_clicks_predictor_xgb(self.value,self.mapped_prd,self.mapped_prm,self.mapped_life)
            self.processLbl.SetLabel("Prediction done!")
            #self.result.SetLabel('The predicted click rate for the email you entered is'+ prediction)
            self.list_ctrl.InsertStringItem(self.index,self.value)
            #self.list_ctrl.SetStringItem(self.index, 1, self.prediction)
            self.list_ctrl.SetStringItem(self.index, 1,str(self.prediction[0]))
        
        elif self.modl == 'Model 3':
            self.processLbl.SetLabel("Predicting..")
            TestThread()
            self.prediction=email_clicks_predictor_randf_w2v(self.value)
            self.processLbl.SetLabel("Prediction done!")
            #self.result.SetLabel('The predicted click rate for the email you entered is'+ prediction)
            self.list_ctrl.InsertStringItem(self.index,self.value)
            #self.list_ctrl.SetStringItem(self.index, 1, self.prediction)
            self.list_ctrl.SetStringItem(self.index, 1,str(self.prediction[0]))
        
        #self.my_btn.Disable()
        
"""   
    def updateDisplay(self, msg):
            
            #Receives data from thread and updates the display
            
            t = msg.data
            if isinstance(t, int):
                self.displayLbl.SetLabel("")
                #self.displayLbl.SetLabel("Prediction done in %s seconds" % t)
            else:
                self.displayLbl.SetLabel("")
                #self.displayLbl.SetLabel("%s" % t)
                #my_btn.Enable()
"""
                
if __name__ == '__main__':
    #app = wx.App()
    frame = MyFrame()
    app.MainLoop()

del app    




