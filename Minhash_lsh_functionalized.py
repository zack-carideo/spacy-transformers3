from snapy import MinHash, LSH
import pandas as pd
import string 
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from text_processing import text_preprocessing as  textpp
import matplotlib.pyplot as plt
pd.set_option('max_colwidth', -1)

PUNC = set(string.punctuation)
STOPWORDS = stopwords.words('english')
stemmer = SnowballStemmer('english')

#GET THIS OUT OF THIS CLASS IN THE AM!!!!
#remove short titles (minhash breaks if ngram < minwords in title)
#remove stop(only becuase we are truncating string length to be 100, so we want the 100 most relevant terms from title (if it really gets that long))
#preprocess after clean becuase clean will influence min_len of string otput
#return: preprocessed text field, original text field, and index to map back to original input dataset 
def preprocess_data(data,col2check,min_str_len = min_str_len, max_str_len = max_str_len,stopwords=None, lower=False):

    #clean text 
    Data[col2check] = data[col2check].apply(lambda title: textpp.clean_sentence(title,lower=lower,stopwords=stopwords))
    Data = Data[~(Data[col2check].isna()) & (Data[col2check].str.len()>min_str_len)].reset_index().copy()

    #truncate text to ensure consistent string length comparisons 
    Data[col2check] = Data[col2check].apply(lambda x: x[:max_str_len])

    #create label set to use in lsh model 
    labels = [i for i in Data.index]

    return Data, docs, labels


#build minhash signitures 
def build_minhash(self,docs, n_gram = n_gram, permutations = permutations, hash_bits = hash_bits , seed = seed):
    self.minhash = MinHash(docs, n_gram=n_gram, permutations=permutations, hash_bits = hash_bits, seed=seed)

#get lsh model to query 
def build_lsh(self, labels, no_of_bands = self.permutations/2):
    self.lsh = LSH(self.minhash, labels, no_of_bands=no_of_bands)

#add new text to lsh model 
def addnew_text(self, new_text, new_label):

    self.new_text = new_text
    self.new_label = new_label
    
    #create minhash sigs for new text
    new_minhash = MinHash([new_text] , n_gram = self.n_gram , permutations = self.permutations, hash_bits = self.hash_bits, seed = self.seed)
    
    #update lsh model with new hash sigs
    self.lsh.update(new_minhash, [new_label])
    
#query updated lsh model for new string input 
def query_something_new(self):
    return self.lsh.query(self.new_label,min_jaccard=.4)



#user input string to add to model 
user_search = 'amazon s3 outage causes ripples'

#data specific paramters 
data_path = "C:\\Users\\zjc10\\Desktop\\Projects\\data\\news\\webhose_news\\webhose_df.pickle"
col2check = 'title'

#minhash & lsh specific parameters
#seed to enable replication 
seed = 3

#size of each overlapping text shingle to break text into prior to hashing
#set to low -> more fps
#set to high -> more fns
n_gram = 10

#number of randomly sampled hash values to use for generating each texts minhash signature (larger = more accurate & slower)
permutations=100

#hash value size to be used to generate minhash signitures from shingles (32,64, or 128 bit). 
#NOTE: should be chosen based on text length and a trade off between performance ad accuracy
hash_bits = 64

#max characters in each string to evaluate 
max_str_len = 100
min_str_len = 20

#clean source data/existing aritcles 
data = pd.read_pickle(data_path)
Data = data[~(data[col2check].isna()) & (data[col2check].str.len()>20)].copy()
Data[col2check] = Data[col2check].apply(lambda title: textpp.clean_sentence(title,lower=False,stopwords=set(STOPWORDS)))
Data = Data[~(Data[col2check].isna()) & (Data[col2check].str.len()>20)].copy()
dtest = Data.head(5000).reset_index()

#determine statistics of title length 
print(
'min',min([len(x) for x in dtest[col2check]]),
'max',max([len(x) for x in dtest[col2check]]),
'avg',np.mean([len(x) for x in dtest[col2check]]))
plt.plot([len(x.split()) for x in dtest[col2check]])

#truncate text to ensure consistent string length comparisons 
dtest[col2check] = dtest[col2check].apply(lambda x: x[:max_str_len])

#docs to process 
docs = dtest[col2check].copy()

#create default set of labels (index of the input data series)
#labels = [i for i in range(1,len(docs)+1)]
labels = [i for i in docs.index]

#create minhash model 
minhash = MinHash(docs, n_gram=n_gram, permutations=permutations, hash_bits = hash_bits, seed=seed)

#create lsh model 
#no_of_bands: higher results in more fps (default #permutations/2)
lsh = LSH(minhash, labels, no_of_bands=permutations/2)

#get doc edge list 
print([item for item in lsh.adjacency_list().items() if item[1] !=[]][:10]) #adjacency list for text similarity graph
print([item for item in lsh.edge_list()][:10]) #edge list for tex similarity graph 

#query doc for similar docs 
print(lsh.query(1,min_jaccard=.4))

#view similar articles 
#NOTICE: there are a lot of subgroups that contain overlapping 'similar articles' (i.e 1{2,3}, 3{4,5}) . in this example we can actually link 3 to both 1,2,4 and 5!
#print(dtest.loc[[1,5,16,33,71,81],'title'])
[dtest.loc[[values[0]]+values[1],'title'] for values in lsh.adjacency_list().items() if values[1]!=[]][:9]


#CREATE AND CLUSTER GRAPH OBJECT 
from hcs_clustering.hcs import labelled_HCS
import numpy as np 
import networkx as nx

#1. Get edge list from lsh 
edges = lsh.edge_list(min_jaccard=.5, jaccard_weighted=True)

#2.create a graph object by passing a list of weighted edges to an empty graph objct 
FG = nx.Graph()
FG.add_weighted_edges_from(edges)

#Analyzing Graph
for n, nbrs in FG.adj.items():
    for nbr, eattr in nbrs.items():
        wt = eattr['weight']
        if wt <= 0.5: print('(%d, %d, %.3f)' % (n, nbr, wt))
            
list(nx.connected_components(FG))
sorted(d for n, d in FG.degree())
nx.clustering(FG)

import matplotlib.pyplot as plt
nx.draw(FG,with_labels=False)