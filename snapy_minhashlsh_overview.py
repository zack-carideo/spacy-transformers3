from snapy import MinHash, LSH

"""
## MinHash function Parameters
 - **text:** list of strings containing texts to compare 
 - **n_gram:** size of each overlapping text shingle to break text into prior to hashing 
  - value should be selected based on avg text length 
   - to low shingle size-> false similarities 
   - to high shingle size-> fail to return true positives 
 - **n_gram_type:** type of ngram to use for shingles(char or term) 
  - **char** splits text into character shingles
  - **term** splits text into overlapping sequences of words 

 - **permutations:** number of randomly sampled hash values to use for generating each texts minhash signature. the larger this is , the more accruate jaccard similarities between texts will be , but at cost of efficency
 - **hash_bits:** hash value size to be used to generate minhash signitures from shingles (32,64, or 128 bit). 
  - should be chosen based on text length and a trade off between performance ad accuracy 
  - **lower** hash values risk fals hash collisions leading to false similiarities between docs for larger corpuses
 - **method:** method for randomly sampling via hashing 
  - **multi_hash** texts are hashed once per permutation and the min hash value selected each time to construct signature(STABLE)
  - **k_smallest_values** each text is hashed once and k smallest values selected for k permutations (NOT STABLE) 
 - **seed:** seed from which to generate random hash function, necessary for reproducivibility and to allow updating of the LSH model with new minihash values. 

## LSH Function Parameters 
##### lsh (local sensativity hashing) creates a model of text similarity that can be used to return similar texts based on estimated jaccard similaritiy
 - **minhash:** Minhash object containing minhash signatures return by MinHash Class
 - **labels:** list, array, or series containing unique labels for each text in minhash object signiture. This should be provided in the same order as texts passed to the MinHash class. 
 - **no_of_bands:** number of bands to break minhash signature into before hashing into buckets 
  - **smaller** number of bands will result in a stricter algo (risk of false negatives ) 
  - **larger** risk of false positives

## LSH Methods 
 - **update:** updtes a model from a Minhash object containing signitures generated from new texts and their labels 
  - .update(minhash,new_labels)
 - **query:** takes a label and returns the labels of similar texts (sensativity = # buckets text must share to be returned as similar) 
  - .query(label,min_jaccard=None, sensitivity = 1)
 
 - **remove:** remove file label andminhash signiture from model 
  - .def remove(label):
 
 - **contains:** returns list of labels contained in the model 
  - .contains()
 
 - **adjancency_list:** returns an adjacency list that can be used to create a **text similarity graph**
 
 - **edge_list:** returns a list of edges as tples of similar paris, that can be used to create a **text similarity graph**
  - .edge_list(min_jaccard= None, jaccard_weighted=False, sensativity = 1) 
  
 """

 #test list of strings to run exmaple
content = [
    'Jupiter is primarily composed of hydrogen with a quarter of its mass being helium',
    'Jupiter moving out of the inner Solar System would have allowed the formation of inner planets.',
    'A helium atom has about four times as much mass as a hydrogen atom, so the composition changes when described as the proportion of mass contributed by different atoms.',
    'Jupiter is primarily composed of hydrogen and a quarter of its mass being helium',
    'A helium atom has about four times as much mass as a hydrogen atom and '
    'the composition changes when described as a proportion of mass contributed by different atoms.',
    'Theoretical models indicate that if Jupiter had much more mass than it does at present, it would shrink.',
    'This process causes Jupiter to shrink by about 2 cm each year.',
    'Jupiter is mostly composed of hydrogen with a quarter of its mass being helium',
    'The Great Red Spot is large enough to accommodate Earth within its boundaries.'
]

#set of default lables for each string beng evaluted
labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

#seed to enable replication 
seed = 3

#size of each overlapping text shingle to break text into prior to hashing
n_gram = 9

#number of randomly sampled hash values to use for generating each texts minhash signature (larger = more accurate & slower)
permutations=100

#hash value size to be used to generate minhash signitures from shingles (32,64, or 128 bit). 
#NOTE: should be chosen based on text length and a trade off between performance ad accuracy
hash_bits = 64

# Create MinHash object.
minhash = MinHash(content, n_gram=n_gram, permutations=permutations, hash_bits=hash_bits, seed=seed)

# Create LSH model.
lsh = LSH(minhash, labels, no_of_bands=50)

#query to find near duplicates for text 1 
print(lsh.query(1,min_jaccard=.5))

#update model
#generate minhash aignitures for new text, and add new texts to LSH model 
new_text = [
    'Jupiter is primarily composed of hydrogen with a quarter of its mass being helium',
    'Jupiter moving out of the inner Solar System would have allowed the formation of inner planets.',
]

new_labels =['new_doc1','new_doc2']

#1.create minhash signitues for new text 
new_minhash = MinHash(new_text, n_gram=n_gram, permutations=permutations, hash_bits=hash_bits, seed=seed)

#2.update lsh model with new hash signitures and verify lsh model updates reflected
lsh.update(new_minhash,new_labels)
print(lsh.contains())

#print the adjacency_list of all docs 
print(lsh.adjacency_list())

#print the edge list of all docs that are flagged as duplicates to plot in text similarity graph 
print(lsh.edge_list())

#remove text and label from model (if its not there , you will get an error returned)
lsh.remove(6)
print(lsh.contains())

#get matrix(n*m) of text signatures generated by minhash function (n=text row, m=selected permutations)
minhash.signatures.shape 