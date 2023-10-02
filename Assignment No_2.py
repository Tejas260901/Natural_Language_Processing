#Assignment no 2
#Name: Sonawane Tejas Abhay
#Roll no: 60
#Batch: B3
#Assignment name:Implementation of Bag of words to find the frequency of the tokens & Creating Tf-IDF.

#---- Creating Bag-of_Words-------

import gensim
from gensim.utils import simple_preprocess
from gensim import corpora

#text2 = open('sample_text.txt', encoding='utf-8')

text2 = ["""Hello, how are you?", "How do you do?", 
   "Hey what are you doing? yes you What are you doing?"""]

tokens2 = []
# for line in text2.read().split('.'):
for line in text2[0].split('.'):
    tokens2.append(simple_preprocess(line, deacc=True))

g_dict2 = corpora.Dictionary(tokens2)

print("The dictionary has: " + str(len(g_dict2)) + " tokens")
print(g_dict2.token2id)
print("\n")

g_bow =[g_dict2.doc2bow(token, allow_update = True) for token in tokens2]
print("Bag of Words : ", g_bow)


#OUTPUT:
#[[(0, 1), (1, 1), (2, 1), (3, 1)], [(2, 1), (3, 1), (4, 2)], [(0, 2), (3, 3), (5, 2), (6, 1), (7, 2), (8, 1)]]
#[[('are', 1), ('hello', 1), ('how', 1), ('you', 1)], [('how', 1), ('you', 1), ('do', 2)], [('are', 2), ('you', 3), ('doing', 2), ('hey', 1), ('what', 2), ('yes', 1)]]



#------ Creating TF-IDF(Term Frequency – Inverse Document Frequency)------

import gensim
from gensim.utils import simple_preprocess
from gensim import corpora

text = ["The climate is excellent but the weather can be better",
        "The food is always delicious and loved the service of restaurant",
        "The food was mediocre and the service was terrible"]

g_dict = corpora.Dictionary([simple_preprocess(line) for line in text])
g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in text]

print("Dictionary : ")
for item in g_bow:
    print([[g_dict[id], freq] for id, freq in item])

g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')

print("TF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])

# Output:

#Dictionary : 
#[['be', 1], ['better', 1], ['but', 1], ['can', 1], ['climate', 1], ['excellent', 1], ['is', 1], ['the', 2], ['weather', 1]]
#[['is', 1], ['the', 2], ['always', 1], ['and', 1], ['delicious', 1], ['food', 1], ['loved', 1], ['of', 1], ['restaurant', 1], ['service', 1]]
#[['the', 2], ['and', 1], ['food', 1], ['service', 1], ['mediocre', 1], ['terrible', 1], ['was', 2]]
