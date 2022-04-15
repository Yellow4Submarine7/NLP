from nltk import tokenize
import nltk.data
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import logging
from gensim.models import word2vec

#Read data
train = pd.read_csv("labeldTrainData.tsv", header = 0, \
    delimiter = "\t", quoting = 3)
test = pd.read_csv("test.Data.tsv",header = 0, delimiter = "\t", quoting = 3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv",header = 0, \
    delimiter = "\t", quoting = 3)

print("Read %d labeled train reviews, %d labeled test reviews,"\
    "and %d unlabeled reviews\n" % (train["review"].size,
    test["review"].size,unlabeled_train['review'].size))

def review_to_wordlist(review, remove_stopwords = False, remove_nums = False):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("^a-zA-Z"," ",review_text)
    if remove_nums:
        review_text = re.sub("0-9",review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review,tokenizer,remove_stopwords = False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence,remove_stopwords))
            return sentences

sentences = []
print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review,tokenizer)

logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s",level=logging.INFO)
num_features = 300
min_word_cout = 40
num_workers = 4
context = 10
downsampling = 1e-3

print("Training model...")
model = word2vec.Word2Vec(sentences,workers=num_workers,
                            size = num_features, min_count = min_word_cout,
                            window = context, sample = downsampling)
model.init_sims(replace=True)

model_name = "1111"
model.save(model_name)





    



