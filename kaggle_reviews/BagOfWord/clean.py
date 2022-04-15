from nltk.corpus.reader.panlex_lite import Meaning
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter='\t',quoting=3)

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub('[^a-zA-Z]',' ',review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words('english')) #using set for speed
    meaningful_words = [w for w in words if not w in stops]
    return(' '.join(meaningful_words))

num_reviews = train["review"].size

clean_train_reviews = []

print("Cleaning the set")
for i in range(0,num_reviews):
    if((i+1)%1000 == 0):
        print("Review %d of %d\n" % (i+1,num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))


