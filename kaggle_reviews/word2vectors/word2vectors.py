import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

train = pd.read_csv("labeledTrainData.tsv",
header = 0,delimiter = "\t", quoting = 3)

test = pd.read_csv("testData.tsv",header = 0,
delimiter = "\t", quoting = 3)

unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header = 0,
delimiter = "\t", quoting = 3)

print("Read %d labeled train reviews, %d labeled test reviews, " \
    "and %d unlabeled reviews \n" % (train["review"].size,
    test["review"].size, unlabeled_train["review"].size))




