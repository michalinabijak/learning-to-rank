import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import nltk
import string
import contractions
import operator
from nltk.stem import WordNetLemmatizer

from scipy import optimize
from scipy.stats import linregress

sns.set(style='white', rc={"grid.linewidth": 0.1})
sns.set_context("paper", font_scale=1.5)

test_queries = pd.read_csv("/.../dataset/test-queries.tsv", sep="\t", header=None)
passages = pd.read_csv("/.../dataset/passage_collection_new.txt", sep="\t", header=None)
candidate = pd.read_csv(".../dataset/candidate_passages_top1000.tsv", sep="\t", header=None)

# rename columns
test_queries.columns=["qid", "query"]
passages.columns=["passage"]
candidate.columns=["qid", "pid", "query", "passage"]

# helper function, flattens a list of lists
flatten = lambda t: [item for sublist in t for item in sublist]

# Zipf's Law: pre-processing, Maximum Likelihood parameter estimation and verification

#contract, tokenize and remove punctuation
full_text = [contractions.fix(sentence) for sentence in passages.passage]
full_text = [sentence for sentence in passages.passage]
full_text = flatten([nltk.word_tokenize(word) for word in full_text])
full_text = [word.lower() for word in full_text if word.isalpha()]

# frequency distribution of full text
fdist = nltk.FreqDist(full_text)
# get tokens, frequencies and probabilities sorted in descending order
tokens = np.array(sorted(fdist.keys(), key = fdist.get, reverse=True))
frequencies = np.array(sorted(fdist.values(), reverse=True))
probabilities = np.array(sorted(fdist.values(), reverse=True)) / fdist.N()

# define a Maximum Likelihood Estimator for parameter c, based on square error between estimated (c/rank) and true probability
def MLE(c):
    mle = 0
    for i in range(5000):
        mle += np.log((probabilities[i] - c/(i+1))**2)
    return mle