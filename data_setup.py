import pandas as pd
import seaborn as sns

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