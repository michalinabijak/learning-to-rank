import numpy as np


def get_AP(df, qid, k):
    """
    df : pandas df with a 'score' column indicating a score based on some model
    return: Average Precision for a given query
          AP = sum(precision(i) * relevancy(i)) / (num of relevant docs)
    """
    # consider only the relevant documents
    relevant_docs = df[(df['qid'] == qid) & (df['relevancy'] != 0)]
    retrieved_relevant = relevant_docs[relevant_docs['rank'] <= k]
    n = len(relevant_docs)

    AP = retrieved_relevant['rank'].apply(lambda x: 1 / x).sum()
    AP /= n
    return AP


def get_MAP(data, k, column):
    """
    df : pandas dataframe with a 'score' column with scores based on a given model
    return: Mean Average Precision for all queries in the df
    """
    MAP = 0
    df = data  # to avoid in-place modyfying the dataframe
    df['rank'] = df.groupby('qid')[column].rank(ascending=False)

    for qid in df['qid'].unique():
        AP = get_AP(df, qid, k)
        MAP += AP
    MAP /= df['qid'].nunique()
    return MAP


def get_IDCG(df, qid):
    """
    non-relevant documents have relevancy 0
    Maximal attainable Discounted Cumulative Gain at k, DCG@k
    IDCG = sum((2^(rel)-1)/(log2(i+1))) for documents sorted by 'relevancy' in descending order
    """
    relevant_docs = df[(df['qid'] == qid) & (df['relevancy'] != 0)]
    relevant_docs['rel_rank'] = relevant_docs['relevancy'].rank(ascending=False)

    IDCG = relevant_docs.apply(lambda row: (2 ** row['relevancy'] - 1) / np.log2(row['rel_rank'] + 1), axis=1).sum()

    return IDCG


def get_NDCG(df, qid, k):
    """
    Normalised Cumulative Gain DCG@k
    """
    IDCG = get_IDCG(df, qid)
    retrieved_relevant = df[(df['qid'] == qid) & (df['relevancy'] != 0) & (df['rank'] <= k)]
    if len(retrieved_relevant) == 0:
        return 0
    else:
        DCG = retrieved_relevant.apply(lambda row: (2 ** row['relevancy'] - 1) / (np.log2(row['rank'] + 1)),
                                       axis=1).sum()
        NDCG = DCG / IDCG
        return NDCG


def get_MNDCG(data, k, column):
    """
    Mean NDCG@k
    """
    df = data  # to avoid in-place modyfying the dataframe
    df['rank'] = df.groupby('qid')[column].rank(ascending=False)
    MNDCG = 0
    for qid in df['qid'].unique():
        MNDCG += get_NDCG(df, qid, k)
    return MNDCG / df['qid'].nunique()
