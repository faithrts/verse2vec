import pandas as pd
import gensim

from gensim.models import Word2Vec

def train_model(tokens):
    model = Word2Vec(sentences = tokens, seed = 0, workers = 1, sg = 0, min_count = 1)
    return model

if __name__ == '__main__':

    poetry_df = pd.read_csv('../data/sentences_poetry_tokens.csv', index_col = 0)
    poetry_df['TOKENS'] = [eval(tokens) for tokens in poetry_df['TOKENS']]

    tokens = poetry_df['TOKENS']
    model = train_model(tokens)

    model.save('../models/sentences_tokens_word2vec.model')