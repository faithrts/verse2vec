import pandas as pd
import gensim

from gensim.models import Word2Vec

def train_model(tokens):
    model = Word2Vec(sentences = tokens, seed = 0, workers = 1, sg = 0, min_count = 2)
    return model

if __name__ == '__main__':

    subset = 'splits'

    poetry_df = pd.read_csv(f'../data/{subset}_poetry_tokens.csv', index_col = 0)
    poetry_df['TOKENS'] = [eval(tokens) for tokens in poetry_df['TOKENS']]

    tokens = poetry_df['TOKENS']
    model = train_model(tokens)

    model.save(f'../models/{subset}_tokens_word2vec.model')