import pandas as pd
import regex as re

# ------------------------------- datasets ------------------------------- #

def gutenberg_poem():
    # data from https://huggingface.co/datasets/google-research-datasets/poem_sentiment
    test = pd.read_parquet('data/test-00000-of-00001.parquet', engine='pyarrow').set_index('id')
    train = pd.read_parquet('data/train-00000-of-00001.parquet', engine='pyarrow').set_index('id')
    valid = pd.read_parquet('data/validation-00000-of-00001.parquet', engine='pyarrow').set_index('id')

    all_df = pd.concat([test, train, valid])
    all_df.rename(columns = {"verse_text": "TEXT"}, inplace = True)

    return all_df

def english_pcd():
    pcd_df = pd.read_csv('merged_data.csv', index_col = 0)
    pcd_df.rename(columns = {"Verse": "TEXT"}, inplace = True)

    return pcd_df

def poki():
    # data from https://github.com/whipson/PoKi-Poems-by-Kids/tree/master
    poki_df = pd.read_csv('data/poki/poki.csv')
    poki_df.rename(columns = {'text': 'TEXT'}, inplace = True)

    return poki_df

def perc():
    # data from https://data.mendeley.com/datasets/n9vbc8g9cx/1
    perc_df = pd.read_csv('data/perc/PERC_mendelly.csv')
    perc_df.rename(columns = {'Poem': 'TEXT'}, inplace = True)

    return perc_df

def poetry_foundation():
    # data from https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems
    poetry_foundation_df = pd.read_csv('data/poetry_foundation/PoetryFoundationData.csv')
    poetry_foundation_df.rename(columns = {'Poem': 'TEXT'}, inplace = True)

    return poetry_foundation_df

# ------------------------------- cleaning text ------------------------------- #

def remove_backslash_breaks(text):
    # replace "[text]\n[text]" with "[text] [text]"
    text = re.sub('(?<=[A-Za-z])(\\[[nr])+(?=[A-Za-z])', ' ', text)

    # replace \n or \r in "[text] \n[text]" or "[text]\n [text]" or other with empty string
    text = re.sub('(\\[nr])+', '', text)

    return text

def tokenize(text):
    # TODO: import library like punkt or TreebankTokenizer
    return text.split(' ')

def preprocess_text(text):
    return tokenize(remove_backslash_breaks(text.lower))

def preprocess_df(df, target_col = 'TEXT'):
    df[target_col] = df[target_col].apply(preprocess_text)

    return df

# ------------------------------- main ------------------------------- #

if __name__ == "__main__":
    gutenberg_df = gutenberg_poem()
    english_pcd_df = english_pcd()
    poki_df = poki()
    perc_df = perc()
    poetry_foundation_df = poetry_foundation()

    for df_name in ['gutenberg', 'english_pcd', 'poki', 'perc', 'poetry_foundation']:

        cur_df = eval(f'{df_name}_df')

        preprocess_df(cur_df)

        cur_df.to_csv(f'{df_name}.csv')