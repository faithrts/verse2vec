import pandas as pd
import regex as re
import numpy as np
import nltk
import tqdm

from tqdm import tqdm

# --------------------------- global variables --------------------------- #

path_to_data_folder = '../data'

# ------------------------------- datasets ------------------------------- #

def gutenberg():
    # data from https://huggingface.co/datasets/google-research-datasets/poem_sentiment
    test = pd.read_parquet(f'{path_to_data_folder}/gutenberg/test-00000-of-00001.parquet', engine='pyarrow').set_index('id')
    train = pd.read_parquet(f'{path_to_data_folder}/gutenberg/train-00000-of-00001.parquet', engine='pyarrow').set_index('id')
    valid = pd.read_parquet(f'{path_to_data_folder}/gutenberg/validation-00000-of-00001.parquet', engine='pyarrow').set_index('id')

    gutenberg_df = pd.concat([test, train, valid])
    gutenberg_df.rename(columns = {"verse_text": "TEXT"}, inplace = True)

    gutenberg_df = gutenberg_df[['TEXT']]

    return gutenberg_df

def english_pcd():
    pcd_df = pd.read_csv(f'{path_to_data_folder}/english_pcd/merged_data.csv', index_col = 0)
    pcd_df.rename(columns = {"Verse": "TEXT"}, inplace = True)

    pcd_df = pcd_df[['TEXT']]
    return pcd_df

def poki():
    # data from https://github.com/whipson/PoKi-Poems-by-Kids/tree/master
    poki_df = pd.read_csv(f'{path_to_data_folder}/poki/poki.csv')
    poki_df.rename(columns = {'text': 'TEXT'}, inplace = True)

    poki_df = poki_df[['TEXT']]

    # split_sentences

    return poki_df

def perc():
    # data from https://data.mendeley.com/datasets/n9vbc8g9cx/1
    perc_df = pd.read_csv(f'{path_to_data_folder}/perc/PERC_mendelly.csv')
    perc_df.rename(columns = {'Poem': 'TEXT'}, inplace = True)

    perc_df = perc_df[['TEXT']]
    return perc_df

def poetry_foundation():
    # data from https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems
    poetry_foundation_df = pd.read_csv(f'{path_to_data_folder}/poetry_foundation/PoetryFoundationData.csv')
    poetry_foundation_df.rename(columns = {'Poem': 'TEXT'}, inplace = True)

    poetry_foundation_df = poetry_foundation_df[['TEXT']]
    return poetry_foundation_df

# ------------------------------- reformatting df ------------------------------- #

def split_sentences(df, focus_col = 'TEXT', default = True, source = ''):
    all_sentences = []

    for index, row in df.iterrows():
        cur_text = row[focus_col]

        if default:
            # default sentence splitter
            all_sentences += nltk.sent_tokenize(cur_text)
        else:
            # invoke source-specific sentence splitter
            all_sentences += eval(f'split_{source}')

def split_perc(text):
    return [phrase for phrase in (text.split('\n')) if phrase != '']

def split_poetry_foundation(text):
    stanzas = [phrase for phrase in (text.split('\r\n\r\n \r\n\r\n'))]
    cleaned_stanzas = [re.sub('\r\n\r\n', ' ', phrase) for phrase in stanzas]
    stanzas_by_sentence = [nltk.sent_tokenize(phrase) for phrase in cleaned_stanzas]
    return [sentence for stanza in stanzas_by_sentence for sentence in stanza]

# ------------------------------- cleaning text ------------------------------- #

def remove_backslash_breaks(text):

    # replace "[text]\n[text]" with "[text] [text]"
    text = re.sub('(?<=[A-Za-z])(\n|\r)+(?=[A-Za-z])', ' ', text)

    # replace \n or \r in "[text] \n[text]" or "[text]\n [text]" or other with empty string
    text = re.sub('(\n|\r)+', '', text)

    return text

""" removes \n, \r, and converts all text to lowercase """
def clean_text(text):
    return remove_backslash_breaks(text.lower())

# ------------------------------- main ------------------------------- #

if __name__ == '__main__':

    list_of_sources = ['gutenberg', 'english_pcd', 'poki', 'perc', 'poetry_foundation']
    
    for cur_source in tqdm(list_of_sources):

        # calls function that creates dataframe from raw dataset
        cur_df = eval(f'{cur_source}()')

        # cleans text
        cur_df['TEXT'] = [clean_text(text) for text in cur_df['TEXT']]

        # removes rows comprised of nothing (before, they were \n\r\n or some combo)
        cur_df.replace('', np.nan, inplace = True)
        cur_df.dropna(inplace = True, ignore_index = True)

        # saves to csv
        cur_df.to_csv(f'{path_to_data_folder}/{cur_source}/{cur_source}_clean_text.csv')