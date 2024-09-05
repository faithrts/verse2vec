import pandas as pd
import regex as re
import numpy as np
import nltk
import tqdm

from tqdm import tqdm

# --------------------------- global variables --------------------------- #

path_to_data_folder = '../data'

# ------------------------------- datasets ------------------------------- #

def standardize_df(df, text_col_name):
    # standardizes the TEXT column name, and removes other columns
    df.rename(columns = {text_col_name: 'TEXT'}, inplace = True)
    df = df[['TEXT']]

    # removes duplicates
    df = df.drop_duplicates()

    # resets index
    df.reset_index(inplace = True, drop = True)

    return df

def gutenberg():
    # data from https://huggingface.co/datasets/google-research-datasets/poem_sentiment
    test = pd.read_parquet(f'{path_to_data_folder}/gutenberg/test-00000-of-00001.parquet', engine='pyarrow').set_index('id')
    train = pd.read_parquet(f'{path_to_data_folder}/gutenberg/train-00000-of-00001.parquet', engine='pyarrow').set_index('id')
    valid = pd.read_parquet(f'{path_to_data_folder}/gutenberg/validation-00000-of-00001.parquet', engine='pyarrow').set_index('id')

    gutenberg_df = pd.concat([test, train, valid])
    gutenberg_df = standardize_df(gutenberg_df, 'verse_text')

    return gutenberg_df

def english_pcd():
    pcd_df = pd.read_csv(f'{path_to_data_folder}/english_pcd/merged_data.csv', index_col = 0)
    pcd_df = standardize_df(pcd_df, 'Verse')

    return pcd_df

def poki():
    # data from https://github.com/whipson/PoKi-Poems-by-Kids/tree/master
    poki_df = pd.read_csv(f'{path_to_data_folder}/poki/poki.csv')
    poki_df = standardize_df(poki_df, 'text')

    # split by sentences
    poki_sentences_df = split_sentences(poki_df)

    return poki_sentences_df

def perc():
    # data from https://data.mendeley.com/datasets/n9vbc8g9cx/1
    perc_df = pd.read_csv(f'{path_to_data_folder}/perc/PERC_mendelly.csv')
    perc_df = standardize_df(perc_df, 'Poem')

    # split by sentences
    perc_sentences_df = split_sentences(perc_df, source = 'perc')

    return perc_sentences_df

def poetry_foundation():
    # data from https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems
    poetry_foundation_df = pd.read_csv(f'{path_to_data_folder}/poetry_foundation/PoetryFoundationData.csv')
    poetry_foundation_df = standardize_df(poetry_foundation_df, 'Poem')
    
    # split by sentences
    poetry_foundation_sentences_df = split_sentences(poetry_foundation_df, source = 'poetry_foundation')

    return poetry_foundation_sentences_df

# ------------------------------- reformatting df ------------------------------- #

def split_sentences(df, focus_col = 'TEXT', source = ''):

    default = (source == '')

    all_sentences = []

    for index, row in df.iterrows():
        cur_text = row[focus_col]

        if default:
            # default sentence splitter
            all_sentences += nltk.sent_tokenize(cur_text)
        else:
            # invoke source-specific sentence splitter
            all_sentences += eval(f'split_{source}(cur_text)')

    return pd.DataFrame(all_sentences, columns = ['TEXT'])

def split_perc(text):
    sentences = [phrase for phrase in (text.split('\n')) if phrase != '']
    sentences = [phrase for phrase in sentences if len(phrase.split(' ')) > 2 and phrase != ' ']

    return sentences

def split_poetry_foundation(text):
    stanzas = [phrase for phrase in (text.split('\r\n\r\n \r\n\r\n'))]
    cleaned_stanzas = [re.sub('\r\n\r\n', ' ', phrase) for phrase in stanzas]
    stanzas_by_sentence = [nltk.sent_tokenize(phrase) for phrase in cleaned_stanzas]
    flattened = [sentence for stanza in stanzas_by_sentence for sentence in stanza if sentence != ' ']
    sentences = [sentence for sentence in flattened if len(sentence.split(' ')) > 2]
    
    return sentences

# ------------------------------- cleaning text ------------------------------- #

def remove_backslash_breaks(text):

    # replace "[text]\n[text]" with "[text] [text]"
    text = re.sub('(?<=[A-Za-z])(\n|\r|\t)+(?=[A-Za-z])', ' ', text)

    # replace \n or \r in "[text] \n[text]" or "[text]\n [text]" or other with empty string
    text = re.sub('(\n|\r|\t)+', '', text)

    return text

""" removes \n, \r, \t and converts all text to lowercase """
def clean_text(text):
    return remove_backslash_breaks(text.lower())

# ------------------------------- main ------------------------------- #

if __name__ == '__main__':

    list_of_sources = ['gutenberg', 'english_pcd', 'poki', 'perc', 'poetry_foundation']
    
    for cur_source in list_of_sources:

        # calls function that creates dataframe from raw dataset
        cur_df = eval(f'{cur_source}()')

        # cleans text
        cur_df['TEXT'] = [clean_text(text) for text in cur_df['TEXT']]

        # removes rows comprised of '', ' ', '.' etc. bc of splitting sentences
        cur_df = cur_df[cur_df.TEXT.map(len) >= 2]

        # reset index
        cur_df.reset_index(drop = True, inplace = True)

        # saves to csv
        cur_df.to_csv(f'{path_to_data_folder}/{cur_source}/{cur_source}_clean_text.csv')