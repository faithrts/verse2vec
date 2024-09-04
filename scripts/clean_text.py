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

    # split by sentences
    poki_sentences = split_sentences(poki_df)
    poki_sentences_df = pd.DataFrame(poki_sentences, columns = ['TEXT'])

    return poki_sentences_df

def perc():
    # data from https://data.mendeley.com/datasets/n9vbc8g9cx/1
    perc_df = pd.read_csv(f'{path_to_data_folder}/perc/PERC_mendelly.csv')
    perc_df.rename(columns = {'Poem': 'TEXT'}, inplace = True)

    perc_df = perc_df[['TEXT']]

    # split by sentences
    perc_sentences = split_sentences(perc_df, source = 'perc')
    perc_sentences_df = pd.DataFrame(perc_sentences, columns = ['TEXT'])

    return perc_sentences_df

def poetry_foundation():
    # data from https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems
    poetry_foundation_df = pd.read_csv(f'{path_to_data_folder}/poetry_foundation/PoetryFoundationData.csv')
    poetry_foundation_df.rename(columns = {'Poem': 'TEXT'}, inplace = True)

    poetry_foundation_df = poetry_foundation_df[['TEXT']]

    # split by sentences
    poetry_foundation_sentences = split_sentences(poetry_foundation_df, source = 'poetry_foundation')
    poetry_foundation__sentences_df = pd.DataFrame(poetry_foundation_sentences, columns = ['TEXT'])

    return poetry_foundation__sentences_df

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
        cur_df.replace(' ', np.nan, inplace = True)
        cur_df.replace('.', np.nan, inplace = True)
        cur_df.dropna(inplace = True, ignore_index = True)

        # saves to csv
        cur_df.to_csv(f'{path_to_data_folder}/{cur_source}/{cur_source}_clean_text.csv')