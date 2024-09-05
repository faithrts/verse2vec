import pandas as pd
import nltk
import tqdm

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

from tqdm import tqdm

# ------------------------------- global variables ------------------------------- #

lemmatizer = WordNetLemmatizer()
path_to_data_folder = '../data'

# ------------------------------- collecting data ------------------------------- #

def find_and_combine_data(list_of_sources):
    poetry_df = pd.DataFrame(columns = ['TEXT', 'SOURCE'])

    for cur_source in list_of_sources:
        cur_df = pd.read_csv(f'{path_to_data_folder}/{cur_source}/{cur_source}_clean_text.csv', index_col = 0)

        cur_df['SOURCE'] = cur_source

        poetry_df = pd.concat([poetry_df, cur_df], ignore_index = True)

    return poetry_df

# ------------------------------- tokenizing text ------------------------------- #

def remove_stopwords(tokens):
    return [token for token in tokens if token not in stopwords.words('english')]

nltk_to_wn = {'J': 'a',
                  'V': 'v',
                  'N': 'n',
                  'R': 'r'}

def tokenize_and_lemmatize(text):

    tokens = word_tokenize(text)

    # removes one-character tokens
    tokens = [token for token in tokens if len(token) > 1]
    tokens = remove_stopwords(tokens)

    tokens_and_tags = nltk.pos_tag(tokens)

    # lemmatization 
    cleaned_tokens = []
    for token, tag in tokens_and_tags:
        if tag[0] not in nltk_to_wn:
            cleaned_tokens.append(lemmatizer.lemmatize(token))
        else:
            wn_tag = nltk_to_wn[tag[0]]
            cleaned_tokens.append(lemmatizer.lemmatize(token, wn_tag))

    return cleaned_tokens

# ------------------------------- main ------------------------------- #

if __name__ == "__main__":

    list_of_sources = ['gutenberg', 'english_pcd', 'poki', 'perc', 'poetry_foundation']

    poetry_df = find_and_combine_data(list_of_sources)

    # adds tokens to df comprised of all poetry datasets combined
    # tqdm shows a progress bar in the terminal
    poetry_df['TOKENS'] = ''
    for index, row in tqdm(poetry_df.iterrows(), total = len(poetry_df)):
        cur_text = row['TEXT']
        cleaned_tokens = tokenize_and_lemmatize(cur_text)

        poetry_df.at[index, 'TOKENS'] = cleaned_tokens

    # keep only sentences with at least 2 tokens
    poetry_df = poetry_df[poetry_df.TOKENS.map(len) >= 2]

    # reset index
    poetry_df.reset_index(drop = True, inplace = True)

    poetry_df.to_csv(f'{path_to_data_folder}/sentences_poetry_tokens.csv')