import pandas as pd

def gutenberg_poem():
    
    # data from https://huggingface.co/datasets/google-research-datasets/poem_sentiment
    test = pd.read_parquet('data/test-00000-of-00001.parquet', engine='pyarrow').set_index('id')
    train = pd.read_parquet('data/train-00000-of-00001.parquet', engine='pyarrow').set_index('id')
    valid = pd.read_parquet('data/validation-00000-of-00001.parquet', engine='pyarrow').set_index('id')

    all_df = pd.concat([test, train, valid])

    # label: the sentiment label
    # 0 = negative
    # 1 = positive
    # 2 = no impact

    all_df.rename(columns = {"verse_text": "TEXT"})

    return all_df

def english_pcd():
    pcd_df = pd.read_csv('merged_data.csv', index_col = 0)
    pcd_df.rename(columns = {"Verse": "TEXT"})

    return pcd_df

if __name__ == "__main__":
    print('hi')
