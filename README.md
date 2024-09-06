# word2vec-training

# Purpose
This repo exists to train a word2vec model on different poetry datasets, yielding poetry-influenced word vectors.

# Datasets
- [English PCD (Poem Comprehensive Dataset)](https://hci-lab.github.io/LearningMetersPoems/)
- [Gutenberg Poem Dataset](https://huggingface.co/datasets/google-research-datasets/poem_sentiment)
- [Poem Emotion Recognition Corpus (PERC)](https://data.mendeley.com/datasets/n9vbc8g9cx/1)
- [PoKi: A Large Dataset of Poems by Children](https://github.com/whipson/PoKi-Poems-by-Kids/tree/master)
- [Poetry Foundation Poems](https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems)

# To do
1. Split poems into sentences for specific sources [DONE]
2. Add list of sentences as new entries in dataframes [DONE]
3. Remove duplicate poems [DONE]
4. Check why \t\t\t\t\t\t\t isn't being removed in clean_text.py ??? [DONE]
5. Find other instances of malformed data
6. Re-tokenize