# verse2vec

This repo exists to train a word2vec model on different poetry datasets, yielding verse-influenced word vectors.

### Datasets used
- [English PCD (Poem Comprehensive Dataset)](https://hci-lab.github.io/LearningMetersPoems/)
- [Gutenberg Poem Dataset](https://huggingface.co/datasets/google-research-datasets/poem_sentiment)
- [Poem Emotion Recognition Corpus (PERC)](https://data.mendeley.com/datasets/n9vbc8g9cx/1)
- [Poetry Foundation Poems](https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems)
- [PoKi: A Large Dataset of Poems by Children](https://github.com/whipson/PoKi-Poems-by-Kids/tree/master)

### Subsets
- poetry: splits the data as is
- sentences: splits the data into component sentences
- splits: leaves English PCD, Gutenberg, and PoKi data as is, splits the others by stanza

### Files
- `[source]\_clean\_text\_[subset].csv`: cleaned versions of the datasets used
- `[subset]_tokens.csv`: tokenized versions of the cleaned data
- `[subset]\_tokens\_word2vec.model`: models trained on tokenized cleaned data

## To do
1. Split poems into sentences for specific sources [DONE]
2. Add list of sentences as new entries in dataframes [DONE]
3. Remove duplicate poems [DONE]
4. Check why `\t\t\t\t\t\t\t` isn't being removed in `clean_text.py` ??? [DONE]
5. Find other instances of malformed data [DONE]
6. Re-tokenize as lines, stanzas, entire poems [DONE]
7. Test word vectors [IN PROGRESS]
