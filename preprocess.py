import numpy as np
import pandas as pd
from tweet import Tweet

PAD_TOKEN = "*$*PAD*$*"
UNK_TOKEN = "*$*UNK*$*"
UNK_AUTHOR = "$#$#Nan#$#$"


def create_batches(tweets, w2i, a2i, pad_tnk=PAD_TOKEN, unk_tkn=UNK_TOKEN, unk_author=UNK_AUTHOR, batch_size=128,
                   max_len=100, sort_data=True):
    """
    :param tweets: a list of Post objects
    :param w2i: a word-to-index dictionary with all embedded words that will be used in training
    :param pad_tnk: the pad token
    :param unk_tkn: the unknown token
    :param unk_author: the unknown author
    :param batch_size: how many posts will be in every batch
    :param max_len: the padding size for the texts
    :param sort_data: boolean indicating if the list of posts  will be sorted by the size of the text
    :param a2i: a author-to-index dictionary
    :unk_author: unk author
    :return: a list of batches
    """
    if sort_data:
        tweets.sort(key=lambda x: -len(x.tokens))
    offset = 0
    batches = []
    while offset < len(tweets):
        batch_texts = []
        batch_authors = []
        batch_labels = []
        start = offset
        end = min(offset + batch_size, len(tweets))
        for i in range(start, end):
            batch_max_size = tweets[start].text_size if sort_data else max(list(map(lambda x: x.text_size, tweets[start:end])))
            batch_texts.append(get_indexed_text(w2i, pad_text(tweets[i].tokens, max(min(max_len, batch_max_size), 1), pad_tkn=pad_tnk), unk_token=unk_tkn))
            batch_authors.append(get_indexed_value(a2i, tweets[i].author, unk_author))
            batch_labels.append(tweets[i].label)
        batches.append({'text': np.array(batch_texts),
                        'author': np.array(batch_authors),
                        'label': np.array(batch_labels, dtype='float32')})
        offset += batch_size
    return batches


def get_embeddings(path='../input/embeddings/twitter/glove.twitter.27B.%dd.txt', size=50):
    """
    :param path: the directory where all glove twitter embeddings are stored.
    glove embeddings can be downloaded from https://nlp.stanford.edu/projects/glove/
    :param size: the size of the embeddings. Must be in [25, 50, 100, 200]
    :return: a word-to-list dictionary with the embedded words and their corresponding embedding
    """
    embeddings_dict = {}
    f_path = path % size
    with open(f_path) as f:
        for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
    return embeddings_dict


def create_freq_vocabulary(tokenized_texts):
    """
    :param tokenized_texts: a list of lists of tokens
    :return: a word-to-integer dictionary with the value representing the frequency of the word in data
    """
    token_dict = {}
    for text in tokenized_texts:
        for token in text:
            try:
                token_dict[token] += 1
            except KeyError:
                token_dict[token] = 1
    return token_dict


def get_frequent_words(token_dict, min_freq):
    """
    :param token_dict: a word-to-integer dictionary with the value representing the frequency of the word in data
    :param min_freq: the minimum frequency
    :return: the list with tokens having frequency >= min_freq
    """
    return [x for x in token_dict if token_dict[x] >= min_freq]


def create_final_dictionary(tweets,
                            min_freq,
                            unk_token,
                            pad_token,
                            embeddings_dict):
    """
    :param tweets: a list of Post objects
    :param min_freq: the min times a word must be found in data in order not to be considered as unknown
    :param unk_token: the unknown token
    :param pad_token: the pad token
    :param embeddings_dict: a word-to-list dictionary with the embedded words and their corresponding embedding
    :return: a word-to-index dictionary with all the words that will be used in training
    """
    tokenized_texts = [x.tokens for x in tweets]
    voc = create_freq_vocabulary(tokenized_texts)
    print("tokens found in training data set:", len(voc))
    freq_words = get_frequent_words(voc, min_freq)
    print("tokens with frequency >= %d: %d" % (min_freq, len(freq_words)))
    words = list(set(freq_words).intersection(embeddings_dict.keys()))
    print("embedded tokens with frequency >= %d: %d" % (min_freq,len(words)))
    words = [pad_token, unk_token] + words
    return {w: i for i, w in enumerate(words)}


def create_author_dictionary(tweets, min_freq):
    authors = [x.author for x in tweets]
    author_dict = {}
    for author in authors:
        try:
            author_dict[author] += 1
        except KeyError:
            author_dict[author] = 1
    final_authors = [UNK_AUTHOR] + [x for x in author_dict if author_dict[x] >= min_freq]
    return {a: i for i, a in enumerate(final_authors)}


def get_embeddings_matrix(word_dict, embeddings_dict, size):
    """
    :param word_dict: a word-to-index dictionary with the tokens found in data
    :param embeddings_dict: a word-to-list dictionary with the embedded words and their corresponding embedding
    :param size: the size of the word embedding
    :return: a matrix with all the embeddings that will be used in training
    """
    embs = np.zeros(shape=(len(word_dict), size))
    for word in word_dict:
        try:
            embs[word_dict[word]] = embeddings_dict[word]
        except KeyError:
            print('no embedding for: ', word)
    embs[1] = np.mean(embs[2:])

    return embs


def get_indexed_value(w2i, word, unk_token):
    """
    return the index of a token in a word-to-index dictionary
    :param w2i: the word-to-index dictionary
    :param word: the token
    :param unk_token: to unknown token
    :return: an integer
    """
    try:
        return w2i[word]
    except KeyError:
        return w2i[unk_token]


def get_indexed_text(w2i, words, unk_token):
    """
    return the indices of the all the tokens in a list in a word-to-index dictionary
    :param w2i: the word-to-index dictionary
    :param words: a list of tokens
    :param unk_token: to unknown token
    :return: a list of integers
    """
    return [get_indexed_value(w2i, word, unk_token) for word in words]


def pad_text(tokenized_text, maxlen, pad_tkn):
    """
    fills a list of tokens with pad tokens if the length of the list is larger than maxlen
    or return the maxlen last tokens of the list
    :param tokenized_text: a list of tokens
    :param maxlen: the max length
    :param pad_tkn: the pad token
    :return: a list of tokens
    """
    if len(tokenized_text) < maxlen:
        return [pad_tkn] * (maxlen - len(tokenized_text)) + tokenized_text
    else:
        return tokenized_text[len(tokenized_text) - maxlen:]


def load_data(csv_file):
    """
    loads a csv file with posts and replaces the empty titles and texts
    :param csv_file: the file containing all available data

    :return:
    """
    data = pd.read_csv(csv_file).dropna()
    return data.sample(len(data))


def get_posts(df, is_test=False, size=200000):
    """
    returns a list of Tweets. Tweets with the same text we be considered as duplicates
    :param df: a dataframe  - the source of the data
    :param is_test: at test.csv there is no labels
    :param size: how many data will be used for training and validation
    (due to hardware constrains not all data can be used)
    :return: all the posts in this dataset
    """
    posts = set()
    if is_test:
        for i, d in df.iterrows():
            post = Tweet(None, d[0],  d[1], d[2], d[3], d[4])
            if post.get_text_size() > 0:
                posts.add(post)
    else:
        for i, d in df.iterrows():
            post = Tweet(*d)
            if post.get_text_size() > 0:
                posts.add(post)
    posts = list(posts)[0:size]
    posts.sort(key=lambda x: x.date)
    return list(posts)


def split_data(data, split_point):
    """
    splits the data (for train and test)
    :param data: a list of tweets
    :param split_point: the point of splitting
    :return: two lists of tweets
    """
    return data[0:split_point], data[split_point:]


def generate_data(split_point, emb_size, size=200000, min_freq=1, min_author_freq=3,
                  max_len=1000):
    """
    generates all necessary components for training and evaluation (posts, embedding matrix, dictionaries and batches
    :param split_point: how many data will be used for training. the rest, will be used for evaluation
    :param size: how many data will be used for training and validation
    :param emb_size: to size of word embeddings
    :param min_freq: how many times a word must be found in data in order t not being considered as unknown
    :param min_author_freq: least number of posts of the author in the dataset in order not to be consider unk
    even if its embedding is available
    :param max_len: the padding size of text
    :return: train_posts, test_posts, w2i, emb_matrix, train_batches, test_batches
    """
    df = load_data("../input/sentiment/training.1600000.processed.noemoticon.csv")
    posts = get_posts(df, size=size)
    train_posts, test_posts = split_data(posts, split_point)
    print('tweets for training:', len(train_posts))
    print('tweets for testing:', len(test_posts))
    embeddings_dict = get_embeddings(size=emb_size)
    w2i = create_final_dictionary(tweets=posts, min_freq=min_freq, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN,
                                  embeddings_dict=embeddings_dict)

    emb_matrix = get_embeddings_matrix(w2i, embeddings_dict, size=emb_size)
    a2i = create_author_dictionary(posts, min_author_freq)
    train_batches = create_batches(train_posts, w2i, a2i, max_len=max_len)
    test_batches = create_batches(test_posts, w2i, a2i, max_len=max_len)

    return {'train_posts': train_posts, 'test posts': test_posts, 'w2i':w2i,
            'a2i': a2i, 'emb_matrix': emb_matrix,
            'train_batches': train_batches, 'test_batches': test_batches}


