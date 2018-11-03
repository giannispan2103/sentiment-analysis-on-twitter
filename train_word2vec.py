import gensim
import re
import pandas as pd


def clean(text, lower=True):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :param text: The string to be cleaned
    :param lower: If True text is converted to lower case
    :return: The clean string
    """
    # text = re.sub('[@][]?[^ @]+', ' #USER ', text)

    text = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", text)
    text = re.sub('[@][A-Za-z0-9]+', ' $USER ', text)
    text = re.sub(r"[#][A-Za-z0-9]+", "$HASHTAG", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower().encode('utf-8') if lower else text.strip().encode('utf-8')


def run_w2v():
    data = pd.read_csv("../input/sentiment/training.1600000.processed.noemoticon.csv").dropna()
    data = data.sample(len(data))
    texts = list(map(lambda x: clean(str(x)).split(), data['text'].values))
    # for i in range(100):
    #     print(texts[i])
    # exit()
    model = gensim.models.Word2Vec(texts, size=100, window=5, min_count=3, workers=4, iter=10)
    model.train(texts, total_examples=len(texts))
    model.save("new_embeddings")

def load():
    model = gensim.models.Word2Vec.load('new_embeddings')
    print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))


if __name__=="__main__":
    # run_w2v()
    load()