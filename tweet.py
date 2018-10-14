import re


def clean_en(text, lower=True):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :param text: The string to be cleaned
    :param lower: If True text is converted to lower case
    :return: The clean string
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
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


class Tweet(object):
    """
    this class encapsulates data taken from https://www.kaggle.com/kazanova/sentiment140
    """
    def __init__(self, label, post_id, date, no_query, author,  text):
        self.post_id = post_id
        self.label = label > 0
        self.text = text
        self.cleaned_text = clean_en(self.text, False)
        init_tokens = self.cleaned_text.split()
        self.uppercase_indicators = [x.isupper() for x in init_tokens]
        self.tokens = [x.lower() for x in init_tokens]
        self.text_size = self.get_text_size()
        self.date = date
        self.author = author
        self.no_query = no_query

    def get_text_size(self):
        return len(self.tokens)

    def get_text_uppercase_score(self):
        return sum(self.uppercase_indicators) / float(self.text_size)

    def __eq__(self, other):
        return self.text == other.text

    def __ne__(self, other):
        return self.text != other.text

    def __hash__(self):
        return hash(self.text)



