import sys
import pickle
import pandas as pd
import numpy as np
import re

from textblob import TextBlob
from collections import Counter
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop += ['rt']

data_path = './data/'

def tweet2vec(text):
    """
    Return the word vector centroid for the text. Sum the word vectors
    for each word and then divide by the number of words. Ignore words
    not in gloves.
    """
    centroid = []
    count = 0
    words_list = TextBlob(text).tokens
    for word in words_list:
        if word in gloves:
            centroid.append(np.array(gloves[word]))
            count = count + 1
    if count!=0:
        centroid = np.sum(centroid, axis=0) / count
        return centroid
    print('Invalid String, None of these words are present in our vector encodings')
    return -1


def load_glove(filename):
    """
    Read all lines from the indicated file and return a dictionary
    mapping word:vector where vectors are of numpy `array` type.
    GloVe file lines are of the form:

    the 0.418 0.24968 -0.41242 0.1217 ...

    So split each line on spaces into a list; the first element is the word
    and the remaining elements represent factor components. The length of the vector
    should not matter; read vectors of any length.
    """
    f = open(filename, 'r')
    word_dict = dict()
    for line in f:
        word_list = line.split()
        word_dict[word_list[0]] = np.array(word_list[1:])
        word_dict[word_list[0]] = word_dict[word_list[0]].astype(np.float)
    return word_dict


def tokenize_input(inp):
    inp = inp.replace('[^\w\s]','')
    inp = inp.replace('[\\r|\\n|\\t|_]',' ')
    inp = inp.strip()
    inp = re.sub(r'[^a-zA-Z ]+', '', inp)
    inp = ' '.join([word for word in inp.split() if word.lower() not in (stop) and len(word)>2])
    return inp


def predict_fake_tweeter(input_str, model, user_encoder):
    input_string = tokenize_input(input_str)
    inp_centroid = tweet2vec(input_string)
    if type(inp_centroid)!='int':
        predicted_user = model.predict(np.array(inp_centroid).reshape(1, -1).astype(np.float))
        predicted_user_id = user_encoder.inverse_transform(predicted_user)
        if len(predicted_user)>0:
            return predicted_user_id[0]
        return print('No similar Fake Tweeters found')


def give_details(user, data):
    filtered_result = data[data.user_id == user].reset_index(drop=True)

    tokens = list(TextBlob(re.sub(r'[^a-zA-Z ]+', '', ' '.join(
        filtered_result.hashtags.values))).tokens)
    c = Counter(tokens)

    print('''Here is your Fake Tweeter avatar ::


    Your Screen Name : %s,
    Your Fake Tweeter alias : %s,
    Description : %s,
    Number of Friends : %s,
    Location : %s,
    Most used Hashtags : %s,
    Recent Tweets : 
    ''' % (filtered_result.screen_name[0], filtered_result.name[0],
           filtered_result.description[0],
           filtered_result.friends_count[0], filtered_result.location[0],
           [x[0] for x in c.most_common(5)]))
    [print('- ', a) for a in
     filtered_result.nlargest(5, 'retweet_count').text.values]


if __name__ == "__main__":

    args = sys.argv
    if len(args)<2:
        print('Arguments or Input missing')
    input_string = str(args[1])
    print('Loading model and data...')
    model = pickle.load(open('knn_model_50enc', 'rb'))
    data = pd.read_pickle('merged_data')
    user_encoder = pickle.load(open('user_encoder', 'rb'))
    print('Loading vector encodings...')
    gloves = load_glove('./data/glove.6B.50d.txt')
    while input_string:
        predicted_user = predict_fake_tweeter(input_string, model, user_encoder)
        if predicted_user:
            give_details(predicted_user, data)
        input_string = input('Enter your input : ')
