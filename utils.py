import re
import string

from emoji.unicode_codes import UNICODE_EMOJI
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix

# def preprocess_review(ReviewText):
#     ReviewText = ReviewText.str.replace("(<br/>)", "")
#     ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', '')
#     ReviewText = ReviewText.str.replace('(&amp)', '')
#     ReviewText = ReviewText.str.replace('(&gt)', '')
#     ReviewText = ReviewText.str.replace('(&lt)', '')
#     ReviewText = ReviewText.str.replace('(\xa0)', ' ')
#     return ReviewText

# Taken from official sklearn doc:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def clean_text(text):
    # lower
    text = text.lower()

    # Remove Punctuation
    punct = string.punctuation
    trantab = str.maketrans(punct, len(punct) * ' ')
    text = text.translate(trantab)

    # Remove Digits
    text = re.sub('\d+', '', text)

    # Remove Urls
    text = re.sub(r'http.?://[^\s]+[\s]?', '', text)

    # Replace emojis with unicode, Separate emojis
    text = ''.join(UNICODE_EMOJI[c][1:-1] + " " if (c in UNICODE_EMOJI) else c for c in text)

    tokens = word_tokenize(text)
    porter = PorterStemmer()
    detokenizer = TreebankWordDetokenizer()

    # Remove StopWords and Stemmize
    stopwords_list = stopwords.words('english')
    whitelist = ["n't", "not", "no"]  # Not to remove
    tokens = [porter.stem(t) for t in tokens if ((t not in stopwords_list) or (t in whitelist))]
    text = detokenizer.detokenize(tokens)

    return text


if __name__ == "__main__":
    text = " Well I just love it 😊😊 that's all I could say"
    print("text:", text)
    print("cleaned text:", clean_text(text))
