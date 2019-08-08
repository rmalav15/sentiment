import re
import string

from emoji.unicode_codes import UNICODE_EMOJI
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


# def preprocess_review(ReviewText):
#     ReviewText = ReviewText.str.replace("(<br/>)", "")
#     ReviewText = ReviewText.str.replace('(<a).*(>).*(</a>)', '')
#     ReviewText = ReviewText.str.replace('(&amp)', '')
#     ReviewText = ReviewText.str.replace('(&gt)', '')
#     ReviewText = ReviewText.str.replace('(&lt)', '')
#     ReviewText = ReviewText.str.replace('(\xa0)', ' ')
#     return ReviewText


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
