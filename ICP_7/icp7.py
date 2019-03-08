# Import libraries
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
from bs4 import BeautifulSoup
from bs4.element import Comment
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import everygrams
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# Download extras from nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# Start of parts 1, 2, and 3


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


# Extracts visible text from html
# @param html_webpage html version of a webpage
#
# @returns array Extracted text
def text_from_html(html_webpage):
    soup = BeautifulSoup(html_webpage, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)


google_wiki_html = urllib2.urlopen("https://en.wikipedia.org/wiki/Google").read()
google_wiki_text = text_from_html(google_wiki_html)

# Open a file if it doesnt exist and overwrite with the state information
file = open("input.txt", "w")
file.write(google_wiki_text)

# Close the file
file.close()

# Apply tokenization
tokens = nltk.word_tokenize(google_wiki_text)
print('\nTokenization output', '\n', tokens)

# Apply POS
pos_tags = nltk.pos_tag(tokens)
print('\nPOS output', '\n', pos_tags)

# Apply stemming
ps = PorterStemmer()
print('\nStemming output')

count = 1
for token in tokens:
    print(token + ":" + ps.stem(token))

    count += 1
    if count == 50:
        break

# Apply lemmatization
lemmatizer = WordNetLemmatizer()
print('\nLemmatization output')

count = 1
for word, tag in pos_tag(word_tokenize(google_wiki_text)):
    wntag = tag[0].lower()
    wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
    lemma = lemmatizer.lemmatize(word, wntag) if wntag else word
    print(word + ':' + lemma)

    count += 1
    if count == 50:
        break

# Apply trigrams
print('\nTrigrams output')
trigrams = list(everygrams(tokens, min_len=3, max_len=3))
print(trigrams[:25])

# Apply named entity recognition
print('\nNamed Entity Recognition output')


def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    return continuous_chunk


print(get_continuous_chunks(google_wiki_text))

# End of parts 1, 2, and 3


# Start of part 4


twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# single gram MultinomialNB
tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print('\nMultinomialNB accuracy\n', score)

# single gram KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print('\nKNeighborsClassifier accuracy\n', score)

# bigram MultinomialNB
tfidf_Vect = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print('\nMultinomialNB bigram accuracy\n', score)

# bigram MultinomialNB with english stop words
tfidf_Vect = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print('\nMultinomialNB bigram accuracy with english stop words\n', score)

# End of part 4
