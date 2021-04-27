# the problem of perceptron classifier
from sklearn.linear_model import perceptron
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

categories = ['alt.atheism', 'sci.med']
train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
perceptron = perceptron.Perceptron(max_iter=100)
cv = CountVectorizer()
x_train_counts = cv.fit_transform(train.data)
tfidf_tf = TfidfTransformer()
x_train_tfidf = tfidf_tf.fit_transform(x_train_counts)

perceptron.fit(x_train_tfidf, train.target)

test_docs = ['Religion is widespread, even in modern times',
             'His kidney failed', 'The pope is a controversial leader',
             'White blood cells fight off infections',
             'The reverend had a heart attack in church']


x_test_counts = cv.transform(test_docs)
x_test_tfidf = tfidf_tf.transform(x_test_counts)

pred = perceptron.predict(x_test_tfidf)

for doc, category in zip(test_docs, pred):
    print('%r => %s' % (doc, train.target_names[category]))
    help(perceptron)


def dot_product_vec(a, b):
    return(sum([a[i] * b[i] for i in range(len(a))]))


a = [1, 2, 3, 4]
b = [1, 2, 5, 4]
b = [a[i] + 3 for i in range(len(a))]
dot_product_vec(a, b)


def delta(x, y):
    if x == y:
        return 1
    if x != y:
        return 0


def IB1(a, b):
    return sum([delta(a[i], b[i]) for i in range(len(a))])


IB1(a, b)
