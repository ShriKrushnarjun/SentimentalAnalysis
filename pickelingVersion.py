import nltk
import re
import random
import pickle
from nltk import SklearnClassifier
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import sentence_polarity
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

filenames = [(filename, category)
             for category in sentence_polarity.categories()
             for filename in sentence_polarity.fileids(category)]

sentences = list()
for file in filenames:
    data = sentence_polarity.raw(file[0])
    for sent in data.split("\n"):
        sentences.append((word_tokenize(sent), file[1]))

random.shuffle(sentences)

save_document = open("pickleDocuments/reviewDocument.pickle", "wb")
pickle.dump(sentences, save_document)
save_document.close()

# All Words containing only adjective and adver and verb
# J = Adjective, V = Verb, R = Adverb
allowed_pos_types = re.compile("J")
all_words = []
for sent in sentences:
    tagged = pos_tag(sent[0])
    for t in tagged:
        if allowed_pos_types.match(t[1][0]) and len(t[1]) > 1:
            all_words.append(t[0])

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())

save_word_features = open("pickleDocuments/reviewDocumentFeatures.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


# Creating features
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


featuresset = [(find_features(rev), category) for (rev, category) in sentences]

training_set = featuresset[100:]
testing_set = featuresset[:100]

save_featuresset = open("pickleDocuments/reviewDocumentFeaturesset.pickle", "wb")
pickle.dump(featuresset, save_featuresset)
save_featuresset.close()

naive_bays_classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bays classifier accuracy :", nltk.classify.accuracy(naive_bays_classifier, testing_set))
print(naive_bays_classifier.show_most_informative_features())

save_original_naive_bays = open("pickleAlgos/original_naive_bays_classifier.pickle", "wb")
pickle.dump(naive_bays_classifier, save_original_naive_bays)
save_original_naive_bays.close()

multinomial_naive_bays_classifier = SklearnClassifier(MultinomialNB())
multinomial_naive_bays_classifier.train(training_set)
print("Multinominal Naive Bays classifier accuracy :", nltk.classify.accuracy(multinomial_naive_bays_classifier, testing_set))

save_multinomial_naive_bays_classifier = open("pickleAlgos/multinomial_naive_bays_classifier.pickle", "wb")
pickle.dump(multinomial_naive_bays_classifier, save_multinomial_naive_bays_classifier)
save_multinomial_naive_bays_classifier.close()

bernoulli_naive_bays_classifier = SklearnClassifier(BernoulliNB())
bernoulli_naive_bays_classifier.train(training_set)
print("Bernoulli Naive Bays classifier accuracy :",
      nltk.classify.accuracy(bernoulli_naive_bays_classifier, testing_set))

save_bernoulli_naive_bays_classifier = open("pickleAlgos/bernoulli_naive_bays_classifier.pickle", "wb")
pickle.dump(bernoulli_naive_bays_classifier, save_bernoulli_naive_bays_classifier)
save_bernoulli_naive_bays_classifier.close()

logistic_regression_classifier = SklearnClassifier(LogisticRegression())
logistic_regression_classifier.train(training_set)
print("Logistic regression classifier accuracy :",
      nltk.classify.accuracy(logistic_regression_classifier, testing_set))

save_logistic_regression_classifier = open("pickleAlgos/logistic_regression_classifier.pickle", "wb")
pickle.dump(logistic_regression_classifier, save_logistic_regression_classifier)
save_logistic_regression_classifier.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGD classifier accuracy :",
      nltk.classify.accuracy(SGDClassifier_classifier, testing_set))

save_SGDClassifier_classifier = open("pickleAlgos/SGDClassifier_classifier.pickle", "wb")
pickle.dump(SGDClassifier_classifier, save_SGDClassifier_classifier)
save_SGDClassifier_classifier.close()

linearSVC_classifier = SklearnClassifier(LinearSVC())
linearSVC_classifier.train(training_set)
print("Liner SVC classifier accuracy :",
      nltk.classify.accuracy(linearSVC_classifier, testing_set))

save_linearSVC_classifier = open("pickleAlgos/linearSVC_classifier.pickle", "wb")
pickle.dump(linearSVC_classifier, save_linearSVC_classifier)
save_linearSVC_classifier.close()

print(multinomial_naive_bays_classifier.classify(
    find_features(word_tokenize("This is just another terifying movie. I don't like this. Stupidity Completely"))))
print(multinomial_naive_bays_classifier.classify(
    find_features(word_tokenize("I love this car. This is so beautiful and fast.It has so many gears either"))))
