from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from statistics import mode
import pickle


class VotingClassfier(ClassifierI):
    def __init__(self, classifiers, word_features):
        self.classifiers = classifiers
        self.word_features = word_features

    def find_features(self, document):
        words = set(document)
        features = {}
        for w in self.word_features:
            features[w] = (w in words)
        return features

    def classifi(self, review):
        votes = []
        for c in self.classifiers:
            v = c.classify(self.find_features(word_tokenize(review)))
            votes.append(v)
        return mode(votes)

    def confidence(self, review):
        votes = []
        for c in self.classifiers:
            v = c.classify(self.find_features(word_tokenize(review)))
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        return choice_votes / len(votes)


# Word Features
word_features_pickle = open("pickleDocuments/reviewDocumentFeatures.pickle", "rb")
word_features = pickle.load(word_features_pickle)
word_features_pickle.close()

# Algorithems
original_naive_bays_pickle = open("pickleAlgos/original_naive_bays_classifier.pickle", "rb")
original_naive_bays = pickle.load(original_naive_bays_pickle)
original_naive_bays_pickle.close()

multinomial_naive_bays_pickle = open("pickleAlgos/multinomial_naive_bays_classifier.pickle", "rb")
multinomial_naive_bays_classifier = pickle.load(multinomial_naive_bays_pickle)
multinomial_naive_bays_pickle.close()

bernoulli_naive_bays_pickle = open("pickleAlgos/bernoulli_naive_bays_classifier.pickle", "rb")
bernoulli_naive_bays_classifier = pickle.load(bernoulli_naive_bays_pickle)
bernoulli_naive_bays_pickle.close()

logistic_regression_pickle = open("pickleAlgos/logistic_regression_classifier.pickle", "rb")
logistic_regression_classifier = pickle.load(logistic_regression_pickle)
logistic_regression_pickle.close()

SGDClassifier_pickle = open("pickleAlgos/SGDClassifier_classifier.pickle", "rb")
SGDClassifier_classifier = pickle.load(SGDClassifier_pickle)
SGDClassifier_pickle.close()

linearSVC_pickle = open("pickleAlgos/linearSVC_classifier.pickle", "rb")
linearSVC_classifier = pickle.load(linearSVC_pickle)
linearSVC_pickle.close()


def sentiment(review):
    abc = VotingClassfier([original_naive_bays,
                           multinomial_naive_bays_classifier,
                           bernoulli_naive_bays_classifier,
                           logistic_regression_classifier,
                           SGDClassifier_classifier,
                           linearSVC_classifier], word_features)
    return abc.classifi(review), abc.confidence(review)

#
# print(sentiment("I love this car. This is so beautiful and fast.It has so many gears either"))
# print(sentiment(""))
# print(sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
# print(sentiment(
#     "This movie was utter junk. There were absolutely 0 pythons. I don't see what the point was at all. Horrible movie, 0/10"))
