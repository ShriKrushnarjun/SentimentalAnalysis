import SentimentAnlyser
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.style as style
from nltk.corpus import movie_reviews
import random

style.use("ggplot")

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)

documents = [' '.join(movie_reviews.words(fileid))
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)


def animate():
    testingdata = documents
    xar = []
    yar = []
    x = 0
    y = 0
    for t in testingdata[:100]:
        res = SentimentAnlyser.sentiment(t)
        x += 1
        if res[0] == "pos":
            y += 1
        elif res[0] == "neg":
            y -= 0.5
        xar.append(x)
        yar.append(y)
        print(res)
    axis.clear()
    axis.plot(xar, yar)
    plt.show()


animate()
