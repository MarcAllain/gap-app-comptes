# --------------------------------------------------------------------
# MODULES
# --------------------------------------------------------------------
import nltk
import pandas as pnd
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# --------------------------------------------------------------------
# PARAMETRES
# --------------------------------------------------------------------

# nltk.download()

# --------------------------------------------------------------------
# FONCTIONS
# --------------------------------------------------------------------

def fnNormalisation(message):
    message = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',message)
    message = re.sub('@[^\s]+','USER',message)
    message = message.lower().replace("ë","e")
    message = re.sub('[^a-zA-Z1-9]',' ',message)
    message = re.sub(' +',' ',message)
    return message.strip()

# --------------------------------------------------------------------
# PREPARATION
# --------------------------------------------------------------------

# chargement des données
print("> Chargement")
dtfMessages = pnd.read_csv("rechauffementClimatique.csv", delimiter=";", encoding="UTF8")

# formatage
print("> Formatage")
dtfMessages['CROYANCE'] = (dtfMessages['CROYANCE']=='Yes').astype(int)

# normalisation
print("> Normalisation")
dtfMessages["TWEET"] = dtfMessages["TWEET"].apply(fnNormalisation)

# stopwords
print("> Suppression des \"stop words\"")
stopWords = stopwords.words('english')
dtfMessages["TWEET"] = dtfMessages["TWEET"].apply(lambda message:
                                                            ' '.join([  mot for mot in message.split()
                                                                        if mot not in (stopWords)]))

# stemmisation
print("> Stemmisation")
stemmer = SnowballStemmer('english')
dtfMessages["TWEET"] = dtfMessages["TWEET"].apply(lambda message:
                                                            ' '.join([  stemmer.stem(mot)
                                                                        for mot in message.split(' ')]))

# lemmatisation
print("> Lemmatisation")
lemmatizer = WordNetLemmatizer()
dtfMessages["TWEET"] = dtfMessages["TWEET"].apply(lambda message:
                                                            ' '.join([lemmatizer.lemmatize(mot)
                                                                        for mot in message.split(' ')]))

# --------------------------------------------------------------------
# MODELISATION
# --------------------------------------------------------------------

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(dtfMessages["TWEET"].values,
                                                    dtfMessages["CROYANCE"].values,
                                                    test_size=0.2)

# pipeline
print("> Pipeline apprentissage")
pipeline = Pipeline( [     ('frequence',CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('algorithme',SVC(kernel='linear', C=1))  ])

#optimisation modele
print("> Optimisation modèle")
params = {'algorithme__C':(1,2,3,4,5,6,7,8,9,10,11,12)}
clf = GridSearchCV(pipeline,params,cv=2)
clf.fit(X_TRAIN, Y_TRAIN)
print("     "+str(clf.best_params_))

# apprentissage
print("> Apprentissage")
modele = pipeline.fit(X_TRAIN, Y_TRAIN)

# affichage
print("> Affichage")
print(classification_report(Y_TEST, modele.predict(X_TEST), digits=4))

# performances
print("> Performances")
dPrecisionApprentissage = round(100*modele.score(X_TRAIN, Y_TRAIN),2)
dPrecisionValidation =round(100*modele.score(X_TEST, Y_TEST),2)
print("     Précision apprent. = %s" % dPrecisionApprentissage)
print("     Précision valid. = %s" % dPrecisionValidation)
print()

# --------------------------------------------------------------------
# TEST DU MODELE
# --------------------------------------------------------------------

phrase = "Why should trust scientists with global warming if they didnt know Pluto wasnt a planet !!!"
print("> Phrase : "+phrase)
phrase = fnNormalisation(phrase)
print("> Phrase : "+phrase)
phrase = ' '.join([mot for mot in phrase.split() if mot not in stopWords])
print("> Phrase : "+phrase)
phrase = ' '.join([stemmer.stem(mot) for mot in phrase.split(' ')])
print("> Phrase : "+phrase)
phrase = ' '.join([lemmatizer.lemmatize(mot) for mot in phrase.split(' ')])
print("> Phrase : "+phrase)

prediction = modele.predict([phrase])
print("> Prédiction : "+str(prediction))
if( prediction[0]==0):
    print(">>> Ne croit pas au réchauffement climatique !")
else:
    print(">>> Croit au réchauffement climatique !")

