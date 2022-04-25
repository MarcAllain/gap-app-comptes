# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------
from datetime import datetime

import pandas as pnd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from mod.ia.gap_mod_ia_modele import IaModele, lstColsPerfs
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# ----------------------------------------------------------------------------------------------------------------------
# VARIABLES GLOBALES
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
# CLASSE "IaModeleSVM" : Modèle "Machine Vecteurs de Supports"
# ----------------------------------------------------------------------------------------------------------------------


class IaModeleSVM(IaModele):

# Attributs


    """# Constructeur

        def __init__(self, sEtude, dtfObserv, dtfPredict, log, dRatioJeuTest=0.2, sDossier="resultats", sFichier="", iIndent=0):

            super().__init__(sEtude=sEtude, dtfObserv=dtfObserv, dtfPredict=dtfPredict, log=log,
                             dRatioJeuTest=dRatioJeuTest, sDossier=sDossier, sFichier=sFichier, iIndent=iIndent)

            self.sCodeModele = "SVM"
            self.sNomModele = "Machine Vecteurs de Supports"""

# Préparation des jeux de données

    def fnc_prepa_data(self) :

        # tObserv = self.dtfObserv['txt_tweet'].values
        # tPredict = self.dtfPredict['predict_croyance'].values

        # Préparation du texte
        self.fnc_prepa_nlp()

        # conversion des dataframes en vecteurs de dimensions (n,)
        tObserv = self.dtfObserv.to_numpy().reshape(len(self.dtfObserv))
        tPredict = self.dtfPredict.to_numpy().reshape(len(self.dtfPredict))
        self.sNomPrediction = self.dtfPredict.columns[0]

        # construction des jeux d'entrainement/validation
        self.log.w("Jeux d'entrainement / validation",iIndent=self.iIndent)
        self.dtfXtrain, self.dtfXtest, self.dtfYtrain, self.dtfYtest = train_test_split(tObserv, tPredict,
                                                                                        test_size=self.dRatioJeuTest,
                                                                                        random_state=42)


# Entrainement

    def fnc_entrainement(self):

        self.log.w("Entrainement", iIndent=self.iIndent)

        # pipeline
        self.log.w("Création du pipeline", iIndent=self.iIndent + 1)
        pipeline = Pipeline([('frequence', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('algorithme', SVC(kernel='linear', C=1))])

        # optimisation des paramètres
        self.log.w("Optimisation des paramètres", iIndent=self.iIndent + 1)
        penalite = {'algorithme__C': range(1, 10)}
        oModOptim = GridSearchCV(pipeline, penalite, cv=2)
        oModOptim.fit(self.dtfXtrain, self.dtfYtrain)
        self.log.w("Meilleurs params : %s" % str(oModOptim.best_params_['algorithme__C']), iIndent=self.iIndent + 2)

        # apprentissage
        self.log.w("Apprentissage", iIndent=self.iIndent + 1)
        pipeline = Pipeline([('frequence', CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('algorithme', SVC(kernel='linear', C=oModOptim.best_params_['algorithme__C'], gamma='auto'))])
        self.oModele = pipeline.fit(self.dtfXtrain, self.dtfYtrain)

        # évaluation
        self.log.w("Evaluation", iIndent=self.iIndent + 1)
        self.dPrecisionApprentissage = pipeline.score(self.dtfXtrain, self.dtfYtrain)
        self.dPrecisionValidation = pipeline.score(self.dtfXtest, self.dtfYtest)

        """self.log.w("Init modèle",iIndent=self.iIndent+3)
        if self.sCodeModele == "RLM":
            algo = LinearRegression()
        elif self.sCodeModele == "ARD":
            algo = DecisionTreeRegressor()
        elif self.sCodeModele == "RDF":
            algo = RandomForestRegressor()
        elif self.sCodeModele == "RLG":
            algo = LogisticRegression(max_iter=10)
        elif self.sCodeModele == "KNB":
            algo = KNeighborsClassifier()
        elif self.sCodeModele == "GBT":
            algo = GradientBoostingClassifier()
        elif self.sCodeModele == "TFW":
            algo = AG_IA_Modele_TFW(12,1,1,1000,0.1)"""

#Evaluation

    def fnc_evaluation(self):

        # Evaluations
        self.log.w("Evaluations",iIndent=self.iIndent)

        # print(classification_report(self.dtfYtest, self.oModele.predict(self.dtfXtest), digits=4))
        return pnd.DataFrame([[ self.sEtude,
                                self.sDate,
                                self.sNomPrediction,
                                self.sCodeModele,
                                self.sNomModele,
                                100*self.dPrecisionApprentissage,
                                100*self.dPrecisionValidation]],
                               columns=lstColsPerfs)

    def fnc_preparation_texte(self, sTexte):
        sTexte = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', sTexte)
        sTexte = re.sub('@[^\s]+', 'USER', sTexte)
        sTexte = sTexte.lower().replace("ë", "e")
        sTexte = re.sub('[^a-zA-Z1-9]', ' ', sTexte)
        sTexte = re.sub(' +', ' ', sTexte)
        return sTexte.strip()

    def fnc_prepa_nlp(self):

        self.log.w("[fnc_prepa_nlp]", iIndent=self.iIndent + 2, bPuce=True)

        self.log.w("Préparation des données de type texte (nlp)", iIndent=self.iIndent + 3)
        for col in self.dtfObserv.columns:
            if self.dtfObserv[col].dtypes == 'object':
                self.log.w(col, iIndent=self.iIndent + 4)

                # Normalisation
                self.log.w("Normalisation", iIndent=self.iIndent + 5)
                # self.dtfObserv[sColonne] = self.dtfObserv[sColonne].apply(self.fnc_preparation_texte)
                # TODO vérif paramétrage

                # stopwords
                self.log.w("Suppression des \"stop words\"", iIndent=self.iIndent + 5)
                stopWords = stopwords.words('french')
                self.dtfObserv[col] = self.dtfObserv[col].apply(lambda message: ' '.join([      mot for mot in str(message).split()
                                                                                                if mot not in (stopWords)]))

                # stemmisation
                self.log.w("Stemmisation", iIndent=self.iIndent + 5)
                stemmer = SnowballStemmer('french')
                self.dtfObserv[col] = self.dtfObserv[col].apply(lambda message: ' '.join([  stemmer.stem(mot)
                                                                                            for mot in message.split(' ')]))

                # lemmatisation
                self.log.w("Lemmatisation", iIndent=self.iIndent + 5)
                lemmatizer = WordNetLemmatizer()
                self.dtfObserv[col] = self.dtfObserv[col].apply(lambda message: ' '.join([  lemmatizer.lemmatize(mot)
                                                                                            for mot in message.split(' ')]))

        self.log.n()

# Prédire

    def fnc_prediction(self, dtfObserv):

        self.log.w("Prédiction", iIndent=self.iIndent)
        # TODO Ajouter une colonne avec le texte brut dans dtfResult (pour contrôle visule)

        # Préparation des données
        self.dtfObserv = dtfObserv
        self.fnc_prepa_nlp()
        #self.log.i("dtfObserv", self.dtfObserv,iIndent=self.iIndent+1)

        # Prédiction
        self.dtfPredict = self.oModele.predict(self.dtfObserv.to_numpy().reshape(len(self.dtfObserv)))

        # Formatage dtf prédiction
        self.dtfPredict = pnd.DataFrame(self.dtfPredict, columns={self.sNomPrediction})
        self.dtfPredict = self.dtfPredict.merge(self.dtfClasses, left_on=self.sNomPrediction, right_on="num_classe", how="left")
        self.dtfPredict = self.dtfPredict.rename(columns={self.sNomPrediction:"%s_num" % self.sNomPrediction,
                                                          "lib_classe":self.sNomPrediction,})
        del self.dtfPredict["nb"]
        del self.dtfPredict["num_classe"]
        #self.log.i("dtfPredict", self.dtfPredict, iIndent=self.iIndent + 1)
        print()

        # Résultat
        dtfResultats = self.dtfObserv
        dtfResultats["index"] = dtfResultats.index
        dtfResultats.reset_index(inplace=True, drop=True)
        dtfResultats = dtfResultats.join(self.dtfPredict)
        dtfResultats = dtfResultats.set_index(dtfResultats["index"])
        del dtfResultats['index']

        # Affichage
        self.log.i("dtfResultats", dtfResultats, iIndent=self.iIndent + 1)
        dtfCols = pnd.DataFrame([[self.dtfObserv.columns[0],        "txt", 100, 0],
                                 ["%s_num" % self.sNomPrediction,    "num", 25, 0],
                                 [self.sNomPrediction,              "txt", 25, 2]], columns=["nom", "format", "long", "precis"])

        self.log.t(dtfData=dtfResultats.head(10), dtfCols=dtfCols, iIndent=self.iIndent + 2)

        return dtfResultats
