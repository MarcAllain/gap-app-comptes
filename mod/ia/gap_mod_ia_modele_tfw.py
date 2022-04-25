# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
"""     
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed 
3 = INFO, WARNING, and ERROR messages are not printed
"""


# ----------------------------------------------------------------------------------------------------------------------
# VARIABLES GLOBALES
# ----------------------------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------------------
# CLASSE "AG_IA_Modele_TFW" : gestion d'un modèle Tensor Flow
# ----------------------------------------------------------------------------------------------------------------------

class AG_IA_Modele_TFW:

    # Attributs

    nbEpoques = 0
    nbNeuronesEntree = 0
    nbNeuronesSortie = 0
    nbNeuronesCachee = 0
    tauxApprentissage = 0.0

    nrnEntree = tf.placeholder
    nrnSortie = tf.placeholder
    poids = {}
    biais = {}

    fPrediction = tf.sigmoid
    fErreur = tf.reduce_sum
    fOptimisation = tf.train.GradientDescentOptimizer
    fPrecision = tf.reduce_mean
    tMSE = []

    session = tf.Session


    # Constructeur
    def __init__(self,  nbNeuronesEntree=1, nbNeuronesSortie=1, nbNeuronesCachee=1, nbEpoques = 300, tauxApprentissage=0.1):

        self.nbNeuronesEntree = nbNeuronesEntree
        self.nbNeuronesSortie = nbNeuronesSortie
        self.nbNeuronesCachee = nbNeuronesCachee
        self.nbEpoques = nbEpoques
        self.tauxApprentissage = tauxApprentissage

        print("     - création des couches d'entrée/sortie")
        self.nrnEntree = tf.placeholder(tf.float32, [None, self.nbNeuronesEntree])
        self.nrnSortie = tf.placeholder(tf.float32, [None, self.nbNeuronesSortie])

        print("     - init poids / biais")
        self.poids = {  'entree_cachee': tf.Variable(tf.random_normal([self.nbNeuronesEntree, self.nbNeuronesCachee]), tf.float32),
                        'cachee_sortie': tf.Variable(tf.random_normal([self.nbNeuronesCachee, self.nbNeuronesSortie]), tf.float32)        }

        self.biais = {  'entree_cachee': tf.Variable(tf.zeros([self.nbNeuronesCachee]), tf.float32),
                        'cachee_sortie': tf.Variable(tf.zeros([self.nbNeuronesSortie]), tf.float32)        }

        print("     - création du réseau de neurones")
        self.fPrediction =       tf.sigmoid(    tf.matmul(self.nrnEntree, self.poids['entree_cachee'])
                                                + self.biais['entree_cachee'])

        self.fPredictionCachee = tf.sigmoid(    tf.matmul(self.fPrediction,  self.poids['cachee_sortie'])
                                                + self.biais['cachee_sortie'])


    # Méthodes
    def fit(self, X_APPRENTISSAGE, Y_APPRENTISSAGE):

        Z = np.zeros((1, Y_APPRENTISSAGE.shape[0]))
        for i in range(Y_APPRENTISSAGE.shape[0]):
            Z[0, i] = Y_APPRENTISSAGE[i]
        Z = Z.T
        Y_APPRENTISSAGE = Z

        print(">>> DEBUG")
        print("Type Y=" + str(type(Y_APPRENTISSAGE)) + " (" + str(np.shape(Y_APPRENTISSAGE)) + ")")
        print(Y_APPRENTISSAGE)
        print("<<< DEBUG")

        print("     - fonction d'évaluation de l'erreur")
        self.fErreur = tf.reduce_sum(tf.pow(Y_APPRENTISSAGE - self.fPredictionCachee, 2))

        print("     - fonction d'optimisation")
        self.fOptimisation = tf.train.GradientDescentOptimizer(learning_rate=self.tauxApprentissage).minimize(
            self.fErreur)

        print("     - apprentissage :")
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

        for i in range(self.nbEpoques):

            self.session.run(    self.fOptimisation,
                                 feed_dict = { self.nrnEntree: X_APPRENTISSAGE,
                                               self.nrnSortie: Y_APPRENTISSAGE}   )

            MSE = self.session.run(    self.fErreur,
                                       feed_dict = { self.nrnEntree: X_APPRENTISSAGE,
                                                     self.nrnSortie: Y_APPRENTISSAGE}     )
            self.tMSE.append(MSE)

            if i ==0 or i+1==self.nbEpoques:
                print("          [" + str(i) + " / " + str(self.nbEpoques) + "] - MSE = " + str(MSE))

    def predict(self, X):
        return self.session.run(    self.fPredictionCachee,
                                    feed_dict={self.nrnEntree: X})

    def score(self,dtfX, dtfY):

        nbObserv = 0
        nbObservOK = 0

        Z = np.zeros((1, dtfY.shape[0]))
        for i in range(dtfY.shape[0]):
            Z[0, i] = dtfY[i]
        Z = Z.T
        dtfY = Z

        #print("     - fonction de précision")
        #prediction = tf.argmax(self.fPredictionCachee, 1)
        #predictionCible = tf.argmax(self.nrnSortie, 1)
        #self.fPrecision = tf.reduce_mean(tf.cast(tf.equal(prediction, predictionCible), tf.float32))

        fPrecision = tf.reduce_mean(tf.cast(tf.equal(self.fPredictionCachee, self.nrnSortie), tf.float32))

        for i in range(0, len(dtfX)):

            sI = ("00" + str(i + 1))[-2:]

            predictionRun = self.predict([dtfX[i]])

            precisionRun = self.session.run(    fPrecision,
                                                feed_dict={self.nrnEntree: [dtfX[i]], self.nrnSortie: [dtfY[i]]})

            nbObserv = nbObserv + 1
            sErreur=""
            if round(predictionRun[0][0], 0) == round(dtfY[i][0], 0): nbObservOK = nbObservOK + 1
            else :   sErreur = " ! Erreur !"

            print("     [" + sI + "] observ:" + str(dtfX[i]) + ", attendu:"+str(dtfY[i])+", predict:"
                  + str(predictionRun[0]) + ", précision:" + str(precisionRun)+", "+sErreur)

        return round(100 * nbObservOK / nbObserv, 4)
