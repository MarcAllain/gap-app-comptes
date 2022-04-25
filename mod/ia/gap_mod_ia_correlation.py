# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pnd    # manipulation et analyse
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------------------------------------------------------------
# VARIABLES GLOBALES
# ----------------------------------------------------------------------------------------------------------------------

dicCorr = {}
dicCorr["PEA"] = {"NOM":"pearson", "DESCRIPTION":"standard correlation coefficient",    "FICHIER":"corr_pearson"}
dicCorr["KEN"] = {"NOM":"kendall", "DESCRIPTION":"Kendall Tau correlation coefficient", "FICHIER":"corr_kendall"}
dicCorr["SPE"] = {"NOM":"spearman","DESCRIPTION":"Spearman rank correlation",           "FICHIER":"corr_spearman"}

# ----------------------------------------------------------------------------------------------------------------------
# CLASSE "AG_IA_Correlation" : gestion d'une corrélation
# ----------------------------------------------------------------------------------------------------------------------

class IaCorrelation:
    "Gestion d'une corrélation"

# Attributs
    dtfX = pnd.DataFrame
    dtfCorr = pnd.DataFrame
    dtfPredict = pnd.DataFrame
    dtfAPredir = pnd.DataFrame
    sCode = ""
    sNom = ""
    sDossier = ""
    sFichier = ""

# Constructeur
    def __init__(self, dtfX, sCodeMethode="PEA" , sDossier="correlations", sFichier=""):

        if dicCorr[sCodeMethode]:
            self.sCode      = sCodeMethode
            self.sNom       = dicCorr[sCodeMethode]["NOM"]
            self.sFichier   = dicCorr[sCodeMethode]["FICHIER"]

        self.dtfX = dtfX

        self.sDossier = sDossier
        if sFichier != "":
            self.sFichier = sFichier

        os.makedirs(self.sDossier, exist_ok=True)

        # purge préalable des fichiers
        fichiers = os.listdir(self.sDossier)
        for i in range(0, len(fichiers)):
            os.remove(self.sDossier + "/" + fichiers[i])

# Méthodes

    def mCorrelation(self, bClassements = False):

        # Correlation
        print("> Méthode \"" + self.sNom+"\"")
        print("     - Corrélation")
        self.dtfCorr = (self.dtfX).loc[:, :].corr(method=self.sNom, min_periods=1)

        # Enregistrement du fichier
        print("     - Enregistrement (" + self.sDossier + "/" + self.sFichier +".csv)")
        os.makedirs(self.sDossier, exist_ok=True)
        self.dtfCorr.to_csv(self.sDossier + "/" + self.sFichier +".csv", sep=';', encoding="UTF8")

        # Enregistrement d'une image
        print("     - Enregistrement (" + self.sDossier + "/" + self.sFichier + ".png)")
        dtfCorr = self.dtfCorr.copy(True)
        dtfCorr = 100 * dtfCorr
        sns.heatmap(dtfCorr,
                    xticklabels=self.dtfX.columns,
                    yticklabels=self.dtfX.columns, annot=True, fmt="2.0f")
        plt.title("Méthode " + self.sNom)
        plt.savefig(self.sDossier + "/" + self.sFichier + ".png")
        plt.close()

        if bClassements == True:

            # Nettoyage de la matrice
            print("     - Nettoyage")
            dtfCorr = self.dtfCorr.copy(True)

            for col in dtfCorr:
                dtfCorr[col] = round(100*dtfCorr[col], 0)
                dtfCorr[col] = dtfCorr[col].apply(lambda x: 0 if x < 0 else x)

            # Suppression des lignes/colonnes redondantes
            print("     - Suppression des lignes/colonnes redondantes")

            for ligne in dtfCorr:
                if ligne[0:5] == "note_":
                    dtfCorr = dtfCorr.drop([ligne])
                else:
                    dtfCorr = dtfCorr.drop([ligne], axis='columns')

            # Classement des variables prédictives
            print("     - Classement des variables prédictives")
            self.dtfPredict = dtfCorr.copy(True)

            self.dtfPredict['TOTAL'] = 0.0
            for col in self.dtfPredict.columns:
                self.dtfPredict['TOTAL'] = self.dtfPredict['TOTAL'] + self.dtfPredict[col]

            self.dtfPredict = self.dtfPredict.sort_values(by='TOTAL', ascending=False)
            self.dtfPredict.to_csv(self.sDossier + "/" + self.sFichier + "_predict.csv", sep=';', encoding="UTF8")

            # Classement des variables à prédire
            print("     - Classement des variables à prédire")
            self.dtfAPredir = dtfCorr.copy(True)
            self.dtfAPredir = self.dtfAPredir.T

            self.dtfAPredir['TOTAL'] = 0.0
            for col in  self.dtfAPredir.columns:
                self.dtfAPredir['TOTAL'] =  self.dtfAPredir['TOTAL'] +  self.dtfAPredir[col]

            self.dtfAPredir =  self.dtfAPredir.sort_values(by='TOTAL', ascending=False)
            self.dtfAPredir.to_csv(self.sDossier + "/" + self.sFichier +"_apredir.csv", sep=';', encoding="UTF8")

    def mAffichage(self, bSubPlot = False, bClassements = False):

        dtfCorr = self.dtfCorr.copy(True)
        dtfCorr = 100 * dtfCorr
        sns.heatmap(dtfCorr,
                    xticklabels=self.dtfX.columns,
                    yticklabels=self.dtfX.columns, annot=True,fmt="2.0f")
        plt.title("Méthode " + self.sNom)

        if bSubPlot == False:
            print("     - Affichage matrice")
            plt.show()

        if bClassements == True:
            print("     - Affichage classements")
            print()
            print(self.dtfPredict)
            print()
            print(self.dtfAPredir)

"""
            # Affichage des tableaux
            tbl1 = plt.subplot(2,1,1)
            tbl1.axis("off")
            tbl1.table(cellText=dtfVarPredictives.values,
                      rowLabels=dtfVarPredictives.index,
                      colLabels=dtfVarPredictives.columns,
                      cellLoc = 'right', rowLoc = 'center', loc='center')
            plt.title("Variables prédictives")
            
            
            tbl2 = plt.subplot(2,1,2)
            tbl2.axis("off")
            tbl2.table(cellText=dtfVarAPredire.values,
                      rowLabels=dtfVarAPredire.index,
                      colLabels=dtfVarAPredire.columns,
                      cellLoc = 'right', rowLoc = 'center', loc='center')
            plt.title("Variables à prédire")
            
            plt.show()
"""

# ----------------------------------------------------------------------------------------------------------------------
# CLASSE "AG_IA_Correlations" : tests des différentes méthodes de corrélations
# ----------------------------------------------------------------------------------------------------------------------

class AG_IA_Correlations:

    # Attributs
    dtfX = pnd.DataFrame
    corrP = AG_IA_Correlation
    corrK = AG_IA_Correlation
    corrS = AG_IA_Correlation
    sDossier = ""

    # Constructeur
    def __init__(self, dtfX, sDossier="correlations"):
        self.dtfX = dtfX
        self.sDossier = sDossier

        self.corrP = AG_IA_Correlation(self.dtfX, "PEA", self.sDossier)
        self.corrK = AG_IA_Correlation(self.dtfX, "KEN", self.sDossier)
        self.corrS = AG_IA_Correlation(self.dtfX, "SPE", self.sDossier)

        os.makedirs(self.sDossier, exist_ok=True)

        # purge préalable des fichiers
        fichiers = os.listdir(self.sDossier)
        for i in range(0, len(fichiers)):
            os.remove(self.sDossier + "/" + fichiers[i])

    # Méthodes
    def mCorrelations(self):
        self.corrP.mCorrelation()
        print()
        self.corrK.mCorrelation()
        print()
        self.corrS.mCorrelation()
        print()

    def mAffichage(self):
        plt.subplot(3, 1, 1)
        self.corrP.mAffichage(True)
        plt.subplot(3, 1, 2)
        self.corrK.mAffichage(True)
        plt.subplot(3, 1, 3)
        self.corrS.mAffichage(True)
        plt.show()

