# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pnd    # manipulation et analyse
import os
import datetime
from mod.log.gap_mod_log import Log
from mod.ia.gap_mod_ia_modele import IaModele
from mod.ia.gap_mod_ia_modele_svm import IaModeleSVM

"""nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')"""

# ----------------------------------------------------------------------------------------------------------------------
# VARIABLES GLOBALES
# ----------------------------------------------------------------------------------------------------------------------

dicModeles = {"RLM": {"NOM": "Régression Linéaire Multiple", "FICHIER": "reg_lineaire",      "REGRESS": 1, "CLASSIF": 0, "LANGAGE": 0},
              "RLG": {"NOM": "Régression Logistique",        "FICHIER": "reg_log",           "REGRESS": 0, "CLASSIF": 1, "LANGAGE": 0},
              "RDF": {"NOM": "Random Forest",                "FICHIER": "random_forest",     "REGRESS": 1, "CLASSIF": 1, "LANGAGE": 0},
              "ARD": {"NOM": "Arbre de Décision",            "FICHIER": "arbre_decis",       "REGRESS": 1, "CLASSIF": 1, "LANGAGE": 0},
              "KNB": {"NOM": "K Neighbors",                  "FICHIER": "k_neighbors",       "REGRESS": 1, "CLASSIF": 1, "LANGAGE": 0},
              "SVM": {"NOM": "Machine Vecteurs de Supports", "FICHIER": "vecteurs_supports", "REGRESS": 0, "CLASSIF": 0, "LANGAGE": 1},
              "GBT": {"NOM": "Gradient Boosting",            "FICHIER": "gradient_boosting", "REGRESS": 0, "CLASSIF": 0, "LANGAGE": 0},
              "TFW": {"NOM": "Tensor Flow",                  "FICHIER": "tensor_flow",       "REGRESS": 0, "CLASSIF": 0, "LANGAGE": 0}}

lstColsPerfs = ['etude','date','var_predict', 'code_modele', 'nom_modele', 'precision_apprent','precision_valid']

# ----------------------------------------------------------------------------------------------------------------------
# CLASSE "AG_IA_Modele" : modélisation à partir d'une seule variable prédictive
# ----------------------------------------------------------------------------------------------------------------------

class IaModelisation:

# Attributs

    # data
    dtfData = pnd.DataFrame()
    dtfPerf = pnd.DataFrame(columns=lstColsPerfs)
    dtfPerfs = pnd.DataFrame(columns=lstColsPerfs)
    dtfClasses = pnd.DataFrame()
    sPrefixeCat = ""
    sPrefixePredict = ""

    # étude
    sEtude = ""
    sDate = ""

    # modèle
    sMethode = ""
    sNomPrediction = ""
    sCodeModele = ""
    sNomModele = ""

    # paramétrages
    dRatioValNull = 0.0
    dRatioJeuTest = 0.0
    dPrecisionApprentissage = 0.0
    dPrecisionValidation = 0.0

    # résultats
    sDossier = ""
    sFichier = ""
    ficLog = ""
    log = Log
    iIndent = 0

# Constructeur
    def __init__(self, sEtude, dtfData, log, sPrefixeCat="lib_", sPrefixePredict="predict_", dRatioValNull=0.05,
                 dRatioJeuTest=0.2, sDossier="resultats", sFichier="", iIndent=0):

        self.sEtude = sEtude
        self.sDate = str(datetime.date.today().isoformat())

        self.dtfData = dtfData
        self.dtfDataBrut = dtfData
        self.sPrefixeCat = sPrefixeCat
        self.sPrefixePredict = sPrefixePredict
        self.dRatioValNull = dRatioValNull
        self.dRatioJeuTest = dRatioJeuTest

        # création du dossier spécifique
        self.sDossier = sDossier + "/" + self.sEtude + "/" + self.sDate
        os.makedirs(self.sDossier, exist_ok=True)
        if sFichier != "" : self.sFichier = sFichier
        self.log = log
        self.iIndent = iIndent

# Affichage des résultats

    def fnc_affichage_resultats(self, sTitre="Résultats", dPerfMax=0.0):

        for sEtude in self.dtfPerf.etude.unique():
            self.log.w(sTitre + " - Etude \""+sEtude+"\"",iIndent=self.iIndent, bPuce=True)

            dtfEtude = self.dtfPerf[self.dtfPerf["etude"]==sEtude]

            for sDate in dtfEtude.date.unique():

                dtfDate = dtfEtude[dtfEtude["date"]==sDate]

                if dPerfMax > 0:
                    dtfDate = dtfDate.sort_values('precision_valid')
                    dtfDate = dtfDate[dtfDate['precision_valid'] < dPerfMax]

                # self.log.i(sNom="dtfDate", dtfData=dtfDate)

                dtfCols = pnd.DataFrame([["var_predict", "txt", 30, 0],
                                         ["nom_modele", "txt", 30, 0],
                                         ["precision_apprent", "num", 30, 2],
                                         ["precision_valid", "num", 30, 2]], columns=["nom", "format", "long", "precis"])


                self.log.t(dtfData=dtfDate, dtfCols=dtfCols, iIndent=self.iIndent + 1)


# Préparation des données

    def fnc_prepa_cat(self):

        self.log.w("[fnc_prepa_cat]", iIndent=self.iIndent + 2, bPuce=True)

        self.log.w("Détection catégories & encodage", iIndent=self.iIndent + 3)
        for col in self.dtfData.columns:
            if col.startswith(self.sPrefixeCat):
                dtfComptage = self.dtfData[col].value_counts().to_frame()
                self.log.w("%s (%s modalités)" % (col, len(dtfComptage)), iIndent=self.iIndent + 4)
                dtfComptage = dtfComptage.rename(columns={col:'nb'})
                dtfComptage[col] = dtfComptage.index
                self.log.t(dtfData=dtfComptage,
                           dtfCols= pnd.DataFrame([ [col, "txt", 30, 0],
                                                    ["nb", "num", 15, 0]],
                                                  columns=["nom", "format", "long", "precis"]),
                           iIndent=self.iIndent + 4)


                self.dtfData = self.dtfData.join(pnd.get_dummies(self.dtfData[col], prefix=col, prefix_sep="_"))
                del self.dtfData[col]

                self.log.n()

        # Encodage des classifications
        if (self.sMethode == "L" or self.sMethode == "C") and (len(self.dtfData)>100):
            self.log.w("Encodage des classifications cibles", iIndent=self.iIndent + 3)
            for col in self.dtfData.columns:
                if col.startswith(self.sPrefixePredict) and self.dtfData[col].dtypes == 'object':
                    dtfComptage = self.dtfData[col].value_counts().to_frame()
                    #dtfComptage = dtfComptage.sort_values(by='lib_classe')
                    self.log.n()
                    self.log.w("%s (%s modalités)" % (col, len(dtfComptage)), iIndent=self.iIndent + 4)
                    dtfComptage["num_classe"] = pnd.RangeIndex(0, len(dtfComptage))
                    dtfComptage["lib_classe"] = dtfComptage.index
                    dtfComptage = dtfComptage.rename(columns={col: 'nb'})
                    dtfComptage = dtfComptage.reindex(columns=['num_classe','lib_classe','nb'])
                    self.log.t(dtfData=dtfComptage,
                               dtfCols=pnd.DataFrame([["num_classe", "num", 15, 0],
                                                      ["lib_classe", "txt", 50, 0],
                                                      ["nb", "num", 15, 0]],
                                                     columns=["nom", "format", "long", "precis"]),
                               iIndent=self.iIndent + 4)
                    self.dtfClasses = dtfComptage
                    dtfComptage.to_csv(self.sDossier + "/" + col + ".cla", sep=";", encoding="UTF8")
                    self.dtfData = self.dtfData.merge(dtfComptage,left_on=col,right_on="lib_classe")
                    del self.dtfData[col]
                    del self.dtfData["lib_classe"]
                    del self.dtfData["nb"]
                    self.dtfData = self.dtfData.rename(columns={"num_classe":col})
                    self.log.i("dtfData",self.dtfData)

        self.log.n()


    def fnc_prepa_null(self):

        self.log.w("[fnc_prepa_null]", iIndent=self.iIndent + 2, bPuce=True)

        # Comptage des valeurs manquantes
        lstNbNulls = self.dtfData.isnull().sum()

        if lstNbNulls.sum()>0:

            self.log.w("Détection des valeurs manquantes", iIndent=self.iIndent + 3)
            lstNbNulls = lstNbNulls.sort_values(ascending=False)
            for col in lstNbNulls[lstNbNulls != 0].index:
                self.log.w(col + " (" + str(lstNbNulls[col]) + " valeurs)", iIndent=self.iIndent + 4)
            self.log.n()

            # Suppression des caractéristiques avec trop de nulls
            nbValNullMax = round(len(self.dtfData) * self.dRatioValNull)
            self.log.w("Suppression des caractéristiques avec plus de "
                       + str(100 * self.dRatioValNull) + "% de valeurs manquantes (" + str(nbValNullMax) + " valeurs)",
                       iIndent=self.iIndent + 3)
            for col in lstNbNulls[lstNbNulls > nbValNullMax].index:
                self.log.w(col + " (" + str(lstNbNulls[col]) + " valeurs)", iIndent=self.iIndent + 4)
                del self.dtfData[col]
            self.log.n()

            self.log.w("Complétion des caractéristiques numériques (valeur la plus fréquente)", iIndent=self.iIndent + 3)
            lstNbNulls = self.dtfData.isnull().sum()
            lstNbNulls = lstNbNulls.sort_values(ascending=False)
            for col in lstNbNulls[lstNbNulls > 0].index:
                if (self.dtfData[col].dtypes == 'int') or (self.dtfData[col].dtypes == 'float'):
                    valMode = (self.dtfData[col].mode())[0]
                    self.log.w(col + " (" + str(valMode) + ")", iIndent=self.iIndent + 4)
                    self.dtfData[col] = self.dtfData[col].fillna(valMode)
            self.log.n()


    def fnc_prepa_date(self):

        self.log.w("[fnc_prepa_date]", iIndent=self.iIndent + 2, bPuce=True)

        if len(self.dtfData.columns[self.dtfData.dtypes == 'date'])>0:

                self.log.w("Suppression des caractéristiques de type date", iIndent=self.iIndent + 3)

                for col in self.dtfData.columns:
                    if self.dtfData[col].dtypes == 'date':
                        self.log.w(col, iIndent=self.iIndent + 4)
                        del self.dtfData[col]
                self.log.n()

    def fnc_prepa_predict(self, sPredict):

        # Suppression des autres variables prédictives
        self.log.w("[fnc_prepa_predict]", iIndent=self.iIndent + 2, bPuce=True)
        self.log.w("Suppression des autres variables prédictives", iIndent=self.iIndent + 3)
        for col in self.dtfData.columns:
            if col.startswith(self.sPrefixePredict) and col!=sPredict:
                self.log.w(col, iIndent=self.iIndent + 4)
                del self.dtfData[col]
        self.log.n()

        # suppression des lignes sans variable prédictive
        iNbLignes = len(self.dtfData.loc[self.dtfData[sPredict].isnull(),])
        self.log.w("Suppression des lignes sans variable prédictive", iIndent=self.iIndent + 3)
        self.log.w("%s lignes supprimées" % iNbLignes, iIndent=self.iIndent + 4)
        self.dtfData = self.dtfData.loc[~self.dtfData[sPredict].isnull(),]
        self.log.n()

        # dataset
        self.log.i("dtfData", self.dtfData, iIndent=self.iIndent + 3)
        self.log.n()

    def fnc_prepa(self, sPredict):

        # Suppression des autres variables prédictives
        self.fnc_prepa_predict(sPredict=sPredict)

        # Gestion des valeurs manquantes
        self.fnc_prepa_null()

        # Gestion des formats dates
        self.fnc_prepa_date()

        # Gestion des catégories
        self.fnc_prepa_cat()

        #self.log.i(sNom="dtfData", dtfData=self.dtfData, iIndent=self.iIndent + 2)
        #self.dtfData.to_csv("resultats/dtfData.csv", sep=";", encoding="UTF8")

# Modélisation élémentaire

    def fnc_modelisation_recurs(self, sCodeModele,  sNomPrediction) :

        self.sNomPrediction = sNomPrediction
        #os.makedirs(self.sDossier + "/" + self.sNomPrediction, exist_ok=True)

        # Construction des dataframes
        self.log.w("Datasets d'entrée", iIndent=self.iIndent + 2, bPuce=True)
        dtfObserv = pnd.DataFrame()
        dtfPredict = pnd.DataFrame()
        for col in self.dtfData.columns:
            if col.startswith(self.sPrefixePredict):
                dtfPredict[col] = self.dtfData[col]
            else:
                dtfObserv[col] = self.dtfData[col]
        self.log.i(sNom="dtfObserv", dtfData=dtfObserv, iIndent=self.iIndent + 3)
        self.log.i(sNom="dtfPredict", dtfData=dtfPredict, iIndent=self.iIndent + 3)

        if dicModeles[sCodeModele]:
            self.sCodeModele = sCodeModele
            self.sNomModele = dicModeles[sCodeModele]["NOM"]
            self.sFichier = dicModeles[sCodeModele]["FICHIER"]
            self.log.w(">>> Modèle \"%s\"" % self.sNomModele,iIndent=self.iIndent+2)
            self.log.n()

        mod = IaModele
        if self.sCodeModele == "SVM":
            mod = IaModeleSVM(sEtude=self.sEtude,
                              dtfObserv=dtfObserv,
                              dtfPredict=dtfPredict,
                              dtfClasses=self.dtfClasses,
                              log=self.log,
                              dRatioJeuTest=0.2,
                              sDossier=self.sDossier,
                              sFichier="mod_svm",
                              iIndent=self.iIndent+3)

        # Infos modèles
        mod.sCodeModele = self.sCodeModele
        mod.sNomModele = dicModeles[self.sCodeModele]['NOM']

        # Préparation des jeux de données
        mod.fnc_prepa_data()

        # entrainement du modèle
        mod.fnc_entrainement()

        # évaluation du modèle
        self.dtfPerf = pnd.concat([self.dtfPerf,mod.fnc_evaluation()], ignore_index=True)

        # export du modèle
        mod.fnc_export_mod(sUrl=self.sDossier + "/" + self.sNomPrediction + "_" + self.sCodeModele.lower() +".mod")

        self.log.n()
        return self.dtfPerf

# Modélisation pour chaque modèle compatible, et chaque variable à prédire

    def fnc_modelisations(self,sMethode):

        self.sMethode = sMethode
        dtfData = self.dtfData.copy()

        os.makedirs(self.sDossier, exist_ok=True)
        fichiers = os.listdir(self.sDossier)
        """for i in range(0, len(fichiers)):
            os.remove(self.sDossier + "/" + fichiers[i])"""

        self.log.w("[fnc_modelisations] sMethode=%s / dRatioValNull=%s / dRatioJeuTest=%s" % (sMethode,str(self.dRatioValNull),str(self.dRatioJeuTest)), iIndent=self.iIndent, bPuce=True)

        self.log.w("Liste des modèles",iIndent=self.iIndent+1)
        for sMod in dicModeles:
            sRegr = " "
            sClass = " "
            sLangue = " "
            if dicModeles[sMod]["REGRESS"] == 1:
                sRegr = "R"
            if dicModeles[sMod]["CLASSIF"] == 1:
                sClass = "C"
            if dicModeles[sMod]["LANGAGE"] == 1:
                sLangue = "L"
            self.log.w(sMod + " [" + sRegr + "," + sClass + "," + sLangue + "] : "+dicModeles[sMod]["NOM"],iIndent=self.iIndent+2)
        self.log.n()

        lstColsPredict = []
        for col in self.dtfData.columns:
            if col.startswith(self.sPrefixePredict):
                lstColsPredict.append(col)

        for sPredict in lstColsPredict :

            self.log.w("Prédiction \"" + sPredict + "\"",iIndent=self.iIndent+1, bPuce=True)

            self.fnc_prepa(sPredict=sPredict)

            if len(self.dtfData)>100:
                for sMod in dicModeles:
                    if          (sMethode == "R" and dicModeles[sMod]["REGRESS"] == 1) \
                            or  (sMethode == "C" and dicModeles[sMod]["CLASSIF"] == 1)\
                            or  (sMethode == "L" and dicModeles[sMod]["LANGAGE"] == 1):

                        dtfPerf = self.fnc_modelisation_recurs(sCodeModele=sMod, sNomPrediction=sPredict)
                        self.dtfPerfs = pnd.concat([self.dtfPerfs, dtfPerf], ignore_index=True)

            self.dtfData = dtfData.copy()


        self.dtfPerfs.to_csv(self.sDossier + "/" + self.sEtude + ".prf", sep=";", encoding="UTF8")

        return self.dtfPerfs

