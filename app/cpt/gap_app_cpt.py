# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pnd
import numpy as nmp
import re
from mod.fic.gap_mod_xls import Xls
from mod.log.gap_mod_log import Log
from mod.ia.gap_mod_ia_modelisation import IaModelisation
from mod.ia.gap_mod_ia_modele_svm import IaModeleSVM
import os
import datetime

# ----------------------------------------------------------------------------------------------------------------------
# CLASSE "GapAppCpt : classe générale de l'application "Comptes"
# ----------------------------------------------------------------------------------------------------------------------


class GapAppCpt:

# Attributs

    log = Log
    iIndent = 0
    dtfComptes = pnd.DataFrame()
    dtfHistorique = pnd.DataFrame()
    dtfOperations = pnd.DataFrame()
    dicTypeImport = {"CA_202203": {"cod_banque": "CA", "dte_effet": "01/03/2022"}}

# Constructeur

    def __init__(self, log, iIndent=0):
        self.log = log
        self.iIndent = iIndent

# Méthodes

    def fnc_charger_comptes(self, sUrl):

        self.log.w("[fnc_charger_comptes] sUrl=%s" % sUrl, iIndent=self.iIndent, bPuce=True)

        # Lire fichier de comptes
        xls = Xls(sUrl=sUrl)
        self.dtfHistorique = xls.mChargerOnglet(sOnglet="Détail")

        # Nettoyage colonnes de type "flag"
        self.dtfHistorique["cod_compte"] = self.dtfHistorique["cod_compte"].apply(str)
        self.dtfHistorique["cod_compte"] = self.dtfHistorique["cod_compte"].replace(to_replace='nan', value='')
        self.dtfHistorique["cod_compte"] = self.dtfHistorique["cod_compte"].apply(lambda x: x[0:-2])
        self.dtfHistorique["flg_interco"] = self.dtfHistorique["flg_interco"].map({   'x': 1, nmp.nan: 0}, na_action=None)
        self.dtfHistorique["flg_commun"] = self.dtfHistorique["flg_commun"].map({ 'x': 1, nmp.nan: 0}, na_action=None)
        self.dtfHistorique["flg_ok"] = 1

        #TODO A supprimer à la prochaine exécution car num_operation déja dans le fichier source
        self.dtfHistorique["num_operation"] = self.dtfHistorique.index

        #TODO Prévoir de préfixer tous les champs par le code du module : num_cpt_operation, cod_cpt_compte, ...

        # Résultats
        self.log.i(sNom="dtfComptes", dtfData=self.dtfHistorique, iIndent=self.iIndent+1)
        # self.dtfHistorique.to_csv("resultats/comptes_avant.csv", sep=";", encoding="UTF8")



    def fnc_charger_operations(self, sUrl, sCodeTypeImport):

        self.log.w("[fnc_charger_operations] sUrl=%s, sType=%s" % (sUrl, sCodeTypeImport), iIndent=self.iIndent, bPuce=True)

        # Lire fichier dernières opérations
        self.dtfOperations = pnd.read_csv(sUrl, delimiter=";", encoding="ANSI",
                                                names=['Date', 'Libellé', 'Débit euros', 'Crédit euros', 'vide'])

        if sCodeTypeImport == "CA_202203":

            self.dtfOperations = self.dtfOperations.rename(columns={    "Date":         "dte_operation",
                                                                        "Libellé":      "lib_operation",
                                                                        "Débit euros":  "mnt_debit",
                                                                        "Crédit euros": "mnt_credit"})
            del self.dtfOperations["vide"]

            self.dtfOperations["cod_compte"] = ""
            sCompte = ""

            # Parcours du dataframe
            for i in self.dtfOperations.index:

                # libelle du compte
                if (self.dtfOperations['dte_operation'][i]).startswith("Compte") \
                        or (self.dtfOperations['dte_operation'][i]).startswith("Livret"):
                    sCompte = self.dtfOperations['dte_operation'][i]
                    self.dtfOperations['dte_operation'][i] = ""
                else:
                    match = re.search(r'(\d+/\d+/\d+)', self.dtfOperations['dte_operation'][i][0:10])
                    if match:
                        self.dtfOperations['dte_operation'][i] = match.group()
                    else:
                        self.dtfOperations['dte_operation'][i] = ""

                self.dtfOperations['cod_compte'][i] = sCompte[-11:]


            # conversions dates
            self.dtfOperations.drop(self.dtfOperations[self.dtfOperations['dte_operation'] == ""].index, axis=0, inplace=True)
            self.dtfOperations['dte_operation'] = pnd.to_datetime(self.dtfOperations['dte_operation'],format= "%d/%m/%Y")

            # conversions montants
            self.dtfOperations["mnt_credit"] = self.dtfOperations["mnt_credit"].replace(to_replace=',', value='.', regex=True)
            self.dtfOperations["mnt_debit"]  = self.dtfOperations["mnt_debit"].replace(to_replace=',', value='.', regex=True)
            self.dtfOperations["mnt_credit"] = self.dtfOperations["mnt_credit"].replace(to_replace='\xa0', value='', regex=True)
            self.dtfOperations["mnt_debit"]  = self.dtfOperations["mnt_debit"].replace(to_replace='\xa0', value='', regex=True)
            self.dtfOperations["mnt_credit"] = self.dtfOperations["mnt_credit"].astype(float)
            self.dtfOperations["mnt_debit"]  = self.dtfOperations["mnt_debit"].astype(float)
            self.dtfOperations.loc[self.dtfOperations["mnt_debit"].isnull(), "mnt_debit"] = 0.0
            self.dtfOperations.loc[self.dtfOperations["mnt_credit"].isnull(), "mnt_credit"] = 0.0
            self.dtfOperations["mnt"] = self.dtfOperations["mnt_credit"] - self.dtfOperations["mnt_debit"]
            self.dtfOperations = self.dtfOperations.drop(columns=['mnt_credit', 'mnt_debit'])

            # nettoyage des libellés
            self.dtfOperations['lib_operation'] = self.dtfOperations['lib_operation'].replace(to_replace='\n', value=' ', regex=True)
            self.dtfOperations['lib_operation'] = self.dtfOperations['lib_operation'].replace(to_replace='  ', value=' ', regex=True)
            self.dtfOperations['lib_operation'] = self.dtfOperations['lib_operation'].replace(to_replace='  ', value=' ', regex=True)
            self.dtfOperations['lib_operation'] = self.dtfOperations['lib_operation'].replace(to_replace='  ', value=' ', regex=True)
            self.dtfOperations['lib_operation'] = self.dtfOperations['lib_operation'].replace(to_replace='  ', value=' ', regex=True)

            # conversions textes
            """for col in dtfOperations.index:
                if dtfOperations[col].dtypes == 'object':
                    dtfOperations[col] = dtfOperations[col].astype(str)"""

        # Rattachement comptes existants
        self.dtfOperations = self.dtfOperations.merge(self.dtfComptes, how="outer", on=["cod_compte"])

        # Ajout ces identifiants
        self.dtfOperations["num_operation"] = self.dtfOperations.index + len(self.dtfHistorique)

        # Résultats
        self.log.i(sNom="dtfOperations", dtfData=self.dtfOperations, iIndent=self.iIndent+1)
        #self.dtfOperations.to_csv("resultats/operations.csv", sep=";", encoding="UTF8")


    def fnc_ia_apprentissages(self):

        self.log.w("[fnc_ia_apprentissages]", iIndent=self.iIndent, bPuce=True)

        # Construction variables prédictives
        dtfData = self.dtfHistorique.loc[(self.dtfHistorique['cod_compte'] != ''),
                                         ['cod_compte', 'lib_operation_banque', 'lib_categorie', 'lib_tag_1', 'lib_tag_2', 'lib_tag_3']]
        tComptes = dtfData['cod_compte'].unique()
        for sCompte in tComptes:
            if len(sCompte)>0:
                dtfData['predict_cat_%s' % sCompte] = dtfData.loc[(self.dtfHistorique['cod_compte'] == sCompte),
                                                                  ['lib_categorie']]
                dtfData['predict_tag1_%s' % sCompte] = dtfData.loc[(self.dtfHistorique['cod_compte'] == sCompte),
                                                                  ['lib_tag_1']]
                dtfData['predict_tag2_%s' % sCompte] = dtfData.loc[(self.dtfHistorique['cod_compte'] == sCompte),
                                                                   ['lib_tag_2']]
                dtfData['predict_tag3_%s' % sCompte] = dtfData.loc[(self.dtfHistorique['cod_compte'] == sCompte),
                                                                   ['lib_tag_3']]


        del dtfData['lib_categorie']
        del dtfData['lib_tag_1']
        del dtfData['lib_tag_2']
        del dtfData['lib_tag_3']
        del dtfData['cod_compte']
        self.log.i("dtfData",dtfData,iIndent=self.iIndent+1)
        # TODO rajouter le séparateur";" dans tous les imports/exports au format CSV

        # Catégorisation
        mods = IaModelisation(  sEtude="predictions_comptes",
                                dtfData=dtfData,
                                dRatioJeuTest=0.1, dRatioValNull=0.2,
                                sPrefixePredict="predict_", sPrefixeCat="cat_",
                                log=self.log, iIndent=self.iIndent+1)
        mods.fnc_modelisations(sMethode="L")
        mods.fnc_affichage_resultats()

    def fnc_ia_predictions(self):

        self.log.w("[fnc_ia_predictions]", iIndent=self.iIndent, bPuce=True)

        sDate = str(datetime.date.today().isoformat())
        sEtude = "predictions_comptes"
        sDossier = "resultats/" + sEtude + "/" + sDate

        tComptes = self.dtfOperations['cod_compte'].unique()
        self.dtfOperations["lib_categorie"]=""
        self.dtfOperations["lib_tag_1"] = ""
        self.dtfOperations["lib_tag_2"] = ""
        self.dtfOperations["lib_tag_3"] = ""
        for sCompte in tComptes:

            for sMod in ["cat", "tag1", "tag2", "tag3"]:

                sFichier = "%s/predict_%s_%s_svm.mod" % (sDossier, sMod, sCompte)
                if os.path.exists(sFichier):
                    self.log.w("predict_%s_%s_svm" % (sMod, sCompte), iIndent=self.iIndent + 1, bPuce=True)
                    dtfData = self.dtfOperations.loc[self.dtfOperations["cod_compte"] == sCompte, ['lib_operation']]
                    dtfClasses = pnd.read_csv(sDossier + "/predict_%s_%s.cla" % (sMod,sCompte), delimiter=";",
                                              encoding="UTF8")
                    del dtfClasses["Unnamed: 0"]
                    mod = IaModeleSVM(sEtude=sEtude, dtfObserv=dtfData, dtfPredict=None, dtfClasses=dtfClasses,
                                      log=self.log,
                                      sDossier="resultats", iIndent=self.iIndent + 2)
                    mod.sNomPrediction = "predict_%s_%s" % (sMod, sCompte)
                    mod.fnc_import_mod(sUrl=sFichier)
                    dtfPredict = mod.fnc_prediction(dtfObserv=dtfData)
                    del dtfPredict['lib_operation']
                    self.dtfOperations = self.dtfOperations.join(dtfPredict)

                    if sMod=="cat":
                        self.dtfOperations["lib_categorie"] = self.dtfOperations["lib_categorie"] + self.dtfOperations[mod.sNomPrediction].fillna("")
                    elif sMod=="tag1":
                        self.dtfOperations["lib_tag_1"] = self.dtfOperations["lib_tag_1"] + self.dtfOperations[mod.sNomPrediction].fillna("")
                    elif sMod=="tag2":
                        self.dtfOperations["lib_tag_2"] = self.dtfOperations["lib_tag_2"] + self.dtfOperations[mod.sNomPrediction].fillna("")
                    elif sMod=="tag3":
                        self.dtfOperations["lib_tag_3"] = self.dtfOperations["lib_tag_3"] + self.dtfOperations[mod.sNomPrediction].fillna("")

                    del self.dtfOperations[mod.sNomPrediction]
                    del self.dtfOperations["%s_num" % mod.sNomPrediction]

        self.log.n()


    def fnc_exporter_comptes(self, sUrl):

        self.log.w("[fnc_exporter_comptes] sUrl=%s" % sUrl, iIndent=self.iIndent, bPuce=True)

        # réindexation des nouvelles opérations
        self.dtfOperations = self.dtfOperations.set_index('num_operation')

        # TODO ajout colonnes manquantes sur nouvelles opérations (infos comptes / proprio ...)

        # Consolidation des comptes
        dtfExport = pnd.concat([self.dtfHistorique,self.dtfOperations])
        dtfExport.to_csv(sUrl, sep=";", encoding="ANSI")

        # TODO Export TdB => Page "Synthèse M" : soldes de tous mes comptes

        # TODO Export TdB => Page "Synthèse M&L" : soldes de tous les comptes communs

        # TODO Export TdB => Page "Charges" : Synthèse des charges par catégorie & mois

        # TODO Export TdB => Page "Maison" : Synthèse des travaux et aménagement par personne / catégorie / fournisseur / mois




