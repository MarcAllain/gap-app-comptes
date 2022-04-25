
from mod.log.gap_mod_log import Log
import pandas as pnd
import datetime
from mod.ia.gap_mod_ia_modelisation import IaModelisation
from mod.ia.gap_mod_ia_modele_svm import IaModeleSVM

sDate = str(datetime.date.today().isoformat())
sEtude = "test_climat"

# Logs
log = Log(sUrl="logs/%s.log" % sDate)
log.w("DEB", iIndent=-1)

log.w("APPRENTISSAGE...",bPuce=True)

# chargement des données
log.w("Chargement des données", iIndent=1)
log.n()
dtfMessages = pnd.read_csv("rechauffementClimatique.csv", delimiter=";", encoding="UTF8")
log.i(sNom="dtfMessages",dtfData=dtfMessages, iIndent=2)

# préparation des données
dtfMessages = dtfMessages.rename(columns={"CROYANCE":"predict_croyance",
                                          "TWEET":"txt_tweet"})
del dtfMessages['CONFIENCE']

modelisation = IaModelisation(sEtude=sEtude,
                              dtfData=dtfMessages, sPrefixePredict="predict_",
                              dRatioJeuTest=0.2,  dRatioValNull=0.5,
                              sDossier="resultats",
                              log=log, iIndent=1)

modelisation.fnc_modelisations(sMethode="L")
modelisation.fnc_affichage_resultats()
log.n()

log.w("PREDICTION...",bPuce=True)

# Nouvelle instance du modèle généré
sDossier = "resultats/" + sEtude+"/"+sDate
dtfObserv = pnd.DataFrame([["Why should trust scientists with global warming if they didnt know Pluto wasnt a planet !!!"],
                           ["Our climate is changing just as we are changing, do you believe in climate change?"],
                           ["How do we solve this global warming thing?"]],
                          columns={"txt_tweet"})
dtfClasses = pnd.read_csv(sDossier+"/predict_croyance.cla", delimiter=";", encoding="UTF8")
mod = IaModeleSVM(sEtude=sEtude,dtfObserv=dtfObserv, dtfPredict=None, dtfClasses=dtfClasses, log=log, sDossier="resultats", iIndent=1)
mod.sNomPrediction = "predict_croyance"
mod.fnc_import_mod(sUrl=sDossier+"/predict_croyance_svm.mod")
mod.fnc_prediction(dtfObserv=dtfObserv)

log.n()
log.w("FIN", iIndent=-1)