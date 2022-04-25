
from app.cpt.gap_app_cpt import GapAppCpt
from mod.log.gap_mod_log import Log
import pandas as pnd
import datetime

# Logs
sDate = str(datetime.date.today().isoformat())
log = Log(sUrl="logs/%s.log" % sDate)
log.w("DEB", iIndent=-1)


# Chargement des données
log.w("CHARGEMENT DES OPERATIONS...", bPuce=True)
appCpt = GapAppCpt(log=log, iIndent=1)
appCpt.dtfComptes = pnd.DataFrame([  ["57459014723", "CA", "Commun", "Nous"],
                                     ["57459212592", "CA", "Courant", "Marc"],
                                     ["57461174428", "CA", "Livret A", "Marc"],
                                     ["57461595636", "CA", "Livret", "Nous"]],
                                     columns=["cod_compte", "cod_banque", "cod_type_compte", "lib_proprio"])
appCpt.fnc_charger_comptes(sUrl="params/comptes.xlsx")
appCpt.fnc_charger_operations(sUrl="params/operations_ca.csv", sCodeTypeImport="CA_202203")

# Entrainements modèles
log.w("APPRENTISSAGES...", bPuce=True)
appCpt.fnc_ia_apprentissages()

# Prédictions sur nouveaux enregistrements
log.w("PREDICTIONS...", bPuce=True)
appCpt.fnc_ia_predictions()

# Export des comptes consolidés
log.w("EXPORTS...", bPuce=True)
appCpt.fnc_exporter_comptes(sUrl="resultats/comptes.csv")

"""
dtfA = pnd.DataFrame([  [1,1,1,1],
                        [2,2,2,2],
                        [3,3,3,3],
                        [4,4,4,4],
                        [5,5,5,5]],
                        columns=["A", "B", "C", "D"])

dtfB = pnd.DataFrame([  [6],
                        [7],
                        [8],
                        [9],
                        [10],
                        [11],
                        [12]],
                        columns=["E"])
dtfB = dtfB.loc[dtfB["E"]>7,]

print(dtfA)
print(dtfB)

dtfB["index"] = dtfA.index
dtfB = dtfB.set_index("index")
print(dtfB)

print('join')
print(dtfA.join(dtfB))
print('merge')
print(dtfA.merge(dtfB))"""

log.n()
log.w("FIN", iIndent=-1)