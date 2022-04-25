# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------

from postgresqlconnector import DB
import pandas as pnd  # manipulation et analyse

# ----------------------------------------------------------------------------------------------------------------------
# CLASSE "Bdd" : gestion des connexions aux bases de données
# ----------------------------------------------------------------------------------------------------------------------

class Bdd:

# Attributs
    sHost = ""
    iPort = 0
    sDb = ""
    sUser = ""
    sMdp = ""

# Constructeur
    def __init__(self,sCodeProjet):
        if sCodeProjet == "CVLF":
            self.sHost = ""
            self.iPort = 0
            self.sDb = ""
            self.sUser = ""
            self.sMdp = ""

# Méthodes

    def mConnexion(self):
        DB.set_connection_info(self.sHost, self.iPort, self.sDb,self.sUser, self.sMdp)

    def mSelection(self, sSQL):

        with DB.create_transaction():
            result = DB.query(sSQL)

        dtfResult = pnd.DataFrame(result)
        return dtfResult

