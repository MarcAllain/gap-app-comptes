# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pnd
from mod.fic.gap_mod_fic import Fic

# ----------------------------------------------------------------------------------------------------------------------
# CLASSE "Xls" : gestion des fichiers Excel
# ----------------------------------------------------------------------------------------------------------------------

class Xls(Fic):

# Attributs


# Constructeur

    def __init__(self, sUrl=""):
        super().__init__(sUrl=sUrl)

# Méthodes

    def mChargerOnglet(self, sOnglet=""):
        self.sOnglet = sOnglet
        return pnd.read_excel(self.sUrl, sheet_name=sOnglet)
