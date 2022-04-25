# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pnd
# ----------------------------------------------------------------------------------------------------------------------
# CLASSE "Fic : classe générique "mère" des classes de gestion des différents types de fichiers
# ----------------------------------------------------------------------------------------------------------------------

class Fic:

# Attributs

    sUrl = ""
    sExtension = ""
    sOnglet = ""
    dtfElements = pnd.DataFrame()
    lstColsElements = ['type','texte','url','largeur', 'hauteur', 'data']

# Constructeur

    def __init__(self, sUrl=""):
        self.sUrl = sUrl
        self.sExtension = sUrl.split(".")[-1]

# Méthodes

    def mAjouterImage(self, sUrl, iLarg=0, iHaut=0):
        self.dtfElements = pnd.concat([self.dtfElements,
                                       pnd.DataFrame([["IMG",       # Type d'élément
                                                       "",          # Texte
                                                       sUrl,        # Url
                                                       iLarg,       # Largeur
                                                       iHaut,       # Hauteur
                                                       None]],      # Data
                                                     columns=self.lstColsElements)],
                                      ignore_index=True)

    def mAjouterEspace(self, iLarg=0, iHaut=0):
        self.dtfElements = pnd.concat([self.dtfElements,
                                       pnd.DataFrame([["ESP",       # Type d'élément
                                                       "",          # Texte
                                                       "",          # Url
                                                       iLarg,       # Largeur
                                                       iHaut,       # Hauteur
                                                       None]],      # Data
                                                     columns=self.lstColsElements)],
                                      ignore_index=True)

    def mAjouterTexte(self, sTexte):
        self.dtfElements = pnd.concat([self.dtfElements,
                                       pnd.DataFrame([["TXT",       # Type d'élément
                                                       sTexte,      # Texte
                                                       "",          # Url
                                                       0,          # Largeur
                                                       0,          # Hauteur
                                                       None]],      # Data
                                                     columns=self.lstColsElements)],
                                      ignore_index=True)

    def mAjouterTableau(self, dtfTable):
        self.dtfElements = pnd.concat([self.dtfElements,
                                       pnd.DataFrame([["TBL",       # Type d'élément
                                                       "",          # Texte
                                                       "",          # Url
                                                       0,           # Largeur
                                                       0,           # Hauteur
                                                       dtfTable]],  # Data
                                                     columns=self.lstColsElements)],
                                      ignore_index=True)