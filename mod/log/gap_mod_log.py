# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pnd

# ----------------------------------------------------------------------------------------------------------------------
# CLASSE "PB_Log" : gestion des logs
# ----------------------------------------------------------------------------------------------------------------------

class Log:

# Attributs

    oFichier = ""

# Constructeur

    def __init__(self, sUrl="default.log"):
        self.oFichier = open(sUrl, "w")

# Méthodes

    def fnc_format_cellule_num(self, dValeur, dMax=0, bNegatif=False, bPourcentage=False, iNbCar=20, iPrecision=2):

        dicFormatNum = {0: "%.0f", 1: "%.1f", 2: "%.2f", 3: "%.3f", 4: "%.4f", 5: "%.5f"}
        sRetour = ""

        if (bNegatif==False and dValeur<0) or (dMax>0 and dValeur>dMax):
            sRetour = "--.--"
        else:
            sRetour = dicFormatNum[iPrecision] % dValeur
        if bPourcentage:
            sRetour = sRetour + " %"

        sRetour = ((100*" ")+sRetour+" ")[-iNbCar:]
        return sRetour

    def fnc_format_cellule_txt(self, sTexte, iNbCar=20):
        sTexte = str(sTexte)
        return (" "+sTexte[:iNbCar-2]+(100*" "))[:iNbCar]

    def fnc_format_liste(self, lstVals):
        sTexte = ""
        for val in lstVals:
            sTexte = sTexte + str(val) + ", "
        sTexte = sTexte[0:-2]
        return sTexte

    # Saut de ligne (\n)
    def n(self):
        print("")
        self.oFichier.write("\n")

    # Séparateur
    def s(self):
        print(100*"-")
        self.oFichier.write(100*"-")

    # Ecriture d'une ligne (write)
    def w(self, sTexte, iIndent=0, bPuce=False):

        if iIndent == -1:
            sTexte = ("*** " + sTexte + " " + (100 * "*"))[0:100]
        else :
            if bPuce: sTexte = "> " + sTexte
            sTexte = (4 * iIndent)*" " + sTexte

        #if (iIndent == -1) | bPuce : self.n()
        print(sTexte)
        self.oFichier.write("\n"+sTexte)
        if (iIndent == -1) | bPuce: self.n()

    # Affichage d'une liste
    def l(self, lstVals, iIndent=0):
        #for index in enumerate(lstVals):
        #    self.w("[%s] %s" % (index[0], index[1]), iIndent=iIndent)
        for index, value in lstVals.items():
            self.w("[%s] %s" % (index, value), iIndent=iIndent)


    # Affichage d'un tableau
    def t(self, dtfCols, dtfData, bColIndex = False, sSeparateurV = "|", sSeparateurH = "-", iIndent=0,):

        # Ajout colonne index dans dtfData
        if bColIndex:
            dtfData = dtfData.reset_index()

        # réordonnacement des colonnes (et suppression des colonnes inutiles)
        dtfData = dtfData[dtfCols["nom"].tolist()]

        # Construction ligne d'entête
        sLigne = ""
        iLongTotal = 0
        for i in range(len(dtfCols.index)):
            iLong = dtfCols["long"][i]
            sNom =  dtfCols["nom"][i]
            sCentrage = int((iLong-len(sNom))/2)*" "
            sLigne = sLigne + self.fnc_format_cellule_txt( sCentrage + sNom,iLong) + sSeparateurV
            iLongTotal = iLongTotal + iLong +1
        iLongTotal = iLongTotal - 1
        sLigne = sLigne[:len(sLigne)-1]

        # Entête
        self.w(iLongTotal * sSeparateurH, iIndent=iIndent)
        self.w(sLigne, iIndent=iIndent)
        self.w(iLongTotal * sSeparateurH, iIndent=iIndent)

        # Données
        for iLig in range(len(dtfData.index)):
            sLigne = ""
            for iCol in range(len(dtfCols.index)):
                sFormat = dtfCols["format"][iCol]
                iLong = dtfCols["long"][iCol]
                iPrecis = dtfCols["precis"][iCol]
                if sFormat=="txt":
                    sLigne = sLigne + self.fnc_format_cellule_txt( dtfData.iloc[iLig,iCol],iLong)
                elif sFormat=="num":
                    sLigne = sLigne + self.fnc_format_cellule_num(dtfData.iloc[iLig,iCol], iPrecision=iPrecis,iNbCar=iLong)
                sLigne = sLigne + "|"

            self.w(sLigne[:len(sLigne)-1], iIndent=iIndent)

        # Pied
        self.w(iLongTotal * sSeparateurH, iIndent=iIndent)
        self.n()

    # affichage des infos d'un dataframe
    def i(self, sNom, dtfData, iIndent=0):

        # récupération des formats
        dtfStruct = pnd.DataFrame(dtfData.dtypes, columns={'format'})

        # comptages des valeurs null
        dtfStruct['non-null'] = 0
        dtfStruct['non-null%'] = 0.0

        if len(dtfData)>0:
            for col in dtfData.columns:
                iNull = dtfData[col].isnull().sum()
                dtfStruct.loc[col,'non-null'] = len(dtfData) - iNull
                dtfStruct.loc[col,'non-null%'] = 100 * (len(dtfData) - iNull) / len(dtfData)

        # noms des colonnes
        dtfStruct['colonne'] = dtfData.columns

        # affichage du tableau
        self.w("%s [%s lignes]" % (sNom,len(dtfData) ), iIndent=iIndent, bPuce=True)
        dtfCols = pnd.DataFrame([   ["colonne",     "txt", 30, 0],
                                    ["format",      "txt", 15, 0],
                                    ["non-null",    "num", 15, 0],
                                    ["non-null%",   "num", 15, 2]], columns=["nom","format","long","precis"])

        self.t(dtfData=dtfStruct, dtfCols=dtfCols, iIndent=iIndent+1)


