# ----------------------------------------------------------------------------------------------------------------------
# PACKAGES
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pnd
from moteur.fic import Fic
from reportlab.platypus import BaseDocTemplate, SimpleDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak, Table, TableStyle, CondPageBreak, Image, ParagraphAndImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle as PS
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# ----------------------------------------------------------------------------------------------------------------------
# CLASSE "Pdf" : gestion des fichiers PDF
# ----------------------------------------------------------------------------------------------------------------------

class Pdf(Fic):

# Attributs



# Constructeur

    def __init__(self, sUrl=""):
        super().__init__(sUrl=sUrl)

# Méthodes

    def mGenerer(self):

        tElements = []

        for i in self.dtfElements.index:

            # Image
            if self.dtfElements["type"][i] == "IMG":
                tElements.append(Image(self.dtfElements["url"][i],width=self.dtfElements["largeur"][i], height=self.dtfElements["hauteur"][i], hAlign=TA_LEFT ))

            # Espace
            elif self.dtfElements["type"][i] == "ESP":
                tElements.append(Spacer(1, self.dtfElements["hauteur"][i]))

            # Texte
            elif self.dtfElements["type"][i] == "TXT":
                tElements.append(Paragraph(self.dtfElements["texte"][i], PS(   name='CORPS',
                                                                               fontName='Helvetica',
                                                                               fontSize=11,
                                                                               alignment=TA_LEFT)))

                """from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import Paragraph, SimpleDocTemplate
    
                styles = getSampleStyleSheet()
                styleT = styles['Title']
                styleH = styles['Heading1']
                styleH2 = styles['Heading2']
                styleN = styles['Normal']
    
                story = []
                story.append(Paragraph("Ceci est le titre", styleT))
                story.append(Paragraph("Ceci est la section", styleH1))
                story.append(Paragraph("Ceci est la sous-section", styleH2))
                story.append(Paragraph("Ceci est le paragraphe", styleN))"""

            # Tableau
            elif self.dtfElements["type"][i] == "TBL":
                dtf = pnd.DataFrame(data=self.dtfElements["data"][i])
                data = dtf.values.tolist()
                #data = dtf.columns
                t = Table(data, colWidths=40, rowHeights=25)
                #t = Table(data)
                t.setStyle(TableStyle([('FONTSIZE', (0, 0), (-1, -1), 12),
                                       ('INNERGRID',  (0, 0), (-1, -1), 0.5, colors.grey),
                                       ('BOX', (0, 0), (-1, 0), 0.5, colors.black),
                                       ('TEXTCOLOR', (0, 0), (-1, 0), colors.red),
                                       ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
                                       ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                       ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                       ('VALIGN', (0, 0), (-1, -1), 'MIDDLE') ]))
                t.hAlign=TA_CENTER
                tElements.append(t)

        doc = SimpleDocTemplate(    self.sUrl,
                                    pagesize = A4,
                                    title = 'Plan B',
                                    author = 'PAG' )
        doc.build(tElements)

