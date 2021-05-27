# -*- coding: utf-8 -*-
'''
Created on 27 mai 2021

@author: Martin

module d'analyse des nationalites 
'''

import pandas as pd
import altair as alt

def modifStateInconnu(state) : 
    """
    modifier le nom du pays dans la table des passages validés après le 31 en ciblant les valeurs avec un doute.
    in : 
        state : string : pays de départ
    out : 
        state_modif : pays modifié si inconu ou si ne faisant pas partie des 10 pays les plus représentés
    """
    if state in ('  ','!!') or pd.isnull(state) or '/' in state : 
        return 'nc' 
    else : 
        return state
    
def modifStateFrAutre(state) : 
    """
    modifier le nom du pays dans la table des passages validés après le 31, en séparant le français et etranger.
    in : 
        state : string : pays de départ
    out : 
        state_modif : 'FR' ou 'Autre'
    """
    if state == 'FR' : 
        return state 
    else : 
        return 'Autre'

def partXNationalite(df, nbNationalite):
    """
    calculer a part que représente les X nationalites les plus représentées adsn la df : 
    in : 
        df : df des passage redresse, issue trajets.trajet2passage (represente les trajets globaux ou transit uniquement)
        nbNationalite : integer : nombre de ntaionalite a prendre en compte
    out : 
        pourcentage de la part que represente le X nationalite par rapport au total
    """
    return df.state_modif_nc.value_counts(dropna=False).head(nbNationalite).sum()/df.state_modif_nc.value_counts(dropna=False).sum()



        