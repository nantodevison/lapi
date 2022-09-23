# -*- coding: utf-8 -*-
'''
Created on 27 mai 2021

@author: Martin

module d'analyse des nationalites 
'''

import pandas as pd
import altair as alt

dicoPaysZone = {'ES': 'Péninsule Ibérique','PT': 'Péninsule Ibérique','PL': "Europe de l'Est",'RO': "Europe de l'Est",'BG': "Europe de l'Est",'CZ': "Europe de l'Est",
     'UK': "Royaume Uni",'IE': "Royaume Uni",'SK': "Europe de l'Est",'SI': "Europe de l'Est",'BE': "Benelux / Allemagne",'NL': "Benelux / Allemagne",
     'DE': "Benelux / Allemagne",'AT': "Benelux / Allemagne",'FR': "France",'IT': "Méditérannée Européenne",'SM': "Méditérannée Européenne",
     'TR': "Méditérannée Européenne"}

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
    return df.state_modif_nc.value_counts(dropna=False).head(nbNationalite).sum()/df.state_modif_nc.value_counts(
        dropna=False).sum()
        
        
def ajoutN10(df_concat_pl):
    """
    ajouter la N10 dans la dataframe des passages de PL
    """
    jointure_cross = df_concat_pl.merge(df_concat_pl, on=['created', 'heure', 'state_fr_etr', 'type'])
    jointure_cross_5_11 = jointure_cross.loc[(jointure_cross['camera_id_x'] == 5) & (jointure_cross['camera_id_y'] == 11)].copy()
    jointure_cross_5_11['nb_veh'] = jointure_cross_5_11.nb_veh_x - jointure_cross_5_11.nb_veh_y
    
    jointure_cross_6_12 = jointure_cross.loc[(jointure_cross['camera_id_x'] == 6) & (jointure_cross['camera_id_y'] == 12)].copy()
    jointure_cross_6_12['nb_veh'] = jointure_cross_6_12.nb_veh_x - jointure_cross_6_12.nb_veh_y
    df_concat_pl_n10 = pd.concat([df_concat_pl,
                                        jointure_cross_5_11.drop(['camera_id_x', 'camera_id_y', 'nb_veh_x', 'nb_veh_y'], axis=1, errors='ignore').assign(camera_id=20),
                                        jointure_cross_6_12.drop(['camera_id_x', 'camera_id_y', 'nb_veh_x', 'nb_veh_y'], axis=1, errors='ignore').assign(camera_id=21)
                                        ]).reset_index(drop=True)
    df_concat_pl_n10.loc[df_concat_pl_n10.nb_veh < 0, 'nb_veh'] = 0
    return df_concat_pl_n10


def creerDataGroupeNationalite(passages_tot_redresse_31, passage_transit_redress_31):
    """
    créer les dataframes qui synthétise la proportion de chaque nationalité au sein des relevés de passages
    in : 
        passages_tot_redresse_31 : dataframe issue de resultat.passage_fictif_od (avec colonne 'state_modif_nc')
        passage_transit_redress_31 : dataframe issue de resultat.passage_fictif_od (avec colonne 'state_modif_nc')
    """
    df10PaysPlusRepresenteGlobal = (passages_tot_redresse_31.state_modif_nc.value_counts(dropna=False).head(9) / passages_tot_redresse_31.state_modif_nc.value_counts(
        ).sum()*100).astype(int).reset_index().rename(columns={'index': 'Pays', 'state_modif_nc': 'Pourcentage du volume total'})
    df10PaysPlusRepresenteTransit = (passage_transit_redress_31.state_modif_nc.value_counts(dropna=False).head(10) / passage_transit_redress_31.state_modif_nc.value_counts(
        ).sum()*100).astype(int).reset_index().rename(columns={'index': 'Pays', 'state_modif_nc': 'Pourcentage du volume total'})
    df6ZonesPlusRepresenteGlobal = (passages_tot_redresse_31.state_modif_nc.replace(dicoPaysZone).value_counts(
        dropna=False).head(6) / passages_tot_redresse_31.state_modif_nc.replace(dicoPaysZone).value_counts().sum(
        )*100).astype(int).reset_index().rename(columns={'index': 'Pays', 'state_modif_nc': 'Pourcentage du volume total'})
    df6ZonesPlusRepresenteTransit = (passage_transit_redress_31.state_modif_nc.replace(
        dicoPaysZone).value_counts(dropna=False).head(6) / passage_transit_redress_31.state_modif_nc.replace(
        dicoPaysZone).value_counts().sum()*100
                                     ).astype(int).reset_index().rename(columns={'index': 'Pays', 'state_modif_nc': 'Pourcentage du volume total'})
    df10PaysPlusRepresenteTransit['Pourcentage_text'] = df10PaysPlusRepresenteTransit['Pourcentage du volume total'].astype(str) + '%'
    df10PaysPlusRepresenteGlobal['Pourcentage_text'] = df10PaysPlusRepresenteGlobal['Pourcentage du volume total'].astype(str) + '%'
    df6ZonesPlusRepresenteGlobal['Pourcentage_text'] = df6ZonesPlusRepresenteGlobal['Pourcentage du volume total'].astype(str) + '%'
    df6ZonesPlusRepresenteTransit['Pourcentage_text'] = df6ZonesPlusRepresenteTransit['Pourcentage du volume total'].astype(str) + '%'
    return df10PaysPlusRepresenteTransit, df10PaysPlusRepresenteGlobal, df6ZonesPlusRepresenteGlobal, df6ZonesPlusRepresenteTransit


def nettoyerDataNc(passages_tot_redresse_31, passage_transit_redress_31):
    """
    simple filtre sur les données qui ont une nationalité modifié égale à nc
    in :
        passages_tot_redresse_31 : dataframe issue de resultat.passage_fictif_od (avec colonne 'state_modif_nc')
        passage_transit_redress_31 : dataframe issue de resultat.passage_fictif_od (avec colonne 'state_modif_nc')
    """
    dfNatGlobale = passages_tot_redresse_31.loc[passages_tot_redresse_31.state_modif_nc != 'nc'].copy()
    dfNatTransit = passage_transit_redress_31.loc[passage_transit_redress_31.state_modif_nc != 'nc'].copy()
    return dfNatGlobale, dfNatTransit

def creerDataGraphsHorairesMoyens(passages_tot_redresse_31, passage_transit_redress_31):
    """
    générer les df nécéssaires aux graphs de données horaires. cf graphs.graphNationaliteRepartitionFrEtranger
    in :
        passages_tot_redresse_31 : dataframe issue de resultat.passage_fictif_od (avec colonne 'state_modif_nc')
        passage_transit_redress_31 : dataframe issue de resultat.passage_fictif_od (avec colonne 'state_modif_nc')
    """
    dfNatGlobale, dfNatTransit = nettoyerDataNc(passages_tot_redresse_31, passage_transit_redress_31)
    df_synthese_pl_tot = dfNatGlobale.groupby(['camera_id', 'state_fr_etr']).resample('H').count()['immat'].reset_index().rename(columns={'immat': 'nb_veh'})
    df_synthese_pl_transit = dfNatTransit.set_index('created').groupby(['camera_id', 'state_fr_etr']).resample('H').count()['immat'].reset_index().rename(
            columns={'immat': 'nb_veh'})
    df_synthese_pl_tot['heure'] = df_synthese_pl_tot.created.dt.hour
    df_synthese_pl_transit['heure'] = df_synthese_pl_transit.created.dt.hour
    df_synthese_pl_tot['type'] = 'PL total'
    df_synthese_pl_transit['type'] = 'PL transit'
    df_concat_pl = pd.concat([df_synthese_pl_tot, df_synthese_pl_transit], sort=False)
    df_concat_pl_n10 = ajoutN10(df_concat_pl)
    df_concat_pl_jo = df_concat_pl_n10.loc[df_concat_pl_n10.set_index('created').index.dayofweek < 5].copy()
    df_concat_pl_joMoy = df_concat_pl_jo.groupby(['camera_id', 'state_fr_etr', 'type', 'heure']).mean().reset_index()
    return df_concat_pl_jo, df_concat_pl_joMoy
        