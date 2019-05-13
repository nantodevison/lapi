# -*- coding: utf-8 -*-
'''
Created on 27 fev. 2019
@author: martin.schoreisz

Module pour graphiques des donnees lapi

'''

import matplotlib #pour éviter le message d'erreurrelatif a rcParams
import pandas as pd
import numpy as np
import altair as alt
import os,math, datetime as dt




def graph_passages_proches(jointure, groupe_pl_rappro):
    """
    Visualiser les stats sur les pasages trop proches
    """
    cam_proche_liste=groupe_pl_rappro.groupby('liste_passag_faux').count()['l']
    cam_proche_liste_triee=cam_proche_liste.sort_values(ascending=False).reset_index()
    base=alt.Chart(jointure).encode(x=alt.X('camera_id:N', axis=alt.Axis(title='num caméra')))
    bar=base.mark_bar().encode(y=alt.Y('nb_pl_x:Q', axis=alt.Axis(title='nb PL')))
    line=base.mark_line(color="#ffd100", strokeWidth=5).encode(y=alt.Y('pct_faux:Q', axis=alt.Axis(title='% PL faux')))
    return (bar+line).resolve_scale(y='independent') | (
        alt.Chart(cam_proche_liste_triee.iloc[:10]).mark_bar().encode(
            x=alt.X('liste_passag_faux', axis=alt.Axis(title='trajet')),
                    y=alt.Y('l:Q', axis=alt.Axis(title='Nb occurence'))))

def graph_nb_veh_ttjours_ttcam(df) : 
    """
    Fonction de graph du nb ceh par jour et par pour l'ensembles des cameras
    en entree : 
       df : la df des passages isssue de la bdd
    en sortie : 
        graph_filtre_tps : une chart altair concatenee vertical. en x les jours, en y le nb de veh 
    """
    nb_passage_j_cam=df.reset_index().set_index('created').groupby('camera_id').resample('D').count().drop('camera_id',axis=1).reset_index()
    graph_filtre_tps = alt.Chart(nb_passage_j_cam).mark_bar(size=20).encode(
                                    x='created',
                                    y='immat').properties(width=800).facet(row='camera_id')
    return graph_filtre_tps

       
def graph_transit_filtre(df_transit, date_debut, date_fin, o_d):
    """
    pour visualiser les graph de seprataion des trajets de transit et des autres
    en entree :
        df_transit : df des o_d
        date_debut : string au format 2019-01-28 00:00:00
        date_fin : string au format 2019-01-28 00:00:00
        o_d : origine destination parmi les possibles du df_transit
    en sortie : 
        une chart altair avec en couleur le type de transit ou non, et en forme la source du temps de parcours pour filtrer   
    """
    titre=pd.to_datetime(date_debut).day_name(locale ='French')+' '+pd.to_datetime(date_debut).strftime('%Y-%m-%d')+' : '+o_d
    test_filtre_tps=(df_transit.loc[(df_transit['date_cam_1']>pd.to_datetime(date_debut)) &
                                             (df_transit['date_cam_1']<pd.to_datetime(date_fin)) &
                                             (df_transit['o_d']==o_d)])
    copie_df=test_filtre_tps[['date_cam_1','tps_parcours','filtre_tps', 'type']].head(5000).copy()
    copie_df.tps_parcours=pd.to_datetime('2018-01-01')+copie_df.tps_parcours
    graph_filtre_tps = alt.Chart(copie_df, title=titre).mark_point().encode(
                                x='date_cam_1',
                                y='hoursminutes(tps_parcours)',
                                tooltip='hoursminutes(tps_parcours)',
                                color=alt.Color('filtre_tps:N', legend=alt.Legend(title="Type de trajet", values=['local', 'transit'])),
                                shape=alt.Shape('type:N',legend=alt.Legend(title="Source temps de référence"))).interactive().properties(width=600)
    return graph_filtre_tps

def graph_transit_filtre_multiple(df_transit_avec_filtre, date_debut, date_fin, o_d, nb_jours):
    """
    Regroupement de charts altair issues de graph_transit_filtre sur un plusieurs jours
    en entre :
        cf graph_transit_filtre
        nb_jours : integer : nb de jours à concatener
    en sortie : 
        une chart altair concatenee verticalement avec un pour chaque jour
    """
    dico_graph={'graph'+str(indice):graph_transit_filtre(df_transit_avec_filtre,str(dates[0]),str(dates[1]), o_d) 
                 for indice,dates in enumerate(zip([str(x) for x in pd.date_range(date_debut, periods=nb_jours, freq='D')],
                        [str(x) for x in pd.date_range(date_fin, periods=nb_jours, freq='D')]))}
    liste_graph=[dico_graph[key] for key in dico_graph.keys()]
    
    return alt.VConcatChart(vconcat=(liste_graph))

def graph_VL_PL_transit_j_cam(synt_nb_veh_cam, date, cam) : 
    """
    pour creer des graph du nb de veh  par heue sur une journee à 1 camera
    en entree : 
        synt_nb_veh_cam : df agregeant les donnees de VL, PL, PL en transit et %PL transit. issu de donnees_postraitees.pourcentage_pl_camera
        date : string : date de debut, forme YYYY-MM-DD
        camera : integer : nume de la camera etudiee
    en sortie : 
        graph : chart altair avec en x l'heure et en y le nb de veh
    """
    #selection df pour graphique
    pour_graph=synt_nb_veh_cam.loc[(synt_nb_veh_cam.apply(lambda x : x['created'].dayofyear==pd.to_datetime(date).dayofyear,axis=1))
                                  &(synt_nb_veh_cam['camera_id']==cam)]
    #graphique
    base=alt.Chart(pour_graph).encode(x=alt.X('created', axis=alt.Axis(title='Heure', format='%H:%M')))
    bar = base.mark_bar(opacity=0.7, size=20).encode(y=alt.Y('nb_veh:Q',stack=None, axis=alt.Axis(title='Nb de véhicules')),color='type')
    line=base.mark_line(color='green').encode(y=alt.Y('pct_pl_transit:Q', axis=alt.Axis(title='% de PL en transit')))
    return (bar+line).resolve_scale(y='independent').properties(width=800) 

def graph_nb_veh_jour_camera_multi_j(synt_nb_veh_cam, date,cam,nb_jour): 
    """
    Regroupement de charts altair issues de graph_VL_PL_transit_j_cam sur plusieurs jours
    en entre :
        cf graph_nb_veh_jour_camera
        nb_jours : integer : nb de jours à concatener
    en sortie : 
        une chart altair concatenee verticalement avec un pour chaque jour
    """
    df_index_ok=df.reset_index()
    dico_graph={'graph'+str(indice):graph_VL_PL_transit_j_cam(synt_nb_veh_cam, date, cam) 
               for indice,date in enumerate(zip([str(x) for x in pd.date_range(date, periods=nb_jour, freq='D')]))}
    liste_graph=[dico_graph[key] for key in dico_graph.keys()]
    return alt.VConcatChart(vconcat=(liste_graph))  
    

    