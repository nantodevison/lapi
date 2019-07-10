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
    cam_proche_liste=groupe_pl_rappro.groupby('liste_passag_faux').count()['state']
    cam_proche_liste_triee=cam_proche_liste.sort_values(ascending=False).reset_index()
    base=alt.Chart(jointure).encode(x=alt.X('camera_id:N', axis=alt.Axis(title='num caméra')))
    bar=base.mark_bar().encode(y=alt.Y('nb_pl_x:Q', axis=alt.Axis(title='nb PL')))
    line=base.mark_line(color="#ffd100", strokeWidth=5).encode(y=alt.Y('pct_faux:Q', axis=alt.Axis(title='% PL faux')))
    return (bar+line).resolve_scale(y='independent') | (
        alt.Chart(cam_proche_liste_triee.iloc[:10]).mark_bar().encode(
            x=alt.X('liste_passag_faux', axis=alt.Axis(title='trajet')),
                    y=alt.Y('state:Q', axis=alt.Axis(title='Nb occurence'))))

def graph_nb_veh_ttjours_ttcam(df) : 
    """
    Fonction de graph du nb ceh par jour et par pour l'ensembles des cameras
    en entree : 
       df : la df des passages isssue de la bdd
    en sortie : 
        graph_filtre_tps : une chart altair concatenee vertical. en x les jours, en y le nb de veh 
    """
    nb_passage_j_cam=df.reset_index().set_index('created').groupby('camera_id').resample('D').count().drop('camera_id',axis=1).reset_index()
    return alt.VConcatChart(vconcat=([alt.Chart(nb_passage_j_cam.loc[nb_passage_j_cam['camera_id']==camera], title=f'camera {camera}').mark_bar(size=20).encode(
                                    x=alt.X('created',axis=alt.Axis(title='Jour', format='%A %d-%m-%y',labelAngle=45)),
                                    y=alt.Y('immat',axis=alt.Axis(title='Nombre de véhicules'))).properties(width=1000)  for camera in 
        [1,2,3,4,5,6,7,8,9,10,11,12,13,15,18,19]]))
def graph_trajet (dico_od, date, o_d): 
    """
    visualiser les trajets, sans affectation en transit ou non
    """
    titre=pd.to_datetime(date).day_name(locale ='French')+' '+pd.to_datetime(date).strftime('%Y-%m-%d')+' : '+o_d
    dico_od_graph=dico_od.loc[(dico_od.date_cam_1.between(pd.to_datetime(date+' 00:00:00'),pd.to_datetime(date+' 23:59:59'))) & (
        dico_od.apply(lambda x : x['o_d']==o_d,axis=1))].copy()
    dico_od_graph.tps_parcours=pd.to_datetime('2019-01-01')+dico_od_graph.tps_parcours
    return alt.Chart(dico_od_graph, title=titre).mark_point().encode(
                            x=alt.X('date_cam_1:T',title='Heure'),
                            y=alt.Y('tps_parcours',axis=alt.Axis(title='temps de parcours', format='%H:%M'))).properties(width=700, height=500).interactive()
    

def graph_trajet_multiple(dico_od, date, o_d, nb_jours):
    """
    Regroupement de charts altair issues de graph_trajet sur un plusieurs jours
    """
    dico_od_graph=dico_od.loc[(dico_od.date_cam_1.between(pd.to_datetime(date+' 00:00:00'),pd.to_datetime(date+' 23:59:59')+pd.Timedelta(str(nb_jours)+'D'))) & (
        dico_od.apply(lambda x : x['o_d']==o_d,axis=1))].copy()
    return alt.VConcatChart(vconcat=([graph_trajet(dico_od_graph,date_g.strftime('%Y-%m-%d'), o_d) 
                                      for date_g in pd.date_range(date, periods=nb_jours, freq='D')]))
     
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
                                             (df_transit['o_d']==o_d)]).copy()
    copie_df=test_filtre_tps[['date_cam_1','tps_parcours','filtre_tps', 'type']].copy()
    copie_df.tps_parcours=pd.to_datetime('2018-01-01')+copie_df.tps_parcours
    try : 
        copie_df['filtre_tps']=copie_df.apply(lambda x : 'Transit' if x['filtre_tps'] else 'Local',axis=1)
        copie_df['type']=copie_df.apply(lambda x : 'Reglementaire' if x['type']=='85eme_percentile' else x['type'],axis=1)
    except ValueError : 
        pass
    graph_filtre_tps = alt.Chart(copie_df, title=titre).mark_point().encode(
                                x=alt.X('date_cam_1',axis=alt.Axis(title='Horaire', format='%Hh%M')),
                                y=alt.Y('tps_parcours',axis=alt.Axis(title='Temps de parcours', format='%H:%M')),
                                tooltip='hoursminutes(tps_parcours)',
                                color=alt.Color('filtre_tps:N', legend=alt.Legend(title="Type de trajet")),
                                shape=alt.Shape('type:N',legend=alt.Legend(title="Source temps de reference"))).interactive().properties(width=800,height=400)
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

def graph_VL_PL_transit_j_cam(synt_nb_veh_cam, date, *cam) : 
    """
    pour creer des graph du nb de veh  par heue sur une journee à 1 camera
    en entree : 
        synt_nb_veh_cam : df agregeant les donnees de VL, PL, PL en transit et %PL transit. issu de Resultats.pourcentage_pl_camera
        date : string : date de debut, forme YYYY-MM-DD, ou 'JO' pour jour ouvrés, ou 'Ma/Je' pour 
        camera : integer : nume de la camera etudiee. on peut en passer plsueiurs et obtenir une somme des nb veh et une moyenne des %PL
    en sortie : 
        graph : chart altair avec en x l'heure et en y le nb de veh
    """
    #selection df pour graphique, on peurt demander 'Jours Ouvré' 
    if date=='JO' : 
        synt_nb_veh_cam['heure']=synt_nb_veh_cam.created.dt.hour
        groupe_jo=synt_nb_veh_cam.loc[synt_nb_veh_cam.set_index('created').index.dayofweek < 5].groupby(['camera_id','heure','type']).mean().reset_index()
        groupe_jo['created']=groupe_jo.apply(lambda x : pd.to_datetime(0)+pd.Timedelta(str(x['heure'])+'H'),axis=1)
        pour_graph=groupe_jo.loc[groupe_jo['camera_id'].isin(cam)]
    elif date=='MJA' :
        synt_nb_veh_cam['heure']=synt_nb_veh_cam.created.dt.hour
        groupe_jo=synt_nb_veh_cam.groupby(['camera_id','heure','type']).mean().reset_index()
        groupe_jo['created']=groupe_jo.apply(lambda x : pd.to_datetime(0)+pd.Timedelta(str(x['heure'])+'H'),axis=1)
        pour_graph=groupe_jo.loc[groupe_jo['camera_id'].isin(cam)]
    elif date == 'Ma/Je' : 
        synt_nb_veh_cam['heure']=synt_nb_veh_cam.created.dt.hour
        groupe_jo=synt_nb_veh_cam.loc[synt_nb_veh_cam.set_index('created').index.dayofweek.isin([1,3])].groupby(['camera_id','heure','type']).mean().reset_index()
        groupe_jo['created']=groupe_jo.apply(lambda x : pd.to_datetime(0)+pd.Timedelta(str(x['heure'])+'H'),axis=1)
        pour_graph=groupe_jo.loc[groupe_jo['camera_id'].isin(cam)]
    else : 
        pour_graph=synt_nb_veh_cam.loc[(synt_nb_veh_cam.apply(lambda x : x['created'].dayofyear==pd.to_datetime(date).dayofyear,axis=1))
                                  &(synt_nb_veh_cam['camera_id'].isin(cam))]
    if len(cam)>1 : 
        pour_graph=pour_graph.groupby(['heure','type']).agg({'nb_veh':'sum','pct_pl_transit':'mean','created':'min'}).reset_index()
    #graphique
    base=alt.Chart(pour_graph).encode(x=alt.X('created', axis=alt.Axis(title='Heure', format='%Hh%M')))
    bar = base.mark_bar(opacity=0.7, size=20).encode(
        y=alt.Y('nb_veh:Q',stack=None, axis=alt.Axis(title='Nb de vehicules')),
        color='type')
    line=base.mark_line(color='green').encode(y=alt.Y('pct_pl_transit:Q', axis=alt.Axis(title='% de PL en transit')))
    return (bar+line).resolve_scale(y='independent').properties(width=800) 

def graph_nb_veh_jour_camera(df, date_d, date_f, camera=4, type='TV') : 
    """
    pour creer des graph du nb de veh  par heue sur une journee à 1 camera
    en entree : 
        df : df des passages initiales, telle qu'importee depuis la bdd
        date_d : string : date de debut, generalement de la forme YYYY-MM-DD 00:00:00
        date_f : string : date de debut, generalement de la forme YYYY-MM-DD 23:59:59
        camera : integer : nume de la camera etudiee
        type : string : dofferenciation des VL, PL ou TV. par defaut : TV        
    en sortie : 
        graph : chart altair avec en x l'heure et en y le nb de veh
    """
    test2=df.loc[(df['created'].between(date_d,date_f)) & 
                 (df['camera_id']==camera)]
    if type=='PL' : 
        test2=test2.loc[test2['l']==1]
    graph=alt.Chart(test2.set_index('created').resample('H').count().reset_index(),title=
                    pd.to_datetime(date_d).day_name(locale ='French')+' '+pd.to_datetime(date_d).strftime('%d-%m-%y')+' ; camera '+ str(camera)).mark_bar().encode(
                   x=alt.X('created', axis=alt.Axis(title='Heure', format='%Hh%M')),
                    y=alt.Y('immat', axis=alt.Axis(title='Nombre de vehicule')) )
    return graph  

def graph_nb_veh_jour_camera_multi_j(df,date_debut,date_fin,cam,nb_jour, type='TV'): 
    """
    Regroupement de charts altair issues de graph_nb_veh_jour_camera sur plusieurs jours
    en entre :
        cf graph_nb_veh_jour_camera
        nb_jours : integer : nb de jours à concatener
    en sortie : 
        une chart altair concatenee verticalement avec un pour chaque jour
    """
    df_index_ok=df.reset_index()
    dico_graph={indice:graph_nb_veh_jour_camera(df_index_ok, dates[0], dates[1], cam, type) 
               for indice,dates in enumerate(zip([str(x) for x in pd.date_range(date_debut, periods=nb_jour, freq='D')],
                                [str(x) for x in pd.date_range(date_fin, periods=nb_jour, freq='D')]))}
    liste_graph=[graph for graph in {'liste_graph'+str(i):alt.HConcatChart(hconcat=([dico_graph[key] for key in dico_graph.keys() if key in range(i-7,i)])) for i in [8,15,22]}.values()]
    return alt.VConcatChart(vconcat=(liste_graph))

def graph_transit_vl_pl_camera_multi_j(synt_nb_veh_cam, date,cam,nb_jour): 
    """
    Regroupement de charts altair issues de graph_VL_PL_transit_j_cam sur plusieurs jours
    en entre :
        cf graph_VL_PL_transit_j_cam
        nb_jours : integer : nb de jours à concatener
    en sortie : 
        une chart altair concatenee verticalement avec un pour chaque jour
    """
    dico_graph={'graph'+str(indice):graph_VL_PL_transit_j_cam(synt_nb_veh_cam, date, cam) 
               for indice,date in enumerate(zip([str(x) for x in pd.date_range(date, periods=nb_jour, freq='D')]))}
    liste_graph=[dico_graph[key] for key in dico_graph.keys()]
    return alt.VConcatChart(vconcat=(liste_graph))  
    
def analyse_passage_proches(groupe_pl_rappro, groupe_pl):
    """
    Creer des df d'analyse des passages proches
    en entree : 
        groupe_pl_rappro : df issues de passages_proches
        groupe_pl : df issues de passages_proches
    en sortie : 
        jointure : df des passages proches et totaux par camera
        
    """
    #isoler les passages rapprochés avec une fiabilité foireuse
    #1. reconversion de la df de liste en df de passages uniques par passage paruneliste
    liste=zip(groupe_pl_rappro.index.tolist(),groupe_pl_rappro.liste_passag_faux.tolist(),groupe_pl_rappro.liste_created_faux.tolist(),groupe_pl_rappro.fiability_faux.tolist())
    liste_finale=[]
    for a in liste :
        for i,l in enumerate(a[1]) :
            liste_inter=[a[0],l,a[2][i],a[3][i]]
            liste_finale.append(liste_inter)    
    #2. conversion en df
    df_passage_rapproches=pd.DataFrame.from_records(liste_finale, columns=['immat', 'camera_id','created','fiability'])
    
    #analyse des cameras concernée
    #nb de pl par camera
    liste=zip(groupe_pl.index.tolist(),groupe_pl.camera_id.tolist(),groupe_pl.created.tolist())
    liste_finale=[]
    for a in liste :
        for i,l in enumerate(a[1]) :
            liste_inter=[a[0],l,a[2][i]]
            liste_finale.append(liste_inter)    
    df_pl=pd.DataFrame.from_records(liste_finale, columns=['immat', 'camera_id','created'])
    nb_pl_cam=df_pl.groupby('camera_id').count()['immat'].reset_index().rename(columns={'immat':'nb_pl'})
    nb_pl_cam['type']='tot'
    #nb_pl faux par cam
    nb_passage_faux_cam=df_passage_rapproches.groupby('camera_id').count()['immat'].reset_index().rename(columns={'immat':'nb_pl'})
    nb_passage_faux_cam['type']='faux'
    
    #jointure
    jointure=nb_pl_cam.merge(nb_passage_faux_cam, on='camera_id')
    jointure['pct_faux']=jointure.apply(lambda x: x['nb_pl_y']/x['nb_pl_x']*100,axis=1)
    
    return jointure

def comp_lapi_gest(df_passages_immat_ok,donnees_gest, camera, date_d='2019-01-28', date_f='2019-02-11'):
    """
    Graph de comparaison entr eles donnes gestionnaire et les donnees lapi
    en entree : 
        df_passages_immat_ok : df des passages valide cf donnees_postraitees.filtre_plaque_non_valable
        donnees_gest : df des donnes gestionnaires, cf donnees_postraitees.donnees_gest
        camera : entier : numero de camera
    en sortie :
        graph_comp_lapi_gest : chaetr altair de type bar
    """
    # regrouper les données PL validées par jour et par camera
    nb_pl_j_cam=df_passages_immat_ok.groupby('camera_id').resample('D').count()['immat'].reset_index().rename(columns={'immat':'nb_veh'})
    nb_pl_j_cam['type']='lapi'
    comp_traf_lapi_gest=pd.concat([nb_pl_j_cam,donnees_gest],sort=False)
    comp_traf_lapi_gest_graph=comp_traf_lapi_gest.loc[comp_traf_lapi_gest.created.between(
        pd.to_datetime(date_d),pd.to_datetime(date_f))].copy()
    
    return alt.Chart(comp_traf_lapi_gest_graph.loc[comp_traf_lapi_gest_graph['camera_id']==camera],title=
                                   f'Camera {camera}').mark_bar(opacity=0.8).encode(
                x=alt.X('yearmonthdate(created):O',axis=alt.Axis(title='date',format='%A %x')),
                y=alt.Y('nb_veh:Q', stack=None,axis=alt.Axis(title='Nombre de véhicule')),
                color='type',
                order=alt.Order("type", sort="ascending")).properties(width=500)
  
def comp_lapi_gest_multicam(df_passages_immat_ok,donnees_gest, date_d='2019-01-28', date_f='2019-02-11',
                            liste_num_cam=list(range(1,14))+[15,18,19]):  
    """
    porposer un graph unique pour toute les cmaraison gest-lapi
    en entree : la liste des cameras
    """
    return alt.VConcatChart(vconcat=[comp_lapi_gest(df_passages_immat_ok,donnees_gest, camera)
                              for camera in liste_num_cam])
    
#def graph_pct_pl_transit():