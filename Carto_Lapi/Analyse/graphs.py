# -*- coding: utf-8 -*-
'''
Created on 27 fev. 2019
@author: martin.schoreisz

Module pour graphiques des donnees lapi

'''

import matplotlib #pour éviter le message d'erreurrelatif a rcParams
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from Import_Forme import dico_corrsp_camera_site
from Resultats import indice_confiance_cam, PL_transit_dir_jo_cam


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
                                x=alt.X('date_cam_1:T',axis=alt.Axis(title='Horaire', format='%Hh%M')),
                                y=alt.Y('tps_parcours:T',axis=alt.Axis(title='Temps de parcours', format='%H:%M')),
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

def graph_VL_PL_transit_j_cam(df_concat_pl_jo,df_pct_pl_transit, *cam) : 
    """
    pour creer des graph du nb de veh  par heue sur une journee à 1 camera
    en entree : 
        df_pct_pl_transit : df du pct de pl en transit, issus de resultat.pourcentage_pl_camera
        df_concat_pl_jo : df du nb de pl par jo classe en transit ou tital, issu de resultat.pourcentage_pl_camera
        cam : integer : numeros de la camera etudiee. on peut en passer plsueiurs et obtenir une somme des nb veh et une moyenne des %PL
    en sortie : 
        graph : chart altair avec en x l'heure et en y le nb de veh
    """
    #selection df pour graphique, on peurt demander 'Jours Ouvré'
    if [voie for voie, cams in dico_corrsp_camera_site.items() if cams==list(cam)] :
        titre=f'Nombre de PL et % de PL en transit sur {[voie for voie, cams in dico_corrsp_camera_site.items() if cams==list(cam)][0]}'
    else :
        if len(cam)>1 :
            titre=f'Nombre de PL et % de PL en transit au droit des caméras {cam}'
        else : 
            titre=f'Nombre de PL et % de PL en transit au droit de la caméra {cam[0]}'
    if len(cam)>1 : 
        df_concat_pl_jo_multi_cam=df_concat_pl_jo.loc[df_concat_pl_jo['camera_id'].isin(cam)].groupby(['heure','type']).agg({'nb_veh':'sum'}).reset_index()
        df_concat_pl_jo_multi_cam['nb_veh']=df_concat_pl_jo_multi_cam['nb_veh']/len(cam)
        df_pct_pl_transit_multi_cam=df_pct_pl_transit.loc[df_pct_pl_transit['camera_id'].isin(cam)].groupby(['heure']).agg({'nb_veh_x':'sum','nb_veh_y':'sum'}).reset_index()
        df_pct_pl_transit_multi_cam['pct_pl_transit']=df_pct_pl_transit_multi_cam['nb_veh_y']/df_pct_pl_transit_multi_cam['nb_veh_x']*100
    else : 
        df_concat_pl_jo_multi_cam=df_concat_pl_jo.loc[df_concat_pl_jo['camera_id'].isin(cam)]
        df_pct_pl_transit_multi_cam=df_pct_pl_transit.loc[df_pct_pl_transit['camera_id'].isin(cam)]
        
    bar=alt.Chart(df_concat_pl_jo_multi_cam,title=titre).mark_bar(opacity=0.7, size=20).encode(
        x='heure:O',
        y=alt.Y('nb_veh:Q',stack=None, axis=alt.Axis(title='Nb de vehicules',grid=False)),
        color='type')
    line=alt.Chart(df_pct_pl_transit_multi_cam).mark_line(color='green').encode(
        x='heure:O',
        y=alt.Y('pct_pl_transit:Q', axis=alt.Axis(title='% de PL en transit')))
    (bar+line).resolve_scale(y='independent').properties(width=800)
    return (bar+line).resolve_scale(y='independent').properties(width=800) 


def intervalle_confiance_cam(df_pct_pl_transit,df_concat_pl_jo, *cam): 
    pour_graph_synth,lien_traf_gest_traf_lapi=indice_confiance_cam(df_pct_pl_transit,df_concat_pl_jo,cam)
    lien_traf_gest_traf_lapi['heure']=lien_traf_gest_traf_lapi.apply(lambda x : pd.to_datetime(0)+pd.Timedelta(str(x['heure'])+'H'), axis=1)
    pour_graph_synth['heure']=pour_graph_synth.apply(lambda x : pd.to_datetime(0)+pd.Timedelta(str(x['heure'])+'H'), axis=1)
    #print(pour_graph_synth,lien_traf_gest_traf_lapi)
    #print(dico_corrsp_camera_site.items(),[voie for voie, cams in dico_corrsp_camera_site.items() if cams==list(cam)])
    if [voie for voie, cams in dico_corrsp_camera_site.items() if cams==list(cam)] :
        titre_interv=f'Nombre de PL et % de PL en transit sur {[voie for voie, cams in dico_corrsp_camera_site.items() if cams==list(cam)][0]}'
        titre_nb_pl=f'Nombre de PL selon la source sur {[voie for voie, cams in dico_corrsp_camera_site.items() if cams==list(cam)][0]}'
    else :
        if len(cam)>1 :
            titre_interv=f'Nombre de PL et % de PL en transit au droit des caméras {cam}'
            titre_nb_pl=f'Nombre de PL selon la source au droit des caméras {cam}'
        else : 
            titre_interv=f'Nombre de PL et % de PL en transit au droit de la caméra {cam[0]}'
            titre_nb_pl=f'Nombre de PL selon la source au droit de la caméra {cam[0]}'
              
    #graph d'intervalle de confiance
    df_intervalle=pour_graph_synth.loc[pour_graph_synth['type'].isin(['LAPI', 'SIREDO recale'])].copy()
    #pour legende
    lien_traf_gest_traf_lapi['legend_pct_transit']='Pourcentage PL transit'
    lien_traf_gest_traf_lapi['legend_i_conf']='Intervalle de confiance'
    line_trafic=alt.Chart(df_intervalle, title=titre_interv).mark_line().encode(
        x=alt.X('hoursminutes(heure)',axis=alt.Axis(title='Heure', titleFontSize=14,labelFontSize=14)),
        y=alt.Y('nb_veh:Q', axis=alt.Axis(title='Nombre de PL SIREDO',titleFontSize=14,labelFontSize=14)), 
        color=alt.Color('type',legend=alt.Legend(title='source du nombre de PL',titleFontSize=14,labelFontSize=14)))
    area_pct_max=alt.Chart(lien_traf_gest_traf_lapi).mark_area(opacity=0.7, color='green').encode(
        x='hoursminutes(heure)',
        y=alt.Y('pct_pl_transit_max:Q',
                axis=alt.Axis(title='Pourcentage de PL en transit',titleFontSize=14,labelFontSize=14,labelColor='green',titleColor='green'),
                scale=alt.Scale(domain=(0,100))), 
        y2='pct_pl_transit_min:Q',
        opacity=alt.Opacity('legend_i_conf'))
    line_pct=alt.Chart(lien_traf_gest_traf_lapi).mark_line(color='green').encode(
        x='hoursminutes(heure)',
        y='pct_pl_transit',
        opacity=alt.Opacity('legend_pct_transit', legend=alt.Legend(title='Analyse du transit LAPI',titleFontSize=14,labelFontSize=14)))
    pct=(area_pct_max+line_pct)
    graph_interval=(line_trafic+pct).resolve_scale(y='independent').properties(width=800, height=400).configure_title(fontSize=18)
    
    #graph comparaison nb_pl
    graph_nb_pl=alt.Chart(pour_graph_synth, title=titre_nb_pl).mark_line(opacity=0.7).encode(
        x=alt.X('hoursminutes(heure)',axis=alt.Axis(title='Heure', titleFontSize=14,labelFontSize=14)),
        y=alt.Y('nb_veh:Q', axis=alt.Axis(title='Nombre de PL SIREDO',titleFontSize=14,labelFontSize=14)), 
        color=alt.Color('type',title='source du nombre de PL', legend=alt.Legend(titleFontSize=14,labelFontSize=14))).properties(
            width=800, height=400).configure_title(fontSize=18)
        
    return graph_interval,graph_nb_pl


def graph_PL_transit_dir_jo_cam(df_pct_pl_transit, *cam):
    """
    graph de synthese du nombre de pl en trasit par heure. Base nb pl dir et pct_pl_transit lapi
    en entree : 
        df_pct_pl_transit : df du pct de pl en transit, issu de resultat.pourcentage_pl_camera
    en sortie : 
        graph : chart altair avec le nb pl, nb pl transit, %PL transit
    """
    #import donnees
    concat_dir_trafic, df_pct_pl_transit_multi_cam=PL_transit_dir_jo_cam(df_pct_pl_transit,cam)
    
    #creation du titre
    if [voie for voie, cams in dico_corrsp_camera_site.items() if cams==list(cam)] :
        titre=f'Nombre de PL et % de PL en transit sur {[voie for voie, cams in dico_corrsp_camera_site.items() if cams==list(cam)][0]}'
    else :
        if len(cam)>1 :
            titre=f'Nombre de PL et % de PL en transit au droit des caméras {cam}'
        else : 
            titre=f'Nombre de PL et % de PL en transit au droit de la caméra {cam[0]}'
    #ajout d'un attribut pour legende
    df_pct_pl_transit_multi_cam['legend']='Pourcentage PL en transit'
    concat_dir_trafic=concat_dir_trafic.loc[concat_dir_trafic['type'].isin(['Tous PL','PL en transit'])].copy()
    
    bar_nb_pl_dir=alt.Chart(concat_dir_trafic, title=titre).mark_bar(opacity=0.7).encode(
        x=alt.X('heure:O',axis=alt.Axis(title='Heure',titleFontSize=14,labelFontSize=14)),
        y=alt.Y('nb_pl:Q',stack=None, axis=alt.Axis(title='Nombre de PL',titleFontSize=14,labelFontSize=14)),
        color=alt.Color('type',legend=alt.Legend(title='Type de PL',titleFontSize=14,labelFontSize=14),sort="descending"))
    line_pct_pl_lapi=alt.Chart(df_pct_pl_transit_multi_cam).mark_line(color='green').encode(
        x=alt.X('heure:O',axis=alt.Axis(title='Heure',titleFontSize=14,labelFontSize=14)),
        y=alt.Y('pct_pl_transit', axis=alt.Axis(title='% PL en transit',labelFontSize=14,labelColor='green',titleFontSize=14,titleColor='green',grid=False)),
        opacity=alt.Opacity('legend', legend=alt.Legend(title='Donnees LAPI',titleFontSize=14,labelFontSize=14,labelLimit=300)))
    return (bar_nb_pl_dir+line_pct_pl_lapi).resolve_scale(y='independent').properties(width=800, height=400).configure_title(fontSize=18)

def graph_TV_jo_cam(df_pct_pl_transit, *cam):
    """
    graph de synthese du nombre de pl en trasit par heure. Base nb pl dir et pct_pl_transit lapi
    en entree : 
        df_pct_pl_transit : df du nb de vehicules, issus de resultat.pourcentage_pl_camera
    en sortie : 
        bar_nb_pl_dir : chart altair avec le nb pl, nb pl transit, tv
    """
    concat_dir_trafic=PL_transit_dir_jo_cam(df_pct_pl_transit, cam)[0]
    #creation du titre
    if [voie for voie, cams in dico_corrsp_camera_site.items() if cams==list(cam)] :
        titre=f'Nombre de véhicules sur {[voie for voie, cams in dico_corrsp_camera_site.items() if cams==list(cam)][0]}'
    else :
        if len(cam)>1 :
            titre=f'Nombre de véhicules au droit des caméras {cam}'
        else : 
            titre=f'Nombre de véhicules au droit de la caméra {cam[0]}'
    #ajout d'un attribut pour legende
    
    bar_nb_pl_dir=alt.Chart(concat_dir_trafic, title=titre).mark_bar().encode(
        x=alt.X('heure:O',axis=alt.Axis(title='Heure',titleFontSize=14,labelFontSize=14)),
        y=alt.Y('nb_pl:Q',stack=None, axis=alt.Axis(title='Nombre de vehicules',titleFontSize=14,labelFontSize=14)),
        color=alt.Color('type',legend=alt.Legend(title='Type de vehicules',titleFontSize=14,labelFontSize=14)),
        order=alt.Order('type', sort='descending')).properties(width=800, height=400).configure_title(fontSize=18)
    return bar_nb_pl_dir 
    

def graph_nb_veh_jour_camera(df, date_d, date_f, camera=4, type_v='TV') : 
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
    if type_v=='PL' : 
        test2=test2.loc[test2['l']==1]
    graph=alt.Chart(test2.set_index('created').resample('H').count().reset_index(),title=
                    pd.to_datetime(date_d).day_name(locale ='French')+' '+pd.to_datetime(date_d).strftime('%d-%m-%y')+' ; camera '+ str(camera)).mark_bar().encode(
                   x=alt.X('created', axis=alt.Axis(title='Heure', format='%Hh%M')),
                    y=alt.Y('immat', axis=alt.Axis(title='Nombre de vehicule')) )
    return graph  

def graph_nb_veh_jour_camera_multi_j(df,date_debut,date_fin,cam,nb_jour, type_v='TV'): 
    """
    Regroupement de charts altair issues de graph_nb_veh_jour_camera sur plusieurs jours
    en entre :
        cf graph_nb_veh_jour_camera
        nb_jours : integer : nb de jours à concatener
    en sortie : 
        une chart altair concatenee verticalement avec un pour chaque jour
    """
    df_index_ok=df.reset_index()
    dico_graph={indice:graph_nb_veh_jour_camera(df_index_ok, dates[0], dates[1], cam, type_v) 
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
                x=alt.X('monthdate(created):O',axis=alt.Axis(title='date',labelAngle=80)),
                y=alt.Y('nb_veh:Q', stack=None,axis=alt.Axis(title='Nombre de PL')),
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
    
def sankey(df, titre) : 
    label_node_start=list(df.origine.unique())
    label_node_end=[a for a in filter(lambda x: x[:6]!='Rocade',df.destination.unique())]

    df['pos_label_o']=df.apply(lambda x : label_node_start.index(x['origine']), axis=1)
    df['pos_label_d']=df.apply(lambda x : label_node_end.index(x['destination'])+len(label_node_start) 
                                           if x['destination'][:6]!='Rocade' else label_node_start.index(x['destination']), axis=1)

    dico_couleur={'A10':'blue', 'A63':'green', 'A89':'red', 'A62':'yellow', 'A660':'orange', 'N10':'purple', 'Rocade Est':'whitesmoke', 'Rocade Ouest':'darkgray'}
    liste_couleur=[dico_couleur[a] for a in label_node_start+label_node_end]

    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = label_node_start+label_node_end,
          color = liste_couleur
        ),
        link = dict(
          source = df.pos_label_o, # indices correspond to labels, eg A1, A2, A2, B1, ...
          target = df.pos_label_d,
          value = df.nb_pl
      ))])

    fig.update_layout({"title": {"text": titre,
                             "font": {"size": 30},'x':0.5}})
    return fig