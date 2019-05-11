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

def graph_nb_veh_jour_camera(df, date_d, date_f, camera=4) : 
    """
    pour creer des graph du nb de veh  par heue sur une journee à 1 camera
    en entree : 
        df : df des passages initiales, telle qu'importee depuis la bdd
        date_d : string : date de debut, generalement de la forme YYYY-MM-DD 00:00:00
        date_f : string : date de debut, generalement de la forme YYYY-MM-DD 23:59:59
        camera : integer : nume de la camera etudiee
    en sortie : 
        graph : chart altair avec en x l'heure et en y le nb de veh
    """
    test2=df.loc[(df['created'].between(date_d,date_f)) & 
                 (df['camera_id']==camera)]
    graph=alt.Chart(test2.set_index('created').resample('H').count().reset_index(),title=date_d+' cam '+ str(camera)).mark_bar().encode(
                   x='created',
                    y='immat' )
    return graph  

def graph_nb_veh_jour_camera_multi_j(df,date_debut,date_fin,cam,nb_jour): 
    """
    Regroupement de charts altair issues de graph_nb_veh_jour_camera sur plusieurs jours
    en entre :
        cf graph_nb_veh_jour_camera
        nb_jours : integer : nb de jours à concatener
    en sortie : 
        une chart altair concatenee verticalement avec un pour chaque jour
    """
    df_index_ok=df.reset_index()
    dico_graph={'graph'+str(indice):graph_nb_veh_jour_camera(df_index_ok, dates[0], dates[1], cam) 
               for indice,dates in enumerate(zip([str(x) for x in pd.date_range(date_debut, periods=nb_jour, freq='D')],
                                [str(x) for x in pd.date_range(date_fin, periods=nb_jour, freq='D')]))}
    liste_graph=[dico_graph[key] for key in dico_graph.keys()]
    return alt.VConcatChart(vconcat=(liste_graph))  
    
def temp_max_cluster(df_pl_ok, delai, coeff=4):
    """obtenir le temps max de parcours en faisant un cluster par dbscan
    on peut faire un cluster sur le couple date + tps de parcours (forme actuelle)
    ou en faire un unqieuemnt sur un ecart sur le tempsde parcour (en enlevant le commentaire devant matrice et en l'utilisant dans le fit
    en entree : la df des temps de parcours pl final
                le delai max pour regrouper en luster,en seconde
                coeff : entier : pour la partd'objet totaux à conserver pour faire un cluster
    en sortie : le nombre de clusters,
                un apndas timedelta
    """
    if df_pl_ok.empty:
        raise ClusterError()
    donnees_src=df_pl_ok.loc[:,['date_cam_1','tps_parcours']].copy() #isoler les données necessaires
    liste_valeur=donnees_src.tps_parcours.apply(lambda x : ((pd.to_datetime('2018-01-01')+x)-pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).tolist()#convertir les temps en integer
    liste_date=donnees_src.date_cam_1.apply(lambda x :(x - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).tolist()
    liste=[[liste_date[i],liste_valeur[i]] for i in range(len(liste_valeur))]
    if len(liste_valeur)<5 : #si il n'y a pas bcp de pl on arrete ; pourraitfair l'objet d'un parametre
        raise ClusterError()
    #mise en forme des données pour passer dans sklearn 
    matrice=np.array(liste_valeur).reshape(-1, 1)
    #faire tourner la clusterisation et recupérer le label (i.e l'identifiant cluster) et le nombre de cluster
    try :
        clustering=DBSCAN(eps=delai, min_samples=len(liste_valeur)/coeff).fit(liste)
    except ValueError :
        raise ClusterError()
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # A AMELIORER EN CREANT UNE ERREUR PERSONALISEE SI ON OBTIENT  CLUSTER
    if n_clusters_== 0 :
        raise ClusterError()
    #mettre en forme au format pandas
    results = pd.DataFrame(pd.DataFrame([donnees_src.index,labels]).T)
    results.columns = ['index_base', 'cluster_num']
    results = pd.merge(results,df_pl_ok, left_on='index_base', right_index=True )
    #obtenir un timedelta unique
    temp_parcours_max=results.loc[results.loc[:,'cluster_num']!=-1].groupby(['cluster_num'])['tps_parcours'].max().min()
    
    return n_clusters_, temp_parcours_max  

def verif_doublons_trajet(dico_od, destination):
    """
    fonction de vérification qu'un passage contenu dans un trajet n'est pas contenu dans un autre
    en entrée : dico_od : le dico des passagesde transit issu de la fonction  transit_temps_complet
                destination : string : destination du trajet (parmi les destination possibles dans liste_complete_trajet
    en sortie : une dataframe avec les passages en doublons
    """
    df_depart=dico_od.loc[dico_od['destination']==destination].copy()
    jointure=df_depart.merge(dico_od, on='immat')
    jointure=jointure.loc[jointure.date_cam_1_x!=jointure.date_cam_1_y].copy()
    df_doublons=(jointure.loc[((jointure.date_cam_1_y>=jointure.date_cam_1_x) & (jointure.date_cam_1_y<=jointure.date_cam_2_x)) |
                  ((jointure.date_cam_2_y>=jointure.date_cam_1_x) & (jointure.date_cam_2_y<=jointure.date_cam_2_x))])
    return df_doublons

def cam_adjacente(immat, date_cam_1, date_cam_2, o_d, df_immats, point_ref='A660') :
    """
    trouver la camera avant ou apres le passage à une origine ou destination
    en entree : 
        immat : immatriiculation issu du dico_od
        horodate : string ou pd.datetime :  au format YYYY-MM-DD HH:MM:SS issu du dico_od
        o_d : string : orgine et destination issu du dico_od
        df_immats : dataframes des immats concernées : limite le temps de traitement
    en sortie : 
        cam_adjacente : integer : le code de la camera proche, ou 0
        horodate_adjacente : pd.datetime ou pd.NaT
    """
    # il faudrait mettre ce dico en entree
    dico_coresp_od_cam={'A660':{'o':'19','d':'18'},
                        'N10':{'o':'6','d':'5'},
                        'A89':{'o':'8','d':'7'},
                        'A62':{'o':'10','d':'12'}}
    cam_immat=df_immats.loc[df_immats['immat']==immat].reset_index()#localiser les passages liés à l'immat
    camera_a660, coeff_index=(dico_coresp_od_cam[point_ref]['o'],-1) if o_d.split('-')[0]==point_ref else (dico_coresp_od_cam[point_ref]['d'],1)
    try : #dans le cas ou il n'y a pas de passage avant ou apres
        position_cam_adjacente=(cam_immat.loc[(cam_immat['created']==date_cam_1) & (cam_immat['camera_id']==camera_a660)].index[0]+coeff_index #trouver la position de la camera suivante
                            if o_d.split('-')[0]==point_ref else
                            cam_immat.loc[(cam_immat['created']==date_cam_2) & (cam_immat['camera_id']==camera_a660)].index[0]+coeff_index)
        if position_cam_adjacente==-1 : 
            return 0, pd.NaT 
        cam_adjacente=cam_immat.iloc[position_cam_adjacente]['camera_id']#la camera suivante
        horodate_adjacente=cam_immat.iloc[position_cam_adjacente]['created']
        return cam_adjacente, horodate_adjacente # et l'heure associées
    except IndexError : 
        return 0, pd.NaT
    
def cam_voisines(immat, date, camera, df) :
    """
    Retrouver les dates et camera de passages d'un vehicule avant et apres un passage donne
    en entree : 
        immat : string : immatribualtion
        date : string : date de passage
        camera : cam de passage
        df : df contenant tous les passages de immats concernees (df 3 semianes ou extraction)
    """
    passage_immat=df.loc[df['immat']==immat].reset_index().copy()
    idx=passage_immat.loc[(passage_immat['created']==date) & (passage_immat['camera_id']==camera)].index
    try :
        cam_suivant, date_suivant=passage_immat.shift(-1).iloc[idx]['camera_id'].values[0], passage_immat.shift(-1).iloc[idx]['created'].values[0]
    except IndexError :
        cam_suivant, date_suivant=0, pd.NaT
    try :
        cam_precedent, date_precedent=passage_immat.shift(1).iloc[idx]['camera_id'].values[0], passage_immat.shift(1).iloc[idx]['created'].values[0]
    except IndexError :
        cam_precedent, date_precedent=0, pd.NaT
    return cam_suivant,date_suivant, cam_precedent,date_precedent

def correction_trajet(df_3semaines, dico_od, voie_ref='A660', cam_ref_1=13, cam_ref_2=15, cam_ref_3=19) : 
    """
    Fonction qui va ré-assigner les origines-destinations à A63 si certanes conditions sont remplie : 
    cas 1 : vue a A660 puis dans l'autre sens sur A63, ou inversement
    cas 2 : vue sur A660 Nord-Sud, puis A660 Sud-Nord, avec plus de 1jd'écart entre les deux
    en entree : 
        df_3semaines : dataframe des passages
        dico_od : dataframe des o_d issue de transit_temps_complet
        voie_ref : string : nom de la voie que l'on souhaite changer
        cam_ref_1 : integer : camera de a changer pour le sens 1 (cas 1)
        cam_ref_2 : integer : camera de a changer pour le sens 2 (cas 1)
        cam_ref_3 : integer camera a changer dans les deux sens (cas2)
    en sortie : 
        dico_od_origine : dataframe des o_d issue de transit_temps_complet complétée et modifée
    """
    
    def MaJ_o_d(correctionType, o, d):
        """
        Fonction de mise à jour des o_d pour les trajets concernants A660 que l'on rabat sur A63
        """
        if correctionType : 
            if o=='A660' : 
                new_o, new_d, od='A63',d,'A63-'+d
            elif o!='A63': 
                new_o, new_d, od=o,'A63',o+'-A63'
            else : 
                new_o, new_d, od=o,d,o+'-'+d
        else : 
            new_o, new_d, od=o,d,o+'-'+d 
        return new_o, new_d, od
        
    #cas 1 : passer sur A660 et vu avant ou apres sur A63
    dico_od_origine=dico_od.copy()
    dico_od_copie=dico_od_origine.loc[(dico_od_origine['origine']==voie_ref) | (dico_od_origine['destination']==voie_ref)].reset_index().copy() #isoler les o_d liées au points en question
    df_immats=df_3semaines.loc[df_3semaines.immat.isin(dico_od_copie.immat.unique().tolist())] #limiter le df_3semaines aux immats concernée   df_adj=dico_od_copie.apply(lambda x : t.cam_adjacente(x['immat'],x['date_cam_1'],x['date_cam_2'],x['o_d'],df_immats),axis=1, result_type='expand') #construire les colonnes de camera adjacente et de temps adjacent 
    df_adj_cam1=dico_od_copie.apply(lambda x : cam_voisines(x['immat'],x['date_cam_1'],x['cameras'][0],df_immats),axis=1, result_type='expand') #construire les colonnes de camera adjacente et de temps adjacent 
    df_adj_cam1.columns=['cam_suivant','date_suivant','cam_precedent1','date_precedent1']
    df_adj_cam1.drop(['cam_suivant','date_suivant'], axis=1, inplace=True)
    df_adj_cam2=dico_od_copie.apply(lambda x : cam_voisines(x['immat'],x['date_cam_2'],x['cameras'][-1],df_immats),axis=1, result_type='expand') #construire les colonnes de camera adjacente et de temps adjacent 
    df_adj_cam2.columns=['cam_suivant2','date_suivant2','cam_precedent','date_precedent']
    df_adj_cam2.drop(['cam_precedent','date_precedent'], axis=1, inplace=True)
    dico_od_copie_adj=pd.concat([dico_od_copie,df_adj_cam1,df_adj_cam2],axis=1)
    #on creer une df de correction 
    dico_od_a_corrige_s_n=dico_od_copie_adj.loc[(dico_od_copie_adj['origine']=='A660') & (dico_od_copie_adj['cam_precedent1']==cam_ref_1)].copy()#recherche des lignes pour lesquelles origine=A660 et camera adjacente = 13 ou destination=A660 et et camera_adjacente = 15
    dico_od_a_corrige_s_n['temps_passage']=dico_od_a_corrige_s_n['date_cam_1']-dico_od_a_corrige_s_n['date_precedent1']#calcul du timedelta
    dico_od_a_corrige_n_s=dico_od_copie_adj.loc[(dico_od_copie_adj['destination']=='A660') & (dico_od_copie_adj['cam_suivant2']==cam_ref_2)].copy()
    dico_od_a_corrige_n_s['temps_passage']=dico_od_a_corrige_n_s['date_suivant2']-dico_od_a_corrige_n_s['date_cam_2']
    dico_temp=pd.concat([dico_od_a_corrige_n_s,dico_od_a_corrige_s_n])
    dico_correction=dico_temp.loc[~dico_temp.temps_passage.isna()]#on ne conserve que les ligne qui ont un timedelta !=NaT 
    dico_od_origine['correction_o_d']=False #création de l'attribut drapeau modification des o_d
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','cameras']).index.isin(dico_correction.set_index(['date_cam_1','immat','cameras']).index),
                       'correction_o_d']=True #mise à jour de l'attribut drapeau
    #mise à jour des  3 attributs liées aux o_d
    dico_od_origine.loc[dico_od_origine['correction_o_d'],'origine']=dico_od_origine.loc[dico_od_origine['correction_o_d']].apply(lambda x : MaJ_o_d(x['correction_o_d'], x['origine'],x['destination'])[0],axis=1)
    dico_od_origine.loc[dico_od_origine['correction_o_d'],'destination']=dico_od_origine.loc[dico_od_origine['correction_o_d']].apply(lambda x : MaJ_o_d(x['correction_o_d'], x['origine'],x['destination'])[1],axis=1)
    dico_od_origine.loc[dico_od_origine['correction_o_d'],'o_d']=dico_od_origine.loc[dico_od_origine['correction_o_d']].apply(lambda x : MaJ_o_d(x['correction_o_d'], x['origine'],x['destination'])[2],axis=1) 
    
    #cas 2 : passer sur A660 Nord-Sud puis Sud-Nord avec au moins 1 jour d'écart
    dico_od_copie=dico_od_origine.loc[(dico_od_origine['destination']=='A660')].reset_index().copy()
    
    df_adj_cam2=dico_od_copie.apply(lambda x : cam_voisines(x['immat'],x['date_cam_2'],x['cameras'][-1],df_immats),axis=1, result_type='expand') #construire les colonnes de camera adjacente et de temps adjacent 
    df_adj_cam2.columns=['cam_suivant','date_suivant','cam_precedent','date_precedent']
    df_adj_cam2.drop(['cam_precedent','date_precedent'], axis=1, inplace=True)
    dico_od_copie_adj=pd.concat([dico_od_copie,df_adj_cam1,df_adj_cam2],axis=1)
    dico_od_a_corrige=dico_od_copie_adj.loc[dico_od_copie_adj['cam_suivant']==cam_ref_3].copy()#filtrer les résultats sur la cameras de fin
    dico_od_a_corrige['temps_passage']=dico_od_a_corrige['date_suivant']-dico_od_a_corrige['date_cam_2']#calcul du temps de passages entre les cameras
    dico_filtre=dico_od_a_corrige.loc[dico_od_a_corrige['temps_passage']>=pd.Timedelta('1 days')]
    
    #pour les lignes ayant 1 temps de passage sup à 1 jour, on va réaffecter d à A63
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','date_cam_2']).index.isin(
    dico_filtre.set_index(['date_cam_1','immat','date_cam_2']).index.tolist()),'destination']='A63'
    #on modifie o_d aussi
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','date_cam_2']).index.isin(
    dico_filtre.set_index(['date_cam_1','immat','date_cam_2']).index.tolist()),'o_d']=(dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','date_cam_2']).index.isin(
    dico_filtre.set_index(['date_cam_1','immat','date_cam_2']).index.tolist())].apply(lambda x : x['origine']+'-'+x['destination'], axis=1))
    #puis on met à jour correction_o_d
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','date_cam_2']).index.isin(
    dico_filtre.set_index(['date_cam_1','immat','date_cam_2']).index.tolist()),'correction_o_d']=True
    #pour les lignes ayant 1 temps de passage sup à 1 jour, on va réaffecter o à A63
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat']).index.isin(
    dico_filtre.set_index(['date_suivant','immat']).index.tolist()),'origine']='A63'
    #on modifie o_d aussi
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat']).index.isin(
    dico_filtre.set_index(['date_suivant','immat']).index.tolist()),'o_d']=(dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat']).index.isin(
    dico_filtre.set_index(['date_suivant','immat']).index.tolist())].apply(lambda x : x['origine']+'-'+x['destination'], axis=1))
    #puis on met à jour correction_o_d
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat']).index.isin(
    dico_filtre.set_index(['date_suivant','immat']).index.tolist()),'correction_o_d']=True

    return dico_od_origine

def corriger_df_tps_parcours (dico_tps_max):
    """ fonction de correction de la df_tps_parcours issue de transit_temps_complet.
    On moyenne les valuers de temps de parcours de type '85_percentile' si encadrer par des Cluster
    en entree : 
        dico_tps_max issu de transit_temps_complet
    en sortie :
        dico_tps_max modifiée
    """
    def moyenne_tps_85pct(type_la, type_avant, type_apres, tps,tpsla) : 
        if type_la=='85eme_percentile' and type_avant=='Cluster' and type_apres=='Cluster' : 
            return tps,'moyenne Cluster'
        else : return tpsla,type_la
    
    dico_tps_max2=dico_tps_max.reset_index().drop('index',axis=1).sort_values(['o_d','date']).copy()
    dico_tps_max2['tps']=(dico_tps_max2.temps.shift(1)+dico_tps_max2.temps.shift(-1)) / 2
    dico_tps_max2['type_tps_1']=dico_tps_max2.type.shift(1)
    dico_tps_max2['type_tps_2']=dico_tps_max2.type.shift(-1)
    dico_tps_max2['temps']=dico_tps_max2.apply(lambda x : moyenne_tps_85pct(x['type'], x['type_tps_1'],x['type_tps_2'], x['tps'], x['temps'])[0],axis=1)
    dico_tps_max2['type']=dico_tps_max2.apply(lambda x : moyenne_tps_85pct(x['type'], x['type_tps_1'],x['type_tps_2'], x['tps'], x['temps'])[1],axis=1)
    dico_tps_max2.drop(['tps','type_tps_1','type_tps_2'],axis=1, inplace=True)
    
    return dico_tps_max2

    