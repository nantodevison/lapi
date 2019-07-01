# -*- coding: utf-8 -*-
'''
Created on 1 juil. 2019

@author: martin.schoreisz
Moduel avec des fonctions pour traiter les resultats : import / exports en json, pourcentage PL par acm, passage fictif rocade...
'''
import pandas as pd

def pourcentage_pl_camera(df_pl,dico_passag):
    """
    fonction de regroupement des nb de vl, pl, et pl en trasit, par heure et par camera
    en entree : 
        df_pl : df des passages pl
        dico_passag : dico des passages PL de transit
        df_vl : df des passages vl
    en sortie : 
        jointure_pct_pl : df allant servir pour representation graphique : 
            colonnes : created, camera_id, nb_veh, type, pct_pl_transit
    """
    def pct_pl(a,b):
        try :
            return a*100/b
        except ZeroDivisionError : 
            return 0
    
    df_synthese_pl_tot=df_pl.groupby('camera_id').resample('H').count()['immat'].reset_index().rename(columns={'immat':'nb_veh'})
    df_synthese_pl_transit=dico_passag.set_index('created').groupby('camera_id').resample('H').count()['immat'].reset_index().rename(
            columns={'immat':'nb_veh'})
    
    df_synthese_pl_tot['type']='PL total'
    df_synthese_pl_transit['type']='PL transit'
    
    df_pct_pl_transit=df_synthese_pl_tot.merge(df_synthese_pl_transit, on=['camera_id','created']).rename(columns={'nb_veh_x':'nb_pl_tot',
                                                                                            'nb_veh_y':'nb_pl_transit'})
    df_pct_pl_transit['pct_pl_transit']=df_pct_pl_transit.apply(lambda x : pct_pl(x['nb_pl_transit'],x['nb_pl_tot']) ,axis=1)
    df_concat_pl=pd.concat([df_synthese_pl_tot,df_synthese_pl_transit],sort=False)
    return df_concat_pl.merge(df_pct_pl_transit[['created','camera_id','pct_pl_transit']], on=['created','camera_id'])

def passages_fictif_rocade (liste_trajet, df_od,df_passages_transit,df_pl):
    """
    Cr�er des passages pour les trajets de transit non vus sur la Rocade mais qui y sont pass�
    en entree : 
        liste_trajet : df des trajets concernes, issu de liste_trajet_rocade
        df_od : df des trajetsde transit valid� selon le temps de parcours
        df_passages_transit : df des passages concern�s par un trajet de transit (issu du traitement o_d)
        df_pl : df de tout passages pl (issu simplement de l'import mise en forme)
    en sortie : 
        df_passage_transit_redresse : df des passages concern�s par un trajet de transit (issu du traitement o_d) + passages fictifs
        df_pl_redresse : df de tout passages pl + passages fictifs
        trajets_rocade_non_vu : df des passgaes fictifs
    """
    def camera_fictive(cam1, cam2) : 
        """
        Connaitre la camera a affectee selon le trajet parcouru
        """
        if cam1 in [15,10,19] and cam2 in [5,11,7] : 
            return 4
        elif cam1 in [12,8,6] and cam2 in [13,9,18] :
            return 3
        else : 
            return -1
    #rechercher les trajets dans le dico des o_d
    trajets_rocade=df_od.loc[df_od.o_d.isin(liste_trajet.trajets.tolist())]
    #trouver ceux qui ne contiennent aucune r�f�rence uax camera de la Rocade
    trajets_rocade_non_vu=trajets_rocade.loc[trajets_rocade.apply(lambda x : all(e not in x['cameras'] for e in [1,2,3,4]),axis=1)].copy()
    #cr�er des passage fictif au niveau de la Rocade avec comme created la moyenne entre date_cam_1 et date_cam_2
    trajets_rocade_non_vu['created_fictif']=trajets_rocade_non_vu.apply(lambda x : x['date_cam_1']+((x['date_cam_2']-x['date_cam_1'])/2),axis=1)
    trajets_rocade_non_vu['camera_fictif']=trajets_rocade_non_vu.apply(lambda x : camera_fictive(x['cameras'][0],x['cameras'][1]),axis=1)
    #virere clolonne inutiles
    trajets_rocade_non_vu=trajets_rocade_non_vu.drop(['date_cam_1', 'index','id', 'date_cam_2',
           'cameras', 'origine', 'destination', 'o_d', 'tps_parcours', 'period',
           'date', 'temps', 'type', 'tps_parcours_theoriq', 'filtre_tps'],axis=1)
    trajets_rocade_non_vu.rename(columns={'created_fictif':'created','camera_fictif':'camera_id'},inplace=True)
    #on ne garde que les trajets concernes par une des cameras fictive de la rocade
    trajets_rocade_non_vu=trajets_rocade_non_vu.loc[trajets_rocade_non_vu['camera_id']!=-1]
    #on ajoute les trajets ainsi cree aux autres (pl en transit et pl normaux)
    df_passage_transit_redresse=pd.concat([trajets_rocade_non_vu,df_passages_transit],axis=0,sort=False)
    df_pl_redresse=pd.concat([trajets_rocade_non_vu.set_index('created'),df_pl],axis=0,sort=False)
    #attributs de tracage
    df_passage_transit_redresse['fictif']=df_passage_transit_redresse.apply(lambda x : 'Rocade' if not x['fiability']>0 else 'Non' ,axis=1)
    df_passage_transit_redresse['fiability']=df_passage_transit_redresse.apply(lambda x : 999 if not x['fiability']>0 else x['fiability'],axis=1)
    return df_passage_transit_redresse, df_pl_redresse, trajets_rocade_non_vu

def save_donnees(df, fichier):
    """
    sauvegarder les df dans des fichiers.
    en entree : 
        df : la dtatfarme � sauvegarder
        fichier : raw string du nom de fichier
    """
    "reset index si c'est un datetime[ns]"
    if df.index.dtype=='<M8[ns]' : 
        fichier_export=df.reset_index().copy()
    else : 
        fichier_export=df.copy()
    #suppr les colonnes qui servent   rine si elles existent
    if 'index' in fichier_export.columns :
        fichier_export.drop('index',axis=1, inplace=True)
    if 'level_0' in fichier_export.columns :
        fichier_export.drop('level_0',axis=1, inplace=True)
    #passer les datetime et texte lisible par pd.to_datetime
    list_attr_datetime=[attr for attr in fichier_export.columns if fichier_export[attr].dtypes in ['<m8[ns]','<M8[ns]']]
    for attr in list_attr_datetime : 
        fichier_export[attr]=fichier_export[attr].apply(lambda x : str(x))
    
    fichier_export.to_json(fichier, orient='index')
        
def ouvrir_donnees(fichier):
    """
    ouvrir les donées suvegradees avec  save_donnees 
    en entree : 
        fichier : raw string du fichier à ouvrir
    en sortie : 
        df : dataframe avec les attributs aux formats datetime et timedelta
    """
    df=pd.read_json(fichier,orient='index')
    for attr in ['date_cam_1','date_cam_2','tps_parcours','date','temps','tps_parcours_theoriq','temps_filtre','created'] : 
        if attr in df.columns : 
            if attr in ['date_cam_1','date_cam_2','date','created'] : 
                df[attr]=df[attr].apply(lambda x : pd.to_datetime(x))
            else :
                df[attr]=df[attr].apply(lambda x : pd.Timedelta(x))
    
    return df   
        
        
        
        
        
        
        
