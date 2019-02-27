# -*- coding: utf-8 -*-
'''
Created on 27 fev. 2019
@author: martin.schoreisz

Module de traitement des donnees lapi

'''

import pandas as pd

def temps_parcours_moyen(df, date_debut, duree, camera1, camera2):
    """fonction de calcul du temps moyen de parcours entre 2 cameras
    en entree : dataframe : le dataframe format pandas qui contient les données
                date_debut : string decrivant une date avec Y M D H M S
                duree : integer : duree en minute
                camera1 : integer : camera de debut
                camera2 : integer : camera de fin
    """
    date_debut=pd.to_datetime(date_debut)
    date_fin=date_debut+pd.Timedelta(minutes=duree)#creer une date 30 min plus tard que la date de départ
    df_duree=df.loc[date_debut:date_fin]#filtrer la df #filtrer la df sur 30 min
    
    #trouver tt les bagnoles passée par cam1 dont la 2eme camera est cam2
    df_duree_cam1=df_duree.loc[df_duree.loc[:,'camera']==camera1] #isoler camera 1
    df_duree_autres_cam=df.loc[df.loc[:,'imat'].isin(df_duree_cam1.loc[:,'imat'])]#on retrouve ces imatriculation
    cam1_croise_autre_cam=df_duree_cam1.reset_index().merge(df_duree_autres_cam.reset_index(), how='left', on='imat') #on fait une jointure entre cam 1 et les autres cam pour avoir une correspondance entre le passage devan la 1ere cmaera et la seconde
    cam1_croise_suivant=cam1_croise_autre_cam.loc[(cam1_croise_autre_cam.loc[:,'index_x']<cam1_croise_autre_cam.loc[:,'index_y'])&(cam1_croise_autre_cam.loc[:,'camera_y']==8)]#on ne garde que les passages à la 2ème caméra postérieur au passage à la première
    cam1_group_cam2=cam1_croise_suivant.loc[cam1_croise_suivant.groupby(['imat'])['index_y'].idxmin()] #on isole le passage le plus rapide pour chaque immatriculation
    cam1_cam2_passages=cam1_group_cam2.set_index('index_y').sort_index()
    cam1_cam2_passages_filtres=cam1_cam2_passages[date_debut:date_debut+pd.Timedelta(hours=2)]#et on ajoute un tri sur les dates possibles genre pas plus de 2h après
    cam1_cam2_passages_filtres=cam1_cam2_passages_filtres.reset_index()
    cam1_cam2_passages_filtres['diff']=cam1_cam2_passages_filtres['index_y']-cam1_cam2_passages_filtres['index_x'] #creer la colonne des differentiel de temps
    resultat=cam1_cam2_passages_filtres['diff'].mean() #calcul du temps moyen #on pourrait aussi faire des calculs sur des precnetiles ou autres
    