# -*- coding: utf-8 -*-
'''
Created on 27 fev. 2019
@author: martin.schoreisz

Module de traitement des donnees lapi

'''

import matplotlib #pour éviter le message d'erreurrelatif a rcParams
import pandas as pd
import geopandas as gp
from Martin_Perso import Connexion_Transfert as ct

def ouvrir_fichier_lapi(date_debut, date_fin) : 
    with ct.ConnexionBdd('gti_lapi') as c : 
        requete=f"select id, camera_id, created, immat, fiability, l, state from data.te_passage where created between '{date_debut}' and '{date_fin}'"
        df=pd.read_sql_query(requete, c.sqlAlchemyConn)
        return df
        

def temps_parcours_moyen(df, date_debut, duree, camera1, camera2):
    """fonction de calcul du temps moyen de parcours entre 2 cameras
    en entree : dataframe : le dataframe format pandas qui contient les données
                date_debut : string decrivant une date avec Y M D H M S : 
                 date de part d'analyse du temp moyen           
                duree : integer : duree en minute
                 c'est le temps entre lequel on va regarder le temps mpyen : de 7h à 7h et 30 min par exemple
                camera1 : integer : camera de debut
                camera2 : integer : camera de fin
    """
    df=df.set_index('created').sort_index()
    date_debut=pd.to_datetime(date_debut)
    date_fin=date_debut+pd.Timedelta(minutes=duree)#creer une date 30 min plus tard que la date de départ
    df_duree=df.loc[date_debut:date_fin]#filtrer la df #filtrer la df sur 30 min
    
    #trouver tt les bagnoles passée par cam1 dont la 2eme camera est cam2
    #isoler camera 1
    df_duree_cam1=df_duree.loc[df_duree.loc[:,'camera_id']==camera1]
    #on retrouve ces immatriculation
    df_duree_autres_cam=df.loc[df.loc[:,'immat'].isin(df_duree_cam1.loc[:,'immat'])]
    #on fait une jointure entre cam 1 et les autres cam pour avoir une correspondance entre le passage devan la 1ere cmaera et la seconde
    cam1_croise_autre_cam=df_duree_cam1.reset_index().merge(df_duree_autres_cam.reset_index(), how='left', on='immat')
    #on ne garde que les passages à la 2ème caméra postérieur au passage à la première
    cam1_croise_suivant=cam1_croise_autre_cam.loc[(cam1_croise_autre_cam.loc[:,'created_x']<cam1_croise_autre_cam.loc[:,'created_y'])]
    #on isole le passage le plus rapide devant cam suivante pour chaque immatriculation
    cam1_fastest_next=cam1_croise_suivant.loc[cam1_croise_suivant.groupby(['immat'])['created_y'].idxmin()]
    #on ne garde que les passage le plus rapide devant la camera 2
    cam1_puis_cam2=cam1_fastest_next.loc[cam1_fastest_next.loc[:,'camera_id_y']==camera2]
    #on trie puis on ajoute un filtre surle temps entre les 2 camera. !!! CE FILTRE EST A DEFINIR ET POURRAIT PASSER EN PARAMETRE FONCTION
    cam1_cam2_passages=cam1_puis_cam2.set_index('created_y').sort_index()
    cam1_cam2_passages_filtres=cam1_cam2_passages[date_debut:date_debut+pd.Timedelta(hours=2)]
    #on ressort la colonne de tempsde l'index et on cree la colonne des differentiel de temps
    cam1_cam2_passages_filtres=cam1_cam2_passages_filtres.reset_index()
    cam1_cam2_passages_filtres['diff']=cam1_cam2_passages_filtres['created_y']-cam1_cam2_passages_filtres['created_x'] #creer la colonne des differentiel de temps
    resultat=cam1_cam2_passages_filtres['diff'].mean() #calcul du temps moyen #on pourrait aussi faire des calculs sur des precnetiles ou autres
    return resultat
    