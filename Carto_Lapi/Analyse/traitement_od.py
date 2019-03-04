# -*- coding: utf-8 -*-
'''
Created on 27 fev. 2019
@author: martin.schoreisz

Module de traitement des donnees lapi

'''
"""TO DO : 
faire le filtre des données dont e type est -1
caractériser les incertitude dans les df : %de fiabilite pourrie, % de type de vehicule =-1
"""

import matplotlib #pour éviter le message d'erreurrelatif a rcParams
import pandas as pd
import Connexion_Transfert as ct

def ouvrir_fichier_lapi(date_debut, date_fin) : 
    with ct.ConnexionBdd('gti_lapi') as c : 
        requete=f"select id, camera_id, created, immat, fiability, l, state from data.te_passage where created between '{date_debut}' and '{date_fin}'"
        df=pd.read_sql_query(requete, c.sqlAlchemyConn)
        return df
        

def df_temps_parcours_moyen(df, date_debut, duree, camera1, camera2):
    """fonction de calcul du temps moyen de parcours entre 2 cameras
    en entree : dataframe : le dataframe format pandas qui contient les données
                date_debut : string decrivant une date avec Y M D H M S : 
                 date de part d'analyse du temp moyen           
                duree : integer : duree en minute
                 c'est le temps entre lequel on va regarder le temps mpyen : de 7h à 7h et 30 min par exemple
                camera1 : integer : camera de debut
                camera2 : integer : camera de fin
    en sortie : une dataframe tout vehicule confondus 
                une dataframe vl
                une dataframe pl
                attributs : horaires de passage, camera_ids, fiabilites des 2 cameras,  champs de diff Vl-PL, pays, tps parcours
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
    #creer la colonne des differentiel de temps
    cam1_cam2_passages_filtres['tps_parcours']=cam1_cam2_passages_filtres['created_y']-cam1_cam2_passages_filtres['created_x'] 
    
    #isoler les vl
    tps_parcours_vl=cam1_cam2_passages_filtres.loc[cam1_cam2_passages_filtres.loc[:,'l_x']==0]
    tps_parcours_pl=cam1_cam2_passages_filtres.loc[cam1_cam2_passages_filtres.loc[:,'l_x']==1]
    
    return cam1_cam2_passages_filtres, tps_parcours_vl, tps_parcours_pl

def filtrer_fiabilite_plaque(df, fiabilite):
    """
    filtrer des df selon la fiabilite de la plaque d'immatriculation
    en entre : fiabilite : integer : entre 0 le moins fiable et 100 le plus fiable
    en sortie : df filtree sans les plques les moins fiables
    """
    liste_nom_col=[element for element in df.columns if element[:9]=='fiability']
    test_fibb=df[liste_nom_col].copy()
    # amodifier pour obtenir une liste avec que des true la ou aucun des valeur <50
    #test.ge(50).any(axis=1)
    

"""
CREER DES VISUALISATION
#pour faire un plot dans altair via le jupyter lab
#on isol les colonnes de date et des tps de parcours dans une nouvelle df
pour_image=cam1_cam2_passages_filtres[['created_x','tps_parcours']].copy()
#on converti le timedelta en datetime relatif à un jour a partir de minuit, car altair (JSON en fait) ne gere pas les timedeltas
pour_image.tps_parcours=pd.to_datetime('2018-01-01')+tps_parcours
#on crée le chart de altair
chart = alt.Chart(pour_image)
#cree l'image
alt.Chart(pour_image).mark_point().encode(
    x='created_x',
    y='hoursminutes(tps_parcours)'
)
"""
    