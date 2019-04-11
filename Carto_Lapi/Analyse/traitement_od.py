# -*- coding: utf-8 -*-
'''
Created on 27 fev. 2019
@author: martin.schoreisz

Module de traitement des donnees lapi

'''

import matplotlib #pour éviter le message d'erreurrelatif a rcParams
import pandas as pd
import geopandas as gp
import numpy as np
import Connexion_Transfert as ct
import altair as alt
import os, datetime as dt
from sklearn.cluster import DBSCAN

dico_renommage={'created_x':'date_cam_1', 'created_y':'date_cam_2'}
fichier_trajet=(pd.DataFrame([{'origine':'A63','destination':'A10','cam_o':15, 'cam_d':11, 'trajets':[
                                                        {'cameras':[15,19,4,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[15,19,1,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[15,4,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[15,1,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[15,19,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[15,19,4,11],'type_trajet':'indirect'},
                                                        {'cameras':[15,19,1,11],'type_trajet':'indirect'},
                                                        {'cameras':[15,4,11],'type_trajet':'indirect'},
                                                        {'cameras':[15,1,11],'type_trajet':'indirect'},
                                                        {'cameras':[15,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[15,19,11],'type_trajet':'indirect'},
                                                        {'cameras':[15,11],'type_trajet':'direct'}
                                                       ]},
                            {'origine':'A10','destination':'A63','cam_o':12, 'cam_d':13,'trajets':[{'cameras':[12,6,2,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[12,6,3,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[12,6,2,13],'type_trajet':'indirect'},
                                                        {'cameras':[12,6,3,13],'type_trajet':'indirect'},
                                                        {'cameras':[12,6,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[12,2,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[12,3,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[12,2,13],'type_trajet':'indirect'},
                                                        {'cameras':[12,3,13],'type_trajet':'indirect'},
                                                        {'cameras':[12,6,13],'type_trajet':'indirect'},
                                                        {'cameras':[12,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[12,13],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'A63','destination':'N10','cam_o':15, 'cam_d':5,'trajets':[{'cameras':[15,19,4,5],'type_trajet':'indirect'},
                                                        {'cameras':[15,19,1,5],'type_trajet':'indirect'},
                                                        {'cameras':[15,4,5],'type_trajet':'indirect'},
                                                        {'cameras':[15,1,5],'type_trajet':'indirect'},
                                                        {'cameras':[15,19,5],'type_trajet':'indirect'}, 
                                                        {'cameras':[15,5],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'N10','destination':'A63','cam_o':6, 'cam_d':13,'trajets':[{'cameras':[6,2,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[6,3,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[6,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[6,2,13],'type_trajet':'indirect'},
                                                        {'cameras':[6,3,13],'type_trajet':'indirect'},
                                                        {'cameras':[6,13],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'A62','destination':'A10','cam_o':10, 'cam_d':11,'trajets':[{'cameras':[10,4,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[10,4,11],'type_trajet':'indirect'},
                                                        {'cameras':[10,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[10,11],'type_trajet':'direct'}
                                                       ]},
                            {'origine':'A10','destination':'A62','cam_o':12, 'cam_d':9,'trajets':[{'cameras':[12,6,3,9],'type_trajet':'indirect'},
                                                        {'cameras':[12,3,9],'type_trajet':'indirect'},
                                                        {'cameras':[12,6,9],'type_trajet':'indirect'},
                                                        {'cameras':[12,9],'type_trajet':'direct'}
                                                       ]},
                            {'origine':'A62','destination':'N10','cam_o':10, 'cam_d':5,'trajets':[{'cameras':[10,4,5],'type_trajet':'indirect'},
                                                        {'cameras':[10,5],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'N10','destination':'A62','cam_o':6, 'cam_d':9,'trajets':[{'cameras':[6,3,9],'type_trajet':'indirect'},
                                                        {'cameras':[6,9],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'A63','destination':'A62','cam_o':15, 'cam_d':9,'trajets':[{'cameras':[15,19,9],'type_trajet':'indirect'},
                                                        {'cameras':[15,9],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'A62','destination':'A63','cam_o':10, 'cam_d':13,'trajets':[{'cameras':[10,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[10,13],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'A89','destination':'A63','cam_o':8 ,'cam_d':13,'trajets':[{'cameras':[8,3,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[8,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[8,3,13],'type_trajet':'indirect'},
                                                        {'cameras':[8,13],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'A63','destination':'A89','cam_o':15, 'cam_d':7,'trajets':[{'cameras':[15,19,4,7],'type_trajet':'indirect'},
                                                        {'cameras':[15,4,7],'type_trajet':'indirect'},
                                                        {'cameras':[15,19,7],'type_trajet':'indirect'},
                                                        {'cameras':[15,7],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'A89','destination':'A62','cam_o':8, 'cam_d':9,'trajets':[{'cameras':[8,3,9],'type_trajet':'indirect'},
                                                        {'cameras':[8,9],'type_trajet':'direct'}
                                                       ]},
                            {'origine':'A62','destination':'A89','cam_o':10, 'cam_d':7,'trajets':[{'cameras':[10,4,7],'type_trajet':'indirect'},
                                                        {'cameras':[10,7],'type_trajet':'direct'}
                                                       ]},
                            {'origine':'A89','destination':'A10','cam_o':8, 'cam_d':11,'trajets':[{'cameras':[8,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[8,11],'type_trajet':'direct'}
                                                       ]},
                            {'origine':'A10','destination':'A89','cam_o':12, 'cam_d':7,'trajets':[{'cameras':[12,6,7],'type_trajet':'indirect'},
                                                        {'cameras':[12,7],'type_trajet':'direct'}
                                                       ]},
                            {'origine':'A89','destination':'N10','cam_o':8, 'cam_d':5,'trajets':[{'cameras':[8,5],'type_trajet':'direct'}
                                                       ]},
                            {'origine':'N10','destination':'A89','cam_o':6, 'cam_d':7,'trajets':[{'cameras':[6,7],'type_trajet':'direct'}
                                                       ]},
                            {'origine':'A10','destination':'A660','cam_o':12, 'cam_d':18,'trajets':[{'cameras':[12,6,2,18],'type_trajet':'indirect'},
                                                         {'cameras':[12,6,3,18],'type_trajet':'indirect'},
                                                         {'cameras':[12,2,18],'type_trajet':'indirect'},
                                                         {'cameras':[12,3,18],'type_trajet':'indirect'},
                                                         {'cameras':[12,6,18],'type_trajet':'indirect'},
                                                         {'cameras':[12,18],'type_trajet':'direct'},
                                                        ]},
                            {'origine':'A660','destination':'A10','cam_o':19, 'cam_d':18,'trajets':[{'cameras':[19,4,5,11],'type_trajet':'indirect'},
                                                         {'cameras':[19,1,5,11],'type_trajet':'indirect'},
                                                         {'cameras':[19,1,11],'type_trajet':'indirect'},
                                                         {'cameras':[19,4,11],'type_trajet':'indirect'},
                                                         {'cameras':[19,5,11],'type_trajet':'indirect'},
                                                         {'cameras':[19,11],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'A660','destination':'A62','cam_o':19, 'cam_d':9,'trajets':[{'cameras':[19,9],'type_trajet':'direct'}
                                                        ]},
                            {'origine':'A62','destination':'A660','cam_o':10, 'cam_d':18,'trajets':[{'cameras':[10,18],'type_trajet':'direct'}
                                                        ]},
                            {'origine':'A660','destination':'A89','cam_o':19, 'cam_d':7,'trajets':[{'cameras':[19,4,7],'type_trajet':'indirect'},
                                                         {'cameras':[19,7],'type_trajet':'direct'}
                                                        ]},
                            {'origine':'A89','destination':'A660','cam_o':8, 'cam_d':18,'trajets':[{'cameras':[8,3,18],'type_trajet':'indirect'},
                                                         {'cameras':[8,18],'type_trajet':'direct'}
                                                        ]},
                            {'origine':'N10','destination':'A660','cam_o':6, 'cam_d':18,'trajets':[{'cameras':[6,2,18],'type_trajet':'indirect'},
                                                         {'cameras':[6,3,18],'type_trajet':'indirect'},
                                                         {'cameras':[6,18],'type_trajet':'direct'},
                                                        ]},
                            {'origine':'A660','destination':'N10','cam_o':19, 'cam_d':5,'trajets':[{'cameras':[19,1,5],'type_trajet':'indirect'},
                                                         {'cameras':[19,4,5],'type_trajet':'indirect'},
                                                         {'cameras':[19,5],'type_trajet':'direct'},
                                                        ]}
                           ]))[['origine', 'destination', 'cam_o', 'cam_d','trajets']]
liste_complete_trajet=pd.DataFrame([[tuple(liste_cam['cameras']),value[0],value[1],value[2],value[3],len(liste_cam['cameras'])] for obj, value in fichier_trajet.iterrows() for liste_cam in value[4]],
             columns=['cameras','origine','destination','cam_o','cam_d','nb_cams'])
liste_complete_trajet=liste_complete_trajet.sort_values('nb_cams',ascending=False)

def ouvrir_fichier_lapi(date_debut, date_fin) : 
    """ouvrir les donnees lapi depuis la Bdd 'lapi' sur le serveur partage GTI
    l'ouvertur se fait par appel d'une connexionBdd Python (scripts de travail ici https://github.com/nantodevison/Outils/blob/master/Outils/Martin_Perso/Connexion_Transfert.py)
    en entree : date_debut : string de type YYYY-MM-DD hh:mm:ss
                date_fin: string de type YYYY-MM-DD hh:mm:ss
    en sortie : dataframe pandas
    """
    with ct.ConnexionBdd('gti_lapi') as c : 
        requete=f"select case when camera_id=13 or camera_id=14 then 13 when camera_id=15 or camera_id=16 then 15 else camera_id end::integer as camera_id , created, immat, fiability, l, state from data.te_passage where created between '{date_debut}' and '{date_fin}'"
        df=pd.read_sql_query(requete, c.sqlAlchemyConn)
        return df


class trajet():
    """
    classe permettant le calcul de trajet : 
    - direct (2 cameras : debut et fin uniquement) à un horaire de depart donne. le veh passe cam 1 puis de suite cam2 
    - indirect (plus de 2 cameras : toute les cameras parcourue) à un horaire de depart donne. Le veh passe a chacune des cameras, dans l'ordre
    - global (2 cameras : debut et fin uniquement) à un horaire de depart donne. le veh part de cam1 et arrive a cam2, selon une liste d etrajet possible. cf fichier XXXXX
    Attributs : 
        df -- une pandas df issue de ouvrir_fichier_lapi ou de tout autre slice ou copy
        date_debut -- string de type YYYY-MM-DD hh:mm:ss ou pandas datetime -- date de debut du passage à la 1ere cam
        duree -- integer -- duree en minute pour periode de passage devant la première camera
        typeTrajet -- type de trajet -- Direct, Indirect, Global. le nombre de Camera est lié et fonction de ce paramètre
        df_filtre -- pandas dataframe -- permetde filtrer les données selonles passages déjà traites. en typeTrajet='Global unqiuement'
        temps_max_autorise -- le temps que l'on autorise pour trouver les vehicules passés par cam1. en typeTrajet='Direct' ou Indirect uniquement
    """
    
    def __init__(self,df,date_debut, duree, cameras,typeTrajet='Direct', df_filtre=None,temps_max_autorise=18) :
        
        #en fonction du df qui est passé on met la date de creation en index ou non
        if isinstance(df.index,pd.DatetimeIndex) :
            self.df=df
        else :
            self.df=df.set_index('created').sort_index()
        
        if self.df.empty:    
            raise PasDePlError()
        
        self.date_debut, self.duree, self.cameras_suivantes, self.temps_max_autorise=pd.to_datetime(date_debut), duree, cameras,temps_max_autorise
        self.date_fin=self.date_debut+pd.Timedelta(minutes=self.duree)
        self.df_duree=self.df.loc[self.date_debut:self.date_fin]  
    
        if typeTrajet=='Direct' :
            self.df_transit=self.trajet_direct()
            self.timedelta_min,self.timedelta_max,self.timestamp_mini,self.timestamp_maxi,self.duree_traj_fut=self.temps_timedeltas_direct()
            try : 
                self.temps_parcours_max=self.temp_max_cluster(self.df_transit,300)[1]
                self.tps_parcours_max_type='Cluster'
            except ClusterError('haha') :
                print(f"nb cluster={ClusterError.nb_cluster}, passage sur percentile") 
                self.temps_parcours_max=self.df_transit.tps_parcours.quantile(0.85)
                self.tps_parcours_max_type='85eme_percentile'
        elif typeTrajet=='Global' :
                self.df_transit, self.df_passag_transit=self.loc_trajet_global(df_filtre)
        elif typeTrajet=='Indirect' : 
            self.dico_traj_directs=self.liste_trajets_directs()
            self.df_transit=self.df_trajet_indirect()
            try : 
                self.temps_parcours_max=self.temp_max_cluster(self.df_transit,300)[1]
                self.tps_parcours_max_type='Cluster'
            except ClusterError() : 
                self.temps_parcours_max=self.df_transit.tps_parcours.quantile(0.85)
                self.tps_parcours_max_type='85eme_percentile'
        
    def trajet_direct(self):
        #trouver tt les bagnoles passée par cam1 dont la 2eme camera est cam2
        #isoler camera 1
        df_duree_cam1=self.df_duree.loc[self.df_duree.loc[:,'camera_id']==self.cameras_suivantes[0]]
        #on retrouve ces immatriculation mais qui ne sont pas à la 1ere camera
        df_duree_autres_cam=self.df.loc[(self.df.loc[:,'immat'].isin(df_duree_cam1.loc[:,'immat']))]
        #on fait une jointure entre cam 1 et les autres cam pour avoir une correspondance entre le passage devan la 1ere cmaera et la seconde
        cam1_croise_autre_cam=df_duree_cam1.reset_index().merge(df_duree_autres_cam.reset_index(), on='immat')
        #on ne garde que les passages à la 2ème caméra postérieur au passage à la première
        cam1_croise_suivant=cam1_croise_autre_cam.loc[(cam1_croise_autre_cam.loc[:,'created_x']<cam1_croise_autre_cam.loc[:,'created_y'])]
        #print(cam1_croise_suivant[['camera_id_x','created_x','created_y','camera_id_y']])
        #on isole le passage le plus rapide devant cam suivante pour chaque immatriculation
        cam1_fastest_next=cam1_croise_suivant.loc[cam1_croise_suivant.groupby(['immat'])['created_y'].idxmin()]
        #print(cam1_fastest_next[['camera_id_x','created_x','created_y','camera_id_y']])
        #Si la df cam1_fastest_next est vide ça crée une erreur ValueError dans la creation de 'l', donc je filtre avant avec une levee d'erreur PasdePl
        if cam1_fastest_next.empty : 
            raise PasDePlError()
        # on regroupe les attributs de description de type et de fiabilite de camera dans des listes (comme ça si 3 camera on pourra faire aussi)
        cam1_fastest_next['l']=cam1_fastest_next.apply(lambda x:self.test_unicite_type([x['l_x'],x['l_y']],mode='1/2'), axis=1)
        #pour la fiabilite on peut faire varier le critere. ici c'est 0 : tous le spassages sont pris
        cam1_fastest_next['fiability']=cam1_fastest_next.apply(lambda x: all(element > 0 for element in [x['fiability_x'],x['fiability_y']]), axis=1)
        #on ne garde que les passage le plus rapide devant la camera 2
        cam1_puis_cam2=cam1_fastest_next.loc[cam1_fastest_next.loc[:,'camera_id_y']==self.cameras_suivantes[1]]
        #on trie puis on ajoute un filtre surle temps entre les 2 camera.
        cam1_cam2_passages=cam1_puis_cam2.set_index('created_y').sort_index()
        cam1_cam2_passages_filtres=cam1_cam2_passages[self.date_debut:self.date_debut+pd.Timedelta(hours=self.temps_max_autorise)]
        #on ressort la colonne de tempsde l'index et on cree la colonne des differentiel de temps
        cam1_cam2_passages_filtres=cam1_cam2_passages_filtres.reset_index()
        cam1_cam2_passages_filtres['tps_parcours']=cam1_cam2_passages_filtres['created_y']-cam1_cam2_passages_filtres['created_x'] #creer la colonne des differentiel de temps
        #isoler les pl fiables
        df_pl=cam1_cam2_passages_filtres.loc[(cam1_cam2_passages_filtres.loc[:,'l']==1) & (cam1_cam2_passages_filtres.loc[:,'fiability']==True)]
        df_tps_parcours_pl_final=df_pl[['immat','created_x', 'created_y','tps_parcours']].rename(columns=dico_renommage)
        if df_tps_parcours_pl_final.empty :
            raise PasDePlError()
        
        return df_tps_parcours_pl_final
 
    def liste_trajets_directs(self):
        """
        pour obtenir un dico contenant les instances de trajet_direct pour chaque trajet éleémentaires
        """  
        dico_traj_directs={}    
        #pour chaque couple de camera
        for indice,couple_cam in enumerate([[self.cameras_suivantes[i],self.cameras_suivantes[i+1]] for i in range(len(self.cameras_suivantes)-1)]) :
            #initialisation du nom de variables pour le dico resultat 
            #print(indice,couple_cam)
            nom_variable='trajet'+str(indice)
            #calculer les temps de parcours et autres attributs issus de trajet_direct selon les resultats du precedent
            if indice==0 : # si c'est le premier tarjet on se base sur des paramètres classiques
                trajet_elem=trajet(self.df, self.date_debut, self.duree, couple_cam,temps_max_autorise=self.temps_max_autorise)
            else : 
                cle_traj_prec='trajet'+str(indice-1)
                trajet_elem=(trajet(self.df, dico_traj_directs[cle_traj_prec].timestamp_mini,
                                         dico_traj_directs[cle_traj_prec].duree_traj_fut,
                                         couple_cam, temps_max_autorise=self.temps_max_autorise))
            dico_traj_directs[nom_variable]=trajet_elem
        
        return dico_traj_directs
   
    def df_trajet_indirect(self):
        """
        On trouve les vehicules passés par les différentes cameras du dico_traj_directs
        """
        dico_rename={'date_cam_1_x':'date_cam_1','date_cam_2_y':'date_cam_2'} #paramètres pour mise en forme donnees
        
        #on fait une jointure de type 'inner, qui ne garde que les lignes présentes dans les deux tables, en iterant sur chaque element du dico
        long_dico=len(self.dico_traj_directs)
        #print (self.dico_traj_directs)
        if long_dico<2 : #si c'est le cas ça veut dire une seule entree dans le dico, ce qui signifie que l'entree est empty, donc on retourne une df empty pour etre raccord avec le type de donnees renvoyee par trajet_direct dans ce cas
            raise PasDePlError()
            return self.dico_traj_directs['trajet0'].df_transit
        for a, val_dico in enumerate(self.dico_traj_directs):
            if a<=long_dico-1:
                #print(f"occurence {a} pour trajet : {val_dico}, lg dico_:{long_dico}")
                variab,variab2 ='trajet'+str(a), 'trajet'+str(a+1)
                if self.dico_traj_directs[variab].df_transit.empty : #si un des trajets aboutit a empty, mais pas le 1er
                    return self.dico_traj_directs[variab].df_transit
                if a==0 :
                    df_transit=pd.merge(self.dico_traj_directs[variab].df_transit,self.dico_traj_directs[variab2].df_transit,on='immat')
                    if df_transit.empty :
                        raise PasDePlError()
                    df_transit['tps_parcours']=df_transit['tps_parcours_x']+df_transit['tps_parcours_y']
                    df_transit=(df_transit.rename(columns=dico_rename))[['immat','date_cam_1','date_cam_2','tps_parcours']]  
                elif a<long_dico-1: 
                    #print(f" avant boucle df_trajet_indirect, df_transit : {df_transit.columns}")
                    df_transit=pd.merge(df_transit,self.dico_traj_directs[variab2].df_transit,on='immat')
                    if df_transit.empty :
                        raise PasDePlError()
                    #print(f" apres boucle df_trajet_indirect, df_transit : {df_transit.columns}")
                    df_transit['tps_parcours']=df_transit['tps_parcours_x']+df_transit['tps_parcours_y']
                    df_transit=(df_transit.rename(columns=dico_rename))[['immat','date_cam_1','date_cam_2','tps_parcours']]
            
            #print(f"1_df_trajet_indirect, df_transit : {df_transit.columns}")
        #print(f"2_df_trajet_indirect, df_transit : {df_transit.columns}")

        df_transit['cameras']=df_transit.apply(lambda x:list(self.cameras_suivantes), axis=1)
        #print(df_transit)

        return df_transit
    
    def loc_trajet_global(self,df_filtre): 
        """
        fonction pour retrouver tous les pl d'une o_d une fois que l'on a identifé la duree_max entre 2 cameras
        permet de retrouver tous les pl apres avoir la duree du trajet indirect
        """
        def filtrer_passage(liste, df_liste_trajet,cam) :
            for liste_cams in [a for a in liste_complete_trajet.cameras.tolist() if a[0]==cam] :
                if liste[:len(liste_cams)]==tuple(liste_cams):
                    return liste[:len(liste_cams)]
            else : return liste
        
        def recuperer_date_cam2(liste,liste_created,df_liste_trajet,cam):
            for liste_cams in [a for a in liste_complete_trajet.cameras.tolist() if a[0]==cam] :
                if liste[:len(liste_cams)]==tuple(liste_cams):
                    return liste_created[len(liste_cams)-1]
            else : return liste_created[-1]
        
        liste_trajet_od=self.recup_trajets()[0]
        camera1=self.cameras_suivantes[0]
        #on limite le nb d'objet entre les 2 heures de depart
        df_duree=self.df.loc[self.date_debut:self.date_fin]
        if df_duree.empty : 
           raise PasDePlError() 
        if isinstance(df_filtre,pd.DataFrame) : 
            df_duree=filtrer_df(df_duree,df_filtre)
        #on trouve les veh passés cameras 1
        df_duree_cam1=df_duree.loc[df_duree.loc[:,'camera_id']==camera1]
        if df_duree_cam1.empty : 
           raise PasDePlError() 
        #on recupere ces immat aux autres cameras
        df_duree_autres_cam=self.df.loc[(self.df.loc[:,'immat'].isin(df_duree_cam1.loc[:,'immat']))]
        groupe=(df_duree_autres_cam.sort_index().reset_index().groupby('immat').agg({'camera_id':lambda x : tuple(x), 'l': lambda x : self.test_unicite_type(list(x),'1/2'),
                                                                                       'created':lambda x: tuple(x)}))
        groupe_pl=groupe.loc[groupe['l']==1].copy() #on ne garde que les pl
        if groupe_pl.empty :
            raise PasDePlError()
        groupe_pl['camera_id']=groupe_pl.apply(lambda x : filtrer_passage(x['camera_id'],liste_complete_trajet,camera1),axis=1)#on filtre les cameras selon la liste des trajets existants
        groupe_pl['created']=groupe_pl.apply(lambda x : recuperer_date_cam2(x['camera_id'],x['created'],liste_complete_trajet,camera1),axis=1)#on recupère les datetimede passages correspondants
        df_ts_trajets=(groupe_pl.reset_index().merge(liste_complete_trajet[['cameras','origine','destination']],right_on='cameras', left_on='camera_id').
                       rename(columns={'created':'date_cam_2'}).drop('camera_id',axis=1))#on récupère les infos par jointure sur les cameras
        if df_ts_trajets.empty :
            raise PasDePlError()
        df_ts_trajets['o_d']=df_ts_trajets.apply(lambda x : x['origine']+'-'+x['destination'],axis=1)
        df_agrege=df_duree_cam1.reset_index().merge(df_ts_trajets,on='immat').drop(['camera_id', 'l_x','fiability'],axis=1).rename(columns={'l_y':'l','created':'date_cam_1'})
        df_agrege['tps_parcours']=df_agrege.apply(lambda x : x.date_cam_2-x.date_cam_1, axis=1)
        df_agrege=df_agrege.loc[df_agrege['date_cam_2'] > df_agrege['date_cam_1']]#pour les cas bizarres de plaques vu a qq minutes d'intervalle au sein d'une même heure
        
        if df_agrege.empty :
            raise PasDePlError()
        
        #pour obtenir la liste des passagesrelevant de trajets de transits :
        #limitation des données des cameras par jointures
        df_joint_passag_transit=df_agrege.merge(df_duree_autres_cam.reset_index(), on='immat')
        df_passag_transit1=df_joint_passag_transit.loc[(df_joint_passag_transit.apply(lambda x : x['camera_id'] in x['cameras'], axis=1))]
        df_passag_transit=(df_passag_transit1.loc[df_passag_transit1.apply(lambda x : x['date_cam_1']<=x['created']<=x['date_cam_2'], axis=1)]
                        [['created','camera_id','immat','fiability','l_y','state_x']].rename(columns={'l_y':'l','state_x':'state'}))
        
        
        return df_agrege,df_passag_transit
        
    def temps_timedeltas_direct(self):
        
        timedelta_min=self.df_transit.tps_parcours.min()
        timedelta_max=self.df_transit.tps_parcours.max()
        timestamp_mini=self.date_debut+timedelta_min
        timestamp_maxi=self.date_fin+timedelta_max
        duree_traj_fut=(((timestamp_maxi-timestamp_mini).seconds)//60)+1
        
        return timedelta_min,timedelta_max,timestamp_mini,timestamp_maxi,duree_traj_fut
        
    def temp_max_cluster(self, df_pl_ok, delai):
        """obtenir le temps max de parcours en faisant un cluster par dbscan
        en entree : la df des temps de parcours pl final
                    le delai max pour regrouper en luster,en seconde
        en sortie : le nombre de clusters,
                    un timedelta
        """
        if df_pl_ok.empty:
            raise ClusterError()
        donnees_src=df_pl_ok.loc[:,['date_cam_1','tps_parcours']].copy() #isoler les données necessaires
        temps_int=((pd.to_datetime('2018-01-01')+donnees_src['tps_parcours'])-pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')#convertir les temps en integer
        #mise en forme des données pour passer dans sklearn 
        donnnes = temps_int.values
        matrice=donnnes.reshape(-1, 1)
        #faire tourner la clusterisation et recupérer le label (i.e l'identifiant cluster) et le nombre de cluster
        try :
            clustering=DBSCAN(eps=delai, min_samples=len(temps_int)/2).fit(matrice)
        except ValueError :
            raise ClusterError('ValueError lors de la creation du Cluster')
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # A AMELIORER EN CREANT UNE ERREUR PERSONALISEE SI ON OBTIENT  CLUSTER
        if n_clusters_== 0 :
            raise ClusterError('Nb_Cluster = 0')
        #mettre en forme au format pandas
        results = pd.DataFrame(pd.DataFrame([donnees_src.index,labels]).T)
        results.columns = ['index_base', 'cluster_num']
        results = pd.merge(results,df_pl_ok, left_on='index_base', right_index=True )
        #obtenir un timedelta unique
        temp_parcours_max=results.loc[results.loc[:,'cluster_num']!=-1].groupby(['cluster_num'])['tps_parcours'].max()
        temp_parcours_max=pd.to_timedelta(temp_parcours_max.values[0])
        
        return n_clusters_, temp_parcours_max
    
    def test_unicite_type(self,liste_l, mode='unique'):
        """test pour voir si un vehicule a ete toujours vu de la mme façon ou non
           en entre : liste de valeur de l iisues d'une df""" 
        if mode=='unique' : 
            if len(set(liste_l))==1 :
                return liste_l[0]
            else : 
                return -1
        elif mode=='1/2' :
            if any(liste_l)==1 : 
                return 1
            else : 
                return -1

    def recup_trajets(self):
        liste_trajet_od=([tuple(cam['cameras']) for camera in fichier_trajet.loc[(fichier_trajet['cam_d']==self.cameras_suivantes[1]) & 
                                                                              (fichier_trajet['cam_o']==self.cameras_suivantes[0])].trajets for cam in camera])
        trajet_max=[]
        for trajet in liste_trajet_od : 
            if len(trajet)==max([len(trajet) for trajet in liste_trajet_od]) : 
                trajet_max.append(trajet)
        
        return liste_trajet_od, trajet_max
    
    def graph(self):
        copie_df=self.df_transit.copy()
        copie_df.tps_parcours=pd.to_datetime('2018-01-01')+copie_df.tps_parcours
        copie_df['temps_parcours_max']=pd.to_datetime('2018-01-01')+self.temps_parcours_max
        graph_tps_parcours = alt.Chart(copie_df).mark_point().encode(
                        x='date_cam_1',
                        y='hoursminutes(tps_parcours)',
                        tooltip='hoursminutes(tps_parcours)').interactive()
        graph_tps_filtre=alt.Chart(copie_df).mark_line(color='yellow').encode(
                         x='date_cam_1',
                         y='hoursminutes(temps_parcours_max)',
                         tooltip='hoursminutes(temps_parcours_max)')
        graph_tps_parcours=graph_tps_parcours+graph_tps_filtre
        
        return graph_tps_parcours

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class ClusterError(Exception):
    """Excpetion si pb dans la construction du cluster
    Attributs : 
        message -- message d'erreur -- text
        nb_cluster -- nombre de cluster -- int 
    """       
    def __init__(self,message,nb_cluster=None ):
        self.message=message 
        self.nb_cluster=nb_cluster
        

class PasDePlError(Error):  
    """
    Exception levee si le trajet direct ne comprend pas de pl
    """     
    def __init__(self):
        print('pas de PL sur la période et les cameras visées')

class TypeTrajet_NbCamera_Error(Error):  
    """
    Exception levee si le type de trajet ne correpond pas au nombre de camera ou si le type e trajet n'est pas Direct, Indirect, Global.
    """     
    def __init__(self, nb_cam, typeTrajet):
        self.typeTrajet=typeTrajet
        self.nb_cam=nb_cam
        print(f"le nb de camera :{self.nb_cam} ne correspond pas au type de trajet : {typeTrajet}, ou le type n'est pas connu") 

    
def transit_1_jour(df_journee,date_jour, liste_trajets, save_graphs=False):
    """Fonction d'agregation des trajets de transit sur une journee
    en entre : 
        date_jour -> str :date de la journee analysee 'YYYY-MM-DD'
        liste_trajets -> DataFrame lue depuis le fichier adquat, cf variable liste_trajet du module
        save_graph -> booleen, par defaut False, pour savoir si on exporte des graphs lies au trajets directs (10* temps sans graph)
    en sortie : DataFrame des trajets de transit
    """
    #parcourir les dates
    dico_trajet_od,dico_passag_od={},{} #dico avec cle par o_d
    for date in pd.date_range(date_jour, periods=2, freq='H') : 
        print(f"date : {date} debut_traitement : {dt.datetime.now()}")
        #parcourir les trajets possibles
        for index, value in liste_trajets.iterrows() :
            origine,destination,cameras=value[0],value[1],[value[2],value[3]]
            o_d=origine+'-'+destination
            #print(f"index : {index},trajet : {origine}-{destination}, date : {date}, debut_traitement : {dt.datetime.now()}")
            try : 
                if 'dico_passag' in locals() : #si la varible existe deja on utilise pour filtrer le df_journee en enlevant les passages dejà pris dans une o_d (sinon double compte ente A63 - A10 et A660 -A10 par exemple 
                    donnees_trajet=trajet(df_journee,date,60,16,cameras, type='Global',df_filtre=dico_passag)
                else : 
                    donnees_trajet=trajet(df_journee,date,60,16,cameras, type='Global')
                df_trajet, df_passag=donnees_trajet.df_transit, donnees_trajet.df_passag_transit
            except PasDePlError :
                continue
            
            df_trajet['o_d'],df_trajet['origine'],df_trajet['destination']=o_d, origine, destination
            if o_d in dico_trajet_od.keys():
                dico_trajet_od[o_d]=pd.concat([dico_trajet_od[o_d],df_trajet])
                dico_passag_od[o_d]=pd.concat([dico_passag_od[o_d],df_passag])
            else : 
                dico_trajet_od[o_d]=df_trajet
                dico_passag_od[o_d]=df_passag
            #dico_passag_od[o_d]=df_passag

            if 'dico_od' in locals() : #si la varible existe deja on la concatene avec le reste
                dico_od=pd.concat([dico_od,df_trajet], sort=False)
            else : #sinon on initilise cette variable
                dico_od=df_trajet  
            if 'dico_passag' in locals() : #si la varible existe deja on la concatene avec le reste
                dico_passag=pd.concat([dico_passag,df_passag], sort=False)
            else : #sinon on initilise cette variable
                dico_passag=df_passag
 
                
    return dico_trajet_od, dico_passag_od, dico_od,  dico_passag         
                

def transit_temps_complet_v2(date_debut, nb_jours,liste_trajets, df_3semaines):
    #utiliser ouvrir_fichier_lapi pour ouvrir un df sur 3 semaine
    date_fin=(pd.to_datetime(date_debut)+pd.Timedelta(days=nb_jours)).strftime('%Y-%m-%d')
    dico_trajet_od,dico_passag_od={},{} 
    #df_3semaines=ouvrir_fichier_lapi(date_debut,date_fin).set_index('created').sort_index()
    #selection de 1 jour par boucle
    for date in pd.date_range(date_debut, periods=nb_jours*24, freq='H') :
        if date.weekday()==5 : # si on est le semadi on laisse la journee de dimanche passer et le pl repart
            df_journee=df_3semaines.loc[date:date+pd.Timedelta(hours=32)]
        else : 
            df_journee=df_3semaines.loc[date:date+pd.Timedelta(hours=18)]
        print(f"date : {date} debut_traitement : {dt.datetime.now()}")
        for cameras in zip([15,12,8,10,19,6],range(6)) : #dans ce mode peu importe la camera d'arrivée, elle sont toutes analysées
            #print(f"cameras{cameras}, date : {date}, debut_traitement : {dt.datetime.now()}")
            try : 
                if 'dico_passag' in locals() : #si la varible existe deja on utilise pour filtrer le df_journee en enlevant les passages dejà pris dans une o_d (sinon double compte ente A63 - A10 et A660 -A10 par exemple 
                    #print(dico_passag.loc[dico_passag['created']>=date])
                    donnees_trajet=trajet(df_journee,date,60,cameras, type='Global',df_filtre=dico_passag.loc[dico_passag['created']>=date].copy())
                else : 
                    donnees_trajet=trajet(df_journee,date,60,cameras, type='Global')
                df_trajet, df_passag=donnees_trajet.df_transit, donnees_trajet.df_passag_transit
            except PasDePlError :
                continue

            if 'dico_od' in locals() : #si la varible existe deja on la concatene avec le reste
                dico_od=pd.concat([dico_od,df_trajet], sort=False)
            else : #sinon on initilise cette variable
                dico_od=df_trajet  
            if 'dico_passag' in locals() : #si la varible existe deja on la concatene avec le reste
                dico_passag=pd.concat([dico_passag,df_passag], sort=False)
            else : #sinon on initilise cette variable
                dico_passag=df_passag
            
            #df_journee=filtrer_df(df_journee, df_passag)
    return dico_od,  dico_passag

def pourcentage_pl_camera(date_debut, nb_jours,df_3semaines,dico_passag):
    #isoler les pl de la source
    df_pl_3semaines=df_3semaines.loc[df_3semaines['l']==1]
    df_pl_3semaines.set_index('created',inplace=True)
    #comparer les pl en transit avec l'ensembles des pl, par heure, par camera
    #obtenir les nb de pl par heure et par camera sur la source
    df_synthese_pl_tot=df_pl_3semaines.groupby('camera_id').resample('H').count()['immat'].rename(column={'immat':'nb_pl_tot'})
    df_synthese_pl_transit=dico_passag.set_index('created').groupby('camera_id').resample('H').count()['immat'].rename(column={'immat':'nb_pl_transit'})
    df_pct_pl_transit=pd.concat([df_synthese_pl_tot,df_synthese_pl_transit], axis=1, join='inner')
    df_pct_pl_transit.columns=[['nb_pl_tot','nb_pl_transit']]
    df_pct_pl_transit['pct_pl_transit']=df_pct_pl_transit.apply(lambda x : float(x['nb_pl_transit'])*100 / x['nb_pl_tot'] ,axis=1)
    
    return df_pct_pl_transit
    
def filtrer_df(df_global,df_filtre): 
    df_global=df_global.reset_index().set_index(['created','immat'])
    df_filtre=df_filtre.reset_index().set_index(['created','immat'])
    df_global_filtre=df_global.loc[~df_global.index.isin(df_filtre.index)].reset_index().set_index('created')
    return df_global_filtre

def temp_max_cluster(self, df_pl_ok, delai):
        """obtenir le temps max de parcours en faisant un cluster par dbscan
        en entree : la df des temps de parcours pl final
                    le delai max pour regrouper en luster,en seconde
        en sortie : le nombre de clusters,
                    un timedelta
        """
        donnees_src=df_pl_ok.loc[:,['created_x','tps_parcours']].copy() #isoler les données necessaires
        temps_int=((pd.to_datetime('2018-01-01')+donnees_src['tps_parcours'])-pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')#convertir les temps en integer
        #mise en forme des données pour passer dans sklearn 
        donnnes = temps_int.values
        matrice=donnnes.reshape(-1, 1)
        #faire tourner la clusterisation et recupérer le label (i.e l'identifiant cluster) et le nombre de cluster
        try :
            clustering=DBSCAN(eps=delai, min_samples=len(temps_int)/2).fit(matrice)
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
        temp_parcours_max=results.loc[results.loc[:,'cluster_num']!=-1].groupby(['cluster_num'])['tps_parcours'].max()
        temp_parcours_max=pd.to_timedelta(temp_parcours_max.values[0])
        
        return n_clusters_, temp_parcours_max
    
    
    
       
    
    