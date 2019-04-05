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
fichier_trajet=(pd.DataFrame([{'origine':'A63','destination':'A10','cam_o':14, 'cam_d':11, 'trajets':[
                                                        {'cameras':[14,19,4,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[14,19,1,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[14,4,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[14,1,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[14,19,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[14,19,4,11],'type_trajet':'indirect'},
                                                        {'cameras':[14,19,1,11],'type_trajet':'indirect'},
                                                        {'cameras':[14,4,11],'type_trajet':'indirect'},
                                                        {'cameras':[14,1,11],'type_trajet':'indirect'},
                                                        {'cameras':[14,5,11],'type_trajet':'indirect'},
                                                        {'cameras':[14,19,11],'type_trajet':'indirect'},
                                                        {'cameras':[14,11],'type_trajet':'direct'}
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
                            {'origine':'A63','destination':'N10','cam_o':14, 'cam_d':5,'trajets':[{'cameras':[14,19,4,5],'type_trajet':'indirect'},
                                                        {'cameras':[14,19,1,5],'type_trajet':'indirect'},
                                                        {'cameras':[14,4,5],'type_trajet':'indirect'},
                                                        {'cameras':[14,1,5],'type_trajet':'indirect'},
                                                        {'cameras':[14,5],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'N10','destination':'A63','cam_o':6, 'cam_d':13,'trajets':[{'cameras':[6,2,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[6,3,18,13],'type_trajet':'indirect'},
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
                            {'origine':'A63','destination':'A62','cam_o':14, 'cam_d':9,'trajets':[{'cameras':[14,19,9],'type_trajet':'indirect'},
                                                        {'cameras':[14,9],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'A62','destination':'A63','cam_o':10, 'cam_d':13,'trajets':[{'cameras':[10,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[10,13],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'A89','destination':'A63','cam_o':8 ,'cam_d':13,'trajets':[{'cameras':[8,3,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[8,18,13],'type_trajet':'indirect'},
                                                        {'cameras':[8,3,13],'type_trajet':'indirect'},
                                                        {'cameras':[8,13],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'A63','destination':'A89','cam_o':14, 'cam_d':7,'trajets':[{'cameras':[14,19,4,7],'type_trajet':'indirect'},
                                                        {'cameras':[14,4,7],'type_trajet':'indirect'},
                                                        {'cameras':[14,19,7],'type_trajet':'indirect'},
                                                        {'cameras':[14,7],'type_trajet':'direct'},
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
                            {'origine':'A10','destination':'A630','cam_o':12, 'cam_d':18,'trajets':[{'cameras':[12,6,2,18],'type_trajet':'indirect'},
                                                         {'cameras':[12,6,3,18],'type_trajet':'indirect'},
                                                         {'cameras':[12,2,18],'type_trajet':'indirect'},
                                                         {'cameras':[12,3,18],'type_trajet':'indirect'},
                                                         {'cameras':[12,6,18],'type_trajet':'indirect'},
                                                         {'cameras':[12,18],'type_trajet':'direct'},
                                                        ]},
                            {'origine':'A630','destination':'A10','cam_o':19, 'cam_d':18,'trajets':[{'cameras':[19,4,5,11],'type_trajet':'indirect'},
                                                         {'cameras':[19,1,5,11],'type_trajet':'indirect'},
                                                         {'cameras':[19,1,11],'type_trajet':'indirect'},
                                                         {'cameras':[19,4,11],'type_trajet':'indirect'},
                                                         {'cameras':[19,5,11],'type_trajet':'indirect'},
                                                         {'cameras':[19,11],'type_trajet':'direct'},
                                                       ]},
                            {'origine':'A630','destination':'A62','cam_o':19, 'cam_d':9,'trajets':[{'cameras':[19,9],'type_trajet':'direct'}
                                                        ]},
                            {'origine':'A62','destination':'A630','cam_o':10, 'cam_d':18,'trajets':[{'cameras':[10,18],'type_trajet':'direct'}
                                                        ]},
                            {'origine':'A630','destination':'A89','cam_o':19, 'cam_d':7,'trajets':[{'cameras':[19,4,7],'type_trajet':'indirect'},
                                                         {'cameras':[19,7],'type_trajet':'direct'}
                                                        ]},
                            {'origine':'A89','destination':'A630','cam_o':8, 'cam_d':18,'trajets':[{'cameras':[8,3,18],'type_trajet':'indirect'},
                                                         {'cameras':[8,18],'type_trajet':'direct'}
                                                        ]},
                            {'origine':'N10','destination':'A630','cam_o':6, 'cam_d':18,'trajets':[{'cameras':[6,2,18],'type_trajet':'indirect'},
                                                         {'cameras':[6,3,18],'type_trajet':'indirect'},
                                                         {'cameras':[6,18],'type_trajet':'direct'},
                                                        ]},
                            {'origine':'A630','destination':'N10','cam_o':19, 'cam_d':5,'trajets':[{'cameras':[19,1,5],'type_trajet':'indirect'},
                                                         {'cameras':[19,4,5],'type_trajet':'indirect'},
                                                         {'cameras':[19,5],'type_trajet':'direct'},
                                                        ]}
                           ]))[['origine', 'destination', 'cam_o', 'cam_d','trajets']]

def ouvrir_fichier_lapi(date_debut, date_fin) : 
    with ct.ConnexionBdd('gti_lapi') as c : 
        requete=f"select case when camera_id=13 or camera_id=15 then 13 when camera_id=14 or camera_id=16 then 14 else camera_id end::integer as camera_id , created, immat, fiability, l, state from data.te_passage where created between '{date_debut}' and '{date_fin}'"
        df=pd.read_sql_query(requete, c.sqlAlchemyConn)
        return df

class df_source():
    """
    classe regroupant les principales caractéristiques d'une df de lapi issues de la fonction df_temps_parcours_moyen
    Dans cette classe une ligne est un trajet (opposé à la classe de base ou un ligne estun vehicule)
    """
    def __init__(self,df):
        """
        constrcuteur, va creer attributs : 
        - nb de vehicules total, nb de vehicules sur, nb vl sur, nb pl sur, nb_plaqu_ok, nb_type_identifie
        - un df pour chaque attribut avec 
        """
        self.df=df
        self.df_vlpl, self.df_tv_plaques_ok, self.df_veh_ok, self.df_vl_ok, self.df_pl_ok=self.df_filtrees()
        self.nb_tv_tot, self.nb_tv_plaque_ok, self.nb_vlpl, self.nb_veh_ok, self.nb_vl_ok, self.nb_pl_ok=self.stats()             
        
    def df_filtrees(self):
        """isole les df depuis la df source
        A RENESEIGNER LES DF QUI SORTENT"""
        #Dataframes intermediaires 
        df_vlpl=self.df.loc[self.df.loc[:,'l']!=-1]
        df_tv_plaques_ok=self.df.loc[self.df.loc[:,'fiability']>=80]
        df_veh_ok=self.df.loc[(self.df.loc[:,'l']!=-1) & (self.df.loc[:,'fiability']>=80)]
        df_vl_ok=self.df.loc[(self.df.loc[:,'l']==0) & (self.df.loc[:,'fiability']>=80)]
        df_pl_ok=self.df.loc[(self.df.loc[:,'l']==1) & (self.df.loc[:,'fiability']>=80)]
        return df_vlpl, df_tv_plaques_ok, df_veh_ok, df_vl_ok, df_pl_ok 
            def stats (self):
        """les stats issue des df_filtrees
        A RENESEIGNER LES DF QUI SORTENT"""
        nb_tv_tot=len(self.df)
        nb_tv_plaque_ok=len(self.df_tv_plaques_ok)
        nb_vlpl=len(self.df_vlpl)
        nb_veh_ok=len(self.df_veh_ok)                      
        nb_vl_ok=len(self.df_vl_ok)
        nb_pl_ok=len(self.df_pl_ok)
        return nb_tv_tot, nb_tv_plaque_ok, nb_vlpl, nb_veh_ok, nb_vl_ok, nb_pl_ok
            def plot_graphs(self):
        """cree et retourne les charts de altair pour un certines nombres de graph"""
        
        #graph des stats de df
        stats_df=pd.DataFrame([{'type': 'nb_tv_tot', 'value':self.nb_tv_tot},{'type': 'nb_tv_plaque_ok', 'value':self.nb_tv_plaque_ok},
                               {'type': 'nb_vlpl', 'value':self.nb_vlpl},{'type': 'nb_veh_ok', 'value':self.nb_veh_ok},
                               {'type': 'nb_vl_ok', 'value':self.nb_vl_ok},{'type': 'nb_pl_ok', 'value':self.nb_pl_ok}])
        graph_stat=alt.Chart(stats_df).mark_bar().encode(
            x='type',
            y='value',
            color='type')
        graph_stat_trie = graph_stat.encode(alt.X(field='type', type='nominal',
                                            sort=alt.EncodingSortField(field='value',op='mean'))).properties(width=100)
        
        #graph de la fiabilte de la plaque dans le temps
        stat_fiability = self.df.loc[:,['created', 'fiability']].copy().set_index('created').sort_index()
        stat_fiability=stat_fiability.groupby(pd.Grouper(freq='5Min')).mean().reset_index()
        graph_fiab=alt.Chart(stat_fiability).mark_line().encode(
            alt.X('created'),
            alt.Y('fiability',scale=alt.Scale(zero=False))).interactive()
        
        return  graph_stat_trie, graph_fiab

class trajet():
    """
    classe regroupant permettant le calcul de trajet direct, indirect ou global
    en entre : une df issue de ouvrir_fichier_lapi
    """
    
    def __init__(self,df,date_debut, duree, temps_max_autorise, cameras,type='Direct') :
        
        #en fonction du df qui est passé on met la date de creation en index ou non
        if isinstance(df.index,pd.DatetimeIndex) :
            self.df=df
        else :
            self.df=df.set_index('created').sort_index()
        
        self.date_debut, self.duree, self.temps_max_autorise, self.cameras_suivantes=pd.to_datetime(date_debut), duree, temps_max_autorise,cameras
        self.date_fin=self.date_debut+pd.Timedelta(minutes=self.duree)
        self.df_duree=self.df.loc[self.date_debut:self.date_fin]  
    
        if len(cameras)==2:
            if type=='Direct' :
                self.df_pl_direct=self.trajet_direct()
                self.timedelta_min,self.timedelta_max,self.timestamp_mini,self.timestamp_maxi,self.duree_traj_fut=self.temps_timedeltas_direct()
            elif type=='Global' :
                #trouver trajet complet
                self.trajet_complet=self.recup_trajets()[1]
                #obtenir le temps de parcours max en minutes
                tps_max=[]
                for trajet_indirect in self.trajet_complet :
                    tps=trajet(df,date_debut, duree, temps_max_autorise, trajet_indirect).temps_parcours_max
                    tps_max.append(tps)
                    tps=1 #pour envoi de l'objet tarjet au GB
                tps_max=np.max(tps_max)
                #rechercher le df des passages avec 
                self.df_global, self.df_passag_transit=self.loc_trajet_global(tps_max)
        else : 
            self.dico_traj_directs=self.liste_trajets_directs()
            self.df_transit=self.df_trajet_indirect()
            self.temps_parcours_max=self.df_transit.tps_parcours.max()
        
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
        # on regroupe les attributs dedescription de type etde fiabilite de camera dans des listes (comme ça si 3 camera on pourra faire aussi
        cam1_fastest_next['l']=cam1_fastest_next.apply(lambda x:self.test_unicite_type([x['l_x'],x['l_y']],mode='1/2'), axis=1)
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
        try : 
            self.temps_parcours_max=self.temp_max_cluster(df_pl,300)[1]
        except ClusterError : 
            self.temps_parcours_max=df_pl.tps_parcours.quantile(0.85)
        df_tps_parcours_pl_final=(df_pl.loc[df_pl['tps_parcours']<self.temps_parcours_max]
                                        [['immat','created_x', 'created_y','tps_parcours']].rename(columns=dico_renommage))
        if df_tps_parcours_pl_final.empty :
            raise PasDePlError()
        
        return df_tps_parcours_pl_final
    
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
            return self.dico_traj_directs['trajet0'].df_pl_direct
        for a, val_dico in enumerate(self.dico_traj_directs):
            if a<=long_dico-1:
                #print(f"occurence {a} pour trajet : {val_dico}, lg dico_:{long_dico}")
                variab,variab2 ='trajet'+str(a), 'trajet'+str(a+1)
                if self.dico_traj_directs[variab].df_pl_direct.empty : #si un des trajets aboutit a empty, mais pas le 1er
                    return self.dico_traj_directs[variab].df_pl_direct
                if a==0 :
                    df_transit=pd.merge(self.dico_traj_directs[variab].df_pl_direct,self.dico_traj_directs[variab2].df_pl_direct,on='immat')
                    if df_transit.empty :
                        raise PasDePlError()
                    df_transit['tps_parcours']=df_transit['tps_parcours_x']+df_transit['tps_parcours_y']
                    df_transit=(df_transit.rename(columns=dico_rename))[['immat','date_cam_1','date_cam_2','tps_parcours']]  
                elif a<long_dico-1: 
                    #print(f" avant boucle df_trajet_indirect, df_transit : {df_transit.columns}")
                    df_transit=pd.merge(df_transit,self.dico_traj_directs[variab2].df_pl_direct,on='immat')
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
    
    def loc_trajet_global(self, duree_max): 
        """
        fonction pour retrouver tous les pl d'une o_d une fois que l'on a identifé la duree_max entre 2 cameras
        permet de retrouver tous les pl apres avoir la duree du trajet indirect
        """
        liste_trajet_od=self.recup_trajets()[0]
        camera1, camera2=self.cameras_suivantes[0], self.cameras_suivantes[1]
        #on limite le nb d'objet entre les 2 heures de depart
        df_duree=self.df.loc[self.date_debut:self.date_fin]
        #on trouve les veh passés cameras 1
        df_duree_cam1=df_duree.loc[df_duree.loc[:,'camera_id']==self.cameras_suivantes[0]]
        #on les retrouve aux autres cameras
        df_duree_autres_cam=self.df.loc[(self.df.loc[:,'immat'].isin(df_duree_cam1.loc[:,'immat']))]
        #on limite ces données selon le temps autorisé à partir de la date de depart
        df_autres_cam_temp_ok=df_duree_autres_cam.loc[self.date_debut:self.date_fin+duree_max]
        #on trie par heure de passage devant les cameras puis on regroupe et on liste les cameras devant lesquelles ils sont passés
        groupe=(df_autres_cam_temp_ok.sort_index().reset_index().groupby('immat').agg({'camera_id':lambda x : tuple(x), 'l': lambda x : self.test_unicite_type(list(x),'1/2'),
                                                                        'created':lambda x: x.max()}))
        #jointure avec la df de départ pour récupérer le passage devant la camera 1
        df_agrege=df_duree_cam1.join(groupe,on='immat',lsuffix='_left')[['immat', 'l','camera_id','created']].rename(columns={'created':'date_cam_2', 'camera_id':'cameras'})
        #temps de parcours
        df_agrege=df_agrege.reset_index().rename(columns={'created':'date_cam_1'})
        df_agrege['tps_parcours']=df_agrege.apply(lambda x : x.date_cam_2-x.date_cam_1, axis=1)
        #on ne garde que les vehicules passé à la camera 2 et qui sont des pl et qui ont un tpsde parcours < au temps pre-calcule
        df_trajet=(df_agrege.loc[(df_agrege['cameras'].apply(lambda x : x[-1])==self.cameras_suivantes[1]) & (df_agrege['l']==1)
                                  & (df_agrege['tps_parcours'] < duree_max)])
        # on filtre les les cameras si ils ne sont pas dans les patterns prévus dans liste_trajet_total
        df_trajet_final=df_trajet.loc[df_trajet['cameras'].isin(liste_trajet_od)]
        if df_trajet_final.empty :
            raise PasDePlError()
        
        #pour obtenir la liste des passagesrelevant de trajets de transits :
        #limitation des données des cameras par jointures
        df_joint_passag_transit=df_trajet_final.merge(df_duree_autres_cam.reset_index(), on='immat')
        #print(f'nb_ob_trajet_final : {len(df_trajet_final)}, \n  dataframe : passage : {df_joint_passag_transit}')
        #filtrer selon les cameras présentent dans la liste et created est compris entre date_cam_1 et date_cam_2
        df_passag_transit1=df_joint_passag_transit.loc[(df_joint_passag_transit.apply(lambda x : x['camera_id'] in x['cameras'], axis=1))]
        df_passag_transit=(df_passag_transit1.loc[df_passag_transit1.apply(lambda x : x['date_cam_1']<=x['created']<=x['date_cam_2'], axis=1)]
                [['created','camera_id','immat','fiability','l_y','state']].rename(columns={'l_y':'l'}))
        
        return df_trajet_final, df_passag_transit
    
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
                trajet_elem=trajet(self.df, self.date_debut, self.duree, self.temps_max_autorise, couple_cam)
            else : 
                cle_traj_prec='trajet'+str(indice-1)
                trajet_elem=(trajet(self.df, dico_traj_directs[cle_traj_prec].timestamp_mini,
                                         dico_traj_directs[cle_traj_prec].duree_traj_fut,self.temps_max_autorise,
                                         couple_cam))
            dico_traj_directs[nom_variable]=trajet_elem
        
        return dico_traj_directs
    
    def temps_timedeltas_direct(self):
        
        timedelta_min=self.df_pl_direct.tps_parcours.min()
        timedelta_max=self.df_pl_direct.tps_parcours.max()
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
    

class trajet_direct():
    """
    Classe decrivant les df des temps de parcours 
    """
    
    def __init__(self,df, date_debut, duree, temps_max_autorise, camera1, camera2, avecGraph=False):
        """
        constrcuteur
        en entree : dataframe : le dataframe format pandas qui contient les données
                date_debut : string ou pd.timestamps decrivant une date avec Y-M-D HH:MM:SS  : date de part d'analyse du temp moyen           
                duree : integer : duree en minute : c'est le temps entre lequel on va regarder le temps mpyen : de 7h à 7h et 30 min par exemple
                temps_max_autorise : integer : le nb heure max entre la camerade debut et de fin
                camera1 : integer : camera de debut
                camera2 : integer : camera de fin
                avecGraph -> booleen traduit si on veut creer les attributs de graph ou non
        """
        self.df, self.duree, self.temps_max_autorise, self.camera1, self.camera2=df, duree, temps_max_autorise, camera1, camera2
        
        #j'en fais un attribut car util edans la fonction recherche_trajet_indirect
        self.date_debut=pd.to_datetime(date_debut)
        self.date_fin=self.date_debut+pd.Timedelta(minutes=self.duree) 
        
        #base pour resultats
        self.df_tps_parcours_brut=self.df_temps_parcours_bruts()
        
        
        #resultats intermediaires
        self.df_vlpl, self.df_tv_plaques_ok, self.df_veh_ok, self.df_vl_ok,self.df_pl_ok=self.df_filtrees(self.df_tps_parcours_brut)
        self.nb_tv_tot, self.nb_tv_plaque_ok, self.nb_vlpl, self.nb_veh_ok, self.nb_vl_ok, self.nb_pl_ok=self.stats(self.df_tps_parcours_brut) 
        
        #si pas de pl sur la periode et les cameras on remonte l'info
        if self.df_pl_ok.empty :
            raise PasDePlError()

        #filtre statistique : caracterisation des temps de parcours qui servent de filtre : test sur les percentuil et sur un cluster
        self.tps_pl_90_qtl=self.df_pl_ok.tps_parcours.quantile(0.9)  
        self.tps_pl_85_qtl=self.df_pl_ok.tps_parcours.quantile(0.85) 
        try : 
            self.temps_pour_filtre=self.temp_max_cluster(300)[1]
            if avecGraph : 
                self.graph_stat_trie, self.graph_tps_bruts, self.graph_prctl=self.plot_graphs()
            
        except ClusterError : 
            self.temps_pour_filtre=self.tps_pl_85_qtl=self.df_pl_ok.tps_parcours.quantile(0.85)
            #print(f"pas de cluster pour trajet {self.camera1, self.camera2} entre  {self.date_debut,self.date_fin} ")
            if avecGraph :
                self.graph_stat_trie, self.graph_tps_bruts, self.graph_prctl=self.plot_graphs(False)
        
        #resultats finaux finaux : un df des vehicules passe par les 2 cameras dans l'ordre, qui sont ok en plque et en typede véhicules
        self.df_tps_parcours_vl_final=self.df_vl_ok[['immat','created_x', 'created_y','tps_parcours']].rename(columns=dico_renommage)
        self.df_tps_parcours_pl_final=(self.df_pl_ok.loc[self.df_pl_ok['tps_parcours']<self.temps_pour_filtre]
                                        [['immat','created_x', 'created_y','tps_parcours']].rename(columns=dico_renommage))
        if self.df_tps_parcours_pl_final.empty :
            raise PasDePlError()
        self.df_tps_parcours_pl_final['cameras']=self.df_tps_parcours_pl_final.apply(lambda x:list([self.camera1,self.camera2]), axis=1)
        
        #resultats finaux : temps de parcours min et max et autres indicatuers utiles si ce trajet direct est partie d'un trajet indirect
        self.timedelta_min=self.df_tps_parcours_pl_final.tps_parcours.min()
        self.timedelta_max=self.df_tps_parcours_pl_final.tps_parcours.max()
        self.timestamp_mini=self.date_debut+self.timedelta_min
        self.timestamp_maxi=self.date_fin+self.timedelta_max
        self.duree_traj_fut=(((self.timestamp_maxi-self.timestamp_mini).seconds)//60)+1
        
        #print(f" trajet direct : {self.camera1, self.camera2} entre  {self.date_debut,self.date_fin}, nb_pl_ok : {self.nb_pl_ok}")
        #print('dataframe : ', self.df_tps_parcours_pl_final)
               
    
    def df_filtrees(self,df):
        """isole les df depuis la df source
        A RENESEIGNER LES DF QUI SORTENT"""
        #Dataframes intermediaires 
        df_vlpl=df.loc[df.loc[:,'l']!=-1]
        df_tv_plaques_ok=df.loc[df.loc[:,'fiability']==True]
        df_veh_ok=df.loc[(df.loc[:,'l']!=-1) & (df.loc[:,'fiability']==True)]
        df_vl_ok=df.loc[(df.loc[:,'l']==0) & (df.loc[:,'fiability']==True)]
        df_pl_ok=df.loc[(df.loc[:,'l']==1) & (df.loc[:,'fiability']==True)]
        return df_vlpl, df_tv_plaques_ok, df_veh_ok, df_vl_ok, df_pl_ok 
        
    def stats (self,df):
        """les stats issue des df_filtrees
        A RENESEIGNER LES DF QUI SORTENT"""
        nb_tv_tot=len(df)
        nb_tv_plaque_ok=len(self.df_tv_plaques_ok)
        nb_vlpl=len(self.df_vlpl)
        nb_veh_ok=len(self.df_veh_ok)                      
        nb_vl_ok=len(self.df_vl_ok)
        nb_pl_ok=len(self.df_pl_ok)
        return nb_tv_tot, nb_tv_plaque_ok, nb_vlpl, nb_veh_ok, nb_vl_ok, nb_pl_ok
    
    def temp_max_cluster(self, delai):
        """obtenir le temps max de parcours en faisant un cluster par dbscan
        en entree : l'attribut df_tps_parcours_pl_final
                    le delai max pour regrouper en luster,en seconde
        en sortie : le nombre de clusters,
                    un timedelta
        """
        donnees_src=self.df_pl_ok.loc[:,['created_x','tps_parcours']].copy() #isoler les données necessaires
        temps_int=((pd.to_datetime('2018-01-01')+donnees_src['tps_parcours'])-pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')#convertir les temps en integer
        #mise en forme des données pour passer dans sklearn 
        donnnes = temps_int.values
        matrice=donnnes.reshape(-1, 1)
        #faire tourner la clusterisation et recupérer le label (i.e l'identifiant cluster) et le nombre de cluster
        clustering=DBSCAN(eps=delai, min_samples=len(temps_int)/2).fit(matrice)
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        # A AMELIORER EN CREANT UNE ERREUR PERSONALISEE SI ON OBTIENT  CLUSTER
        if n_clusters_== 0 :
            raise ClusterError()
        #mettre en forme au format pandas
        results = pd.DataFrame(pd.DataFrame([donnees_src.index,labels]).T)
        results.columns = ['index_base', 'cluster_num']
        results = pd.merge(results,self.df_pl_ok, left_on='index_base', right_index=True )
        #obtenir un timedelta unique
        temp_parcours_max=results.loc[results.loc[:,'cluster_num']!=-1].groupby(['cluster_num'])['tps_parcours'].max()
        temp_parcours_max=pd.to_timedelta(temp_parcours_max.values[0])
        
        return n_clusters_, temp_parcours_max
              
    def df_temps_parcours_bruts(self):
        """fonction de calcul du temps moyen de parcours entre 2 cameras
        en entree : dataframe : le dataframe format pandas qui contient les données
                    date_debut : string decrivant une date avec Y-M-D H:M:S : 
                     date de part d'analyse du temp moyen           
                    duree : integer : duree en minute
                     c'est le temps entre lequel on va regarder le temps mpyen : de 7h à 7h et 30 min par exemple
                    temps_max_autorise : integer : le nb heure max entre la camerade debut et de fin
                    camera1 : integer : camera de debut
                    camera2 : integer : camera de fin
        """
        df2=self.df.set_index('created').sort_index()
        df_duree=df2.loc[self.date_debut:self.date_fin]
        
        #trouver tt les bagnoles passée par cam1 dont la 2eme camera est cam2
        #isoler camera 1
        df_duree_cam1=df_duree.loc[df_duree.loc[:,'camera_id']==self.camera1]
        #on retrouve ces immatriculation mais qui ne sont pas à la 1ere camera
        df_duree_autres_cam=df2.loc[(df2.loc[:,'immat'].isin(df_duree_cam1.loc[:,'immat']))]
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
        # on regroupe les attributs dedescription de type etde fiabilite de camera dans des listes (comme ça si 3 camera on pourra faire aussi
        cam1_fastest_next['l']=cam1_fastest_next.apply(lambda x:self. test_unicite_type([x['l_x'],x['l_y']],mode='1/2'), axis=1)
        cam1_fastest_next['fiability']=cam1_fastest_next.apply(lambda x: all(element > 0 for element in [x['fiability_x'],x['fiability_y']]), axis=1)
        #on ne garde que les passage le plus rapide devant la camera 2
        cam1_puis_cam2=cam1_fastest_next.loc[cam1_fastest_next.loc[:,'camera_id_y']==self.camera2]
        #on trie puis on ajoute un filtre surle temps entre les 2 camera.
        cam1_cam2_passages=cam1_puis_cam2.set_index('created_y').sort_index()
        cam1_cam2_passages_filtres=cam1_cam2_passages[self.date_debut:self.date_debut+pd.Timedelta(hours=self.temps_max_autorise)]
        #on ressort la colonne de tempsde l'index et on cree la colonne des differentiel de temps
        cam1_cam2_passages_filtres=cam1_cam2_passages_filtres.reset_index()
        cam1_cam2_passages_filtres['tps_parcours']=cam1_cam2_passages_filtres['created_y']-cam1_cam2_passages_filtres['created_x'] #creer la colonne des differentiel de temps
        
        return cam1_cam2_passages_filtres
    
    def loc_trajet_global(self,df_journee, date_jour, duree, duree_max, cameras,liste_trajet_od ): 
        """
        fonction pour retrouver tous les pl d'une o_d une fois que l'on a identifé la duree_max entre 2 cameras
        permet de retrouver tous les pl apres avoir la duree du trajet indirect
        """
        camera1, camera2=cameras[0], cameras[1]
        date_jour=pd.to_datetime(date_jour)
        df2=df_journee.set_index('created').sort_index()
        #on limite le nb d'objet entre les 2 heures de depart
        df_duree=df2.loc[date_jour:date_jour+pd.Timedelta(minutes=duree)]
        #on trouve les veh passés cameras 1
        df_duree_cam1=df_duree.loc[df_duree.loc[:,'camera_id']==camera1]
        #on les retrouve aux autres cameras
        df_duree_autres_cam=df2.loc[(df2.loc[:,'immat'].isin(df_duree_cam1.loc[:,'immat']))]
        #on limite ces données selon le temps autorisé à partir de la date de depart
        df_autres_cam_temp_ok=df_duree_autres_cam.loc[date_jour:date_jour+pd.Timedelta(minutes=duree_max)]
        #on trie par heure de passage devant les cameras puis on regroupe et on liste les cameras devant lesquelles ils sont passés
        groupe=(df_autres_cam_temp_ok.sort_index().reset_index().groupby('immat').agg({'camera_id':lambda x : tuple(x), 'l': lambda x : self.test_unicite_type(list(x),'1/2'),
                                                                        'created':lambda x: x.max()}))
        #jointure avec la df de départ pour récupérer le passage devant la camera 1
        df_agrege=df_duree_cam1.join(groupe,on='immat',lsuffix='_left')[['immat', 'l','camera_id','created']].rename(columns={'created':'date_cam_2', 'camera_id':'cameras'})
        #temps de parcours
        df_agrege=df_agrege.reset_index().rename(columns={'created':'date_cam_1'})
        df_agrege['tps_parcours']=df_agrege.apply(lambda x : x.date_cam_2-x.date_cam_1, axis=1)
        #on ne garde que les vehicules passé à la camera 2 et qui sont des pl 
        df_trajet=df_agrege.loc[(df_agrege['cameras'].apply(lambda x : x[-1])==camera2) & (df_agrege['l']==1)]
        # on filtre les les cameras si ils ne sont pas dans les patterns prévus dans liste_trajet_total
        df_trajet_final=df_trajet.loc[df_trajet['cameras'].isin(liste_trajet_od)]
    
        return df_trajet_final
        
        
        
        
        
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
                       
    def plot_graphs(self, cluster=True):
        """cree et retourne les charts de altair pour un certines nombres de graph
        en sortie : graph_stat_trie : grphs de stat sur les nb de veh
                    graph_tps_bruts : graph de temps de passage entre 2 cam pour tout les types de veh non redresses 
        """
        
        #graph des stats de df
        stats_df=pd.DataFrame([{'type': 'nb_tv_tot', 'value':self.nb_tv_tot},{'type': 'nb_tv_plaque_ok', 'value':self.nb_tv_plaque_ok},
                               {'type': 'nb_vlpl', 'value':self.nb_vlpl},{'type': 'nb_veh_ok', 'value':self.nb_veh_ok},
                               {'type': 'nb_vl_ok', 'value':self.nb_vl_ok},{'type': 'nb_pl_ok', 'value':self.nb_pl_ok}])
        graph_stat=alt.Chart(stats_df).mark_bar(size=20).encode(
            x='type',
            y='value',
            color='type')
        graph_stat_trie = graph_stat.encode(alt.X(field='type', type='nominal',
                                            sort=alt.EncodingSortField(field='value',op='mean'))).properties()
        
        #graph des temps de parcours sur df non filtree, selection possible sur type de veh
        tps_parcours_bruts=self.df_tps_parcours_brut[['created_x','tps_parcours','l']].copy() #copier les données avec juste ce qu'il faut
        tps_parcours_bruts.tps_parcours=pd.to_datetime('2018-01-01')+tps_parcours_bruts.tps_parcours #refernce à une journée à 00:00 car timedeltas non geres par altair (json en general)
        tps_parcours_bruts['pl_90pctl']=pd.to_datetime('2018-01-01')+self.tps_pl_90_qtl
        tps_parcours_bruts['pl_85pctl']=pd.to_datetime('2018-01-01')+self.tps_pl_85_qtl
        
        selection = alt.selection_multi(fields=['l'])
        color = alt.condition(selection,
                      alt.Color('l:N', legend=None),
                      alt.value('lightgray'))
                
        graph_tps_bruts = alt.Chart(tps_parcours_bruts).mark_point().encode(
                        x='created_x',
                        y='hoursminutes(tps_parcours)',
                        color=color,
                        shape='l:N',
                        tooltip='tps_parcours').interactive()
        legend_graph_tps_bruts = alt.Chart(tps_parcours_bruts).mark_point().encode(
                                y=alt.Y('l:N', axis=alt.Axis(orient='right')),
                                color=color,
                                shape='l:N',
                                ).add_selection(
                                selection)
        
        graph_pl_ok=alt.Chart(tps_parcours_bruts.loc[tps_parcours_bruts.loc[:,'l']==1]).mark_point(color='gray').encode(
                                x='created_x',
                                y='hoursminutes(tps_parcours)',
                                tooltip='hoursminutes(tps_parcours)').interactive()
        graph_pl_90pctl=alt.Chart(tps_parcours_bruts.loc[tps_parcours_bruts.loc[:,'l']==1]).mark_line(color='blue').encode(
                                 x='created_x',
                                 y='hoursminutes(pl_90pctl)')
        graph_pl_85pctl=alt.Chart(tps_parcours_bruts.loc[tps_parcours_bruts.loc[:,'l']==1]).mark_line(color='red').encode(
                                 x='created_x',
                                 y='hoursminutes(pl_85pctl)')
        
        graph_prctl=graph_pl_ok + graph_pl_90pctl + graph_pl_85pctl
        
        if cluster : 
            tps_parcours_bruts['pl_cluster']=pd.to_datetime('2018-01-01')+self.temps_pour_filtre
            graph_pl_cluster=alt.Chart(tps_parcours_bruts.loc[tps_parcours_bruts.loc[:,'l']==1]).mark_line(color='yellow').encode(
                         x='created_x',
                         y='hoursminutes(pl_cluster)')
            graph_prctl=graph_pl_ok + graph_pl_90pctl + graph_pl_85pctl + graph_pl_cluster
            return  graph_stat_trie, graph_tps_bruts|legend_graph_tps_bruts, graph_prctl 
        
        return  graph_stat_trie, graph_tps_bruts|legend_graph_tps_bruts, graph_prctl

    def exporter_graph(self,path,o_d,graph):
        """
        Fonction pour exporter automatiquement certains graphs
        """
        date=self.date_debut.strftime("%Y-%m-%d")
        heures=self.date_debut.strftime("%Hh")+'-'+self.date_fin.strftime("%Hh")
        if not os.path.exists(os.path.join(path,o_d, date)) :
            os.makedirs(os.path.join(path,o_d, date),exist_ok=True)
        path=os.path.join(os.path.join(path,os.path.join(o_d, date)),'_'.join([heures,str(self.camera1),str(self.camera2)])+'.png' )
        graph.save(path, scale_factor=2.0)
        

class trajet_indirect():
    """
    classe pour les trajets passant par plus de 2 cameras
    """
    
    def __init__(self,df, date_debut, duree, temps_max_autorise, cameras):
        """
        constructeur. se base sur classe trajet_direct
        df : dataframe sur 2 jours issue de ouvrir_fichier_lapi
        date debut : string de type Y-M-D H:M:S ou pandas datetime
        duree : plage de passage des veh devant la 1ere camera, en minutes
        temps_max_autorise : temps autorise pour passage devant la derniere cameras, en heure
        cameras : liste des cameras devant lesquelles le veh doit passer
        """

        self.cameras_suivantes=cameras
        self.df, self.date_debut, self.duree, self.temps_max_autorise=df, date_debut, duree, temps_max_autorise
        self.nb_cam=len(cameras)
        
        #il y a des trajets directs redondant inérent à tous les trajets indirects : les trajets de départ entre les cameras [14,19], [16,19] et [12-6], qui sont répétés 
        #respectivement 11*, 11* et 17*
        #on peut donc ne les déterminer qu'une fois, et rappeler directement ces resultats pour les fois suivantes
        
        #dico de resultats
        self.dico_traj_directs=self.liste_trajets_directs()
        self.df_transit=self.df_trajet_indirect()
        self.temps_parcours_max=self.df_transit.tps_parcours.max()

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
                trajet=trajet_direct(self.df, self.date_debut, self.duree, self.temps_max_autorise, couple_cam[0], couple_cam[1])
            else : 
                cle_traj_prec='trajet'+str(indice-1)
                trajet=(trajet_direct(self.df, dico_traj_directs[cle_traj_prec].timestamp_mini,
                                         dico_traj_directs[cle_traj_prec].duree_traj_fut,self.temps_max_autorise,
                                         couple_cam[0], couple_cam[1]))
            dico_traj_directs[nom_variable]=trajet
        
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
            return self.dico_traj_directs['trajet0'].df_tps_parcours_pl_final
        for a, val_dico in enumerate(self.dico_traj_directs):
            if a<=long_dico-1:
                #print(f"occurence {a} pour trajet : {val_dico}, lg dico_:{long_dico}")
                variab,variab2 ='trajet'+str(a), 'trajet'+str(a+1)
                if self.dico_traj_directs[variab].df_tps_parcours_pl_final.empty : #si un des trajets aboutit a empty, mais pas le 1er
                    return self.dico_traj_directs[variab].df_tps_parcours_pl_final
                if a==0 :
                    df_transit=pd.merge(self.dico_traj_directs[variab].df_tps_parcours_pl_final,self.dico_traj_directs[variab2].df_tps_parcours_pl_final,on='immat')
                    df_transit['tps_parcours']=df_transit['tps_parcours_x']+df_transit['tps_parcours_y']
                    df_transit=(df_transit.rename(columns=dico_rename))[['immat','date_cam_1','date_cam_2','tps_parcours']]  
                elif a<long_dico-1: 
                    #print(f" avant boucle df_trajet_indirect, df_transit : {df_transit.columns}")
                    df_transit=pd.merge(df_transit,self.dico_traj_directs[variab2].df_tps_parcours_pl_final,on='immat')
                    #print(f" apres boucle df_trajet_indirect, df_transit : {df_transit.columns}")
                    df_transit['tps_parcours']=df_transit['tps_parcours_x']+df_transit['tps_parcours_y']
                    df_transit=(df_transit.rename(columns=dico_rename))[['immat','date_cam_1','date_cam_2','tps_parcours']]
            #print(f"1_df_trajet_indirect, df_transit : {df_transit.columns}")
        #print(f"2_df_trajet_indirect, df_transit : {df_transit.columns}")
        
        df_transit['cameras']=df_transit.apply(lambda x:list(self.cameras_suivantes), axis=1)
        #print(df_transit)

        return df_transit
    
    def exporter_graph(self,path,o_d) :
        """
        fonction d'export des graphes en se basant sur la fonction homonyme de la classe trajet_direct
        """
        for trajet in self.dico_traj_directs.values() :
            trajet.exporter_graph(path,o_d,trajet.graph_prctl)
        
class ClusterError(Exception):       
    def __init__(self):
        Exception.__init__(self,'nb de Cluster valable = 0 ') 

class PasDePlError(Exception):  
    """
    Exception levee si le trajet direct ne comprend pas de pl
    """     
    def __init__(self):
        Exception.__init__(self,'pas de PL sur la période et les cameras visées') 

    
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
            print(f"trajet : {origine}-{destination}, date : {date}, debut_traitement : {dt.datetime.now()}")
            try : 
                donnees_trajet=trajet(df_journee,date,60,16,cameras, type='Global')
                df_trajet, df_passag=donnees_trajet.df_global, donnees_trajet.df_passag_transit
            except PasDePlError :
                continue
            
            df_trajet['o_d'],df_trajet['origine'],df_trajet['destination']=o_d, origine, destination
            dico_trajet_od[o_d], dico_passag_od=df_trajet,df_passag
            """
            #por dico total
            if 'dico_od' in locals() : #si la varible existe deja on la concatene avec le reste
                dico_od=pd.concat([dico_od,df_trajet], sort=False)
            else : #sinon on initilise cette variable
                dico_od=df_trajet  
            if 'dico_passag' in locals() : #si la varible existe deja on la concatene avec le reste
                dico_passag=pd.concat([dico_passag,df_passag], sort=False)
            else : #sinon on initilise cette variable
                dico_passag=df_passag
            """  
                
    return dico_trajet_od, dico_passag_od           
                
                
def transit_temps_complet(date_debut, nb_jours,liste_trajets):
    #utiliser ouvrir_fichier_lapi pour ouvrir un df sur 3 semaine
    date_fin=(pd.to_datetime(date_debut)+pd.Timedelta(days=nb_jours)).strftime('%Y-%m-%d')
    print(f"import  : {dt.datetime.now()}")
    df_3semaines=ouvrir_fichier_lapi(date_debut,date_fin)
    #selection de 1 jour par boucle
    print(f" fin import  : {dt.datetime.now()}")
    for date in pd.date_range(date_debut, periods=nb_jours, freq='D') :
        df_journee=df_3semaines.loc[date:date+pd.Timedelta(days=2)]
        df_transit_jour, df_passage_jour=transit_1_jour(df_journee,date,liste_trajets)
        
        if 'df_transit_total' in locals() : #si la varible existe deja on la concatene avec le reste
                df_transit_total=pd.concat([df_transit_total,df_trajet], sort=False)
        else : #sinon on initilise cette variable
                df_transit_total=df_transit_jour 
        if 'df_passag_total' in locals() : #si la varible existe deja on la concatene avec le reste
                df_passag_total=pd.concat([df_passag_total,df_trajet], sort=False)
        else : #sinon on initilise cette variable
                df_passag_total=df_passage_jour
    #se baser la dessus pour lancer transit 1 jour
    #stocker les résultats de transit1jour dans une df au fur et a mesure
    return df_transit_total

def pourcentage_pl_camera():
    #isoler les pl 
    pass
    
    
    
    