# -*- coding: utf-8 -*-
'''
Created on 27 fev. 2019
@author: martin.schoreisz

Module de traitement des donnees lapi

'''

import matplotlib #pour éviter le message d'erreurrelatif a rcParams
import pandas as pd
import geopandas as gp
import Connexion_Transfert as ct
import altair as alt

def ouvrir_fichier_lapi(date_debut, date_fin) : 
    with ct.ConnexionBdd('gti_lapi') as c : 
        requete=f"select camera_id, created, immat, fiability, l, state from data.te_passage where created between '{date_debut}' and '{date_fin}'"
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

class df_tps_parcours():
    """
    Classe decrivant les df des temps de parcours 
    attributs : 
    - tous les attributs de la classe df_source
    - tps_vl_90_qtl : integer : vitesse en dessous de laquelle circule 90 % des vl
    - tps_pl_90_qtl : integer : vitesse en dessous de laquelle circule 90 % des pl
    """
    
    def __init__(self,df, date_debut, duree, temps_max_autorise, camera1, camera2):
        """
        constrcuteur
        en entree : dataframe : le dataframe format pandas qui contient les données
                date_debut : string ou pd.timestamps decrivant une date avec Y-M-D HH:MM:SS  : date de part d'analyse du temp moyen           
                duree : integer : duree en minute : c'est le temps entre lequel on va regarder le temps mpyen : de 7h à 7h et 30 min par exemple
                temps_max_autorise : integer : le nb heure max entre la camerade debut et de fin
                camera1 : integer : camera de debut
                camera2 : integer : camera de fin
        """
        self.df=df
        self.df, self.duree, self.temps_max_autorise, self.camera1, self.camera2=df,date_debut, duree, temps_max_autorise, camera1, camera2
        
        #j'en fais un attribut car util edans la fonction recherche_trajet_indirect
        self.date_debut=pd.to_datetime(self.date_debut)
        self.date_fin=self.date_debut+pd.Timedelta(minutes=self.duree) 
        
        self.df_tps_parcours_brut=self.df_temps_parcours_bruts()
        self.df_vlpl, self.df_tv_plaques_ok, self.df_veh_ok, self.df_vl_ok,self.df_pl_ok=self.df_filtrees(self.df_tps_parcours_brut)
        self.nb_tv_tot, self.nb_tv_plaque_ok, self.nb_vlpl, self.nb_veh_ok, self.nb_vl_ok, self.nb_pl_ok=self.stats(self.df_tps_parcours_brut) 
    
        #filtre statistique : on en garde que ce qui est les 90 % de temps les plus rapides
        self.tps_vl_90_qtl=self.df_vl_ok.tps_parcours.quantile(0.9)
        self.tps_pl_90_qtl=self.df_pl_ok.tps_parcours.quantile(0.9)
        
        
        #df finaux : un df des vehicules passe par les 2 cameras dans l'ordre, qui sont ok en plque et en typede véhicules
        self.df_tps_parcours_vl_final=self.df_vl_ok.loc[(self.df_vl_ok.loc[:,'tps_parcours']<=self.tps_vl_90_qtl)]
        self.df_tps_parcours_pl_final=self.df_pl_ok.loc[(self.df_pl_ok.loc[:,'tps_parcours']<=self.tps_pl_90_qtl)]
    
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
        #on retrouve ces immatriculation
        df_duree_autres_cam=df2.loc[df2.loc[:,'immat'].isin(df_duree_cam1.loc[:,'immat'])]
        #on fait une jointure entre cam 1 et les autres cam pour avoir une correspondance entre le passage devan la 1ere cmaera et la seconde
        cam1_croise_autre_cam=df_duree_cam1.reset_index().merge(df_duree_autres_cam.reset_index(), how='left', on='immat')
        #on ne garde que les passages à la 2ème caméra postérieur au passage à la première
        cam1_croise_suivant=cam1_croise_autre_cam.loc[(cam1_croise_autre_cam.loc[:,'created_x']<cam1_croise_autre_cam.loc[:,'created_y'])]
        #on isole le passage le plus rapide devant cam suivante pour chaque immatriculation
        cam1_fastest_next=cam1_croise_suivant.loc[cam1_croise_suivant.groupby(['immat'])['created_y'].idxmin()]
        # on regroupe les attributs dedescription de type etde fiabilite de camera dans des listes (comme ça si 3 camera on pourra faire aussi
        cam1_fastest_next['l']=cam1_fastest_next.apply(lambda x:self. test_unicite_type([x['l_x'],x['l_y']]), axis=1)
        cam1_fastest_next['fiability']=cam1_fastest_next.apply(lambda x: all(element > 80 for element in [x['fiability_x'],x['fiability_y']]), axis=1)
        #on ne garde que les passage le plus rapide devant la camera 2
        cam1_puis_cam2=cam1_fastest_next.loc[cam1_fastest_next.loc[:,'camera_id_y']==self.camera2]
        #on trie puis on ajoute un filtre surle temps entre les 2 camera.
        cam1_cam2_passages=cam1_puis_cam2.set_index('created_y').sort_index()
        cam1_cam2_passages_filtres=cam1_cam2_passages[date_debut:date_debut+pd.Timedelta(hours=self.temps_max_autorise)]
        #on ressort la colonne de tempsde l'index et on cree la colonne des differentiel de temps
        cam1_cam2_passages_filtres=cam1_cam2_passages_filtres.reset_index()
        cam1_cam2_passages_filtres['tps_parcours']=cam1_cam2_passages_filtres['created_y']-cam1_cam2_passages_filtres['created_x'] #creer la colonne des differentiel de temps
     
        return cam1_cam2_passages_filtres

    def test_unicite_type(self,liste_l):
        """test pour voir si un vehicule a ete toujours vu de la mme façon ou non
           en entre : liste de valeur de l iisues d'une df""" 
        if len(set(liste_l))==1 :
            return liste_l[0]
        else : 
            return -1
                               
    def plot_graphs(self):
        """cree et retourne les charts de altair pour un certines nombres de graph"""
        
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
        
        #graph des temps de parcours sur df non filtree
        tps_parcours_bruts=self.df_tps_parcours_brut[['created_x','tps_parcours','l']].copy() #copier les données avec juste ce qu'il faut
        tps_parcours_bruts.tps_parcours=pd.to_datetime('2018-01-01')+tps_parcours_bruts.tps_parcours #refernce à une journée à 00:00 car timedeltas non geres par altair (json en general)
        graph_tps_bruts=alt.Chart(tps_parcours_bruts).mark_point().encode(
                                x='created_x',
                                y='hoursminutes(tps_parcours)',
                                color='l:N',
                                shape='l:N').interactive()
        
        return  graph_stat_trie, graph_tps_bruts
        
def recherche_trajet_indirect(df_trajet_1, cam1_trajet2, cam2_trajet2):
    """
    Recherche des vehicules passés entre 2 cameras qui sont ensuite passés entre 2 autres
    en entrée : 
     - df_trajet_1 : dataframe : df_tps_parcours_pl_final issus de la classe df_source pour le tajet 1
     - cam1_trajet2 : integer : amera 1 du deuxeime trajet
     - cam2_trajet2 : integer : camera 2 du deuxieme trajet
    """
    # rechercher le temps de parcours mini et max d'un pl passé entre 8h et 9h entre camera 19 et 4
    timedelta_min=df_trajet_1.df_tps_parcours_pl_final.tps_parcours.min()
    timedelta_min=df_trajet_1.df_tps_parcours_pl_final.tps_parcours.max()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
     
    