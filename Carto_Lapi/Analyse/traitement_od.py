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
import os
from sklearn.cluster import DBSCAN

dico_renommage={'created_x':'date_cam_1', 'created_y':'date_cam_2'}
fichier_trajet=pd.read_json(r'Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\liste_trajets.json')

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

class trajet_direct():
    """
    Classe decrivant les df des temps de parcours 
    attributs : 
    - tous les attributs de la classe df_source
    - tps_vl_90_qtl : integer : vitesse en dessous de laquelle circule 90 % des vl
    - tps_pl_85_qtl : integer : vitesse en dessous de laquelle circule 80 % des pl
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
        self.tps_vl_90_qtl=self.df_vl_ok.tps_parcours.quantile(0.9)
        self.tps_pl_90_qtl=self.df_pl_ok.tps_parcours.quantile(0.9)  
        self.tps_pl_85_qtl=self.df_pl_ok.tps_parcours.quantile(0.85) 
        try : 
            self.tps_pl_cluster =self.temp_max_cluster(300)[1]
            if avecGraph : 
                self.graph_stat_trie, self.graph_tps_bruts, self.graph_prctl=self.plot_graphs()
        except ClusterError : 
            print(f"pas de cluster pour trajet {self.camera1, self.camera2} entre  {self.date_debut,self.date_fin} ")
            if avecGraph :
                self.graph_stat_trie, self.graph_tps_bruts, self.graph_prctl=self.plot_graphs(False)
        
        #resultats finaux finaux : un df des vehicules passe par les 2 cameras dans l'ordre, qui sont ok en plque et en typede véhicules
        self.df_tps_parcours_vl_final=self.df_vl_ok[['immat','created_x', 'created_y','tps_parcours']].rename(columns=dico_renommage)
        self.df_tps_parcours_pl_final=self.df_pl_ok[['immat','created_x', 'created_y','tps_parcours']].rename(columns=dico_renommage)
        if not self.df_tps_parcours_pl_final.empty: #je met un if car sinon ça me modifie le empty et cree le bordel par la suite
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
        clustering=DBSCAN(eps=delai, min_samples=len(temps_int)/1.5).fit(matrice)
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
        #on retrouve ces immatriculation
        df_duree_autres_cam=df2.loc[df2.loc[:,'immat'].isin(df_duree_cam1.loc[:,'immat'])]
        #on fait une jointure entre cam 1 et les autres cam pour avoir une correspondance entre le passage devan la 1ere cmaera et la seconde
        cam1_croise_autre_cam=df_duree_cam1.reset_index().merge(df_duree_autres_cam.reset_index(), how='left', on='immat')
        #on ne garde que les passages à la 2ème caméra postérieur au passage à la première
        cam1_croise_suivant=cam1_croise_autre_cam.loc[(cam1_croise_autre_cam.loc[:,'created_x']<cam1_croise_autre_cam.loc[:,'created_y'])]
        #on isole le passage le plus rapide devant cam suivante pour chaque immatriculation
        cam1_fastest_next=cam1_croise_suivant.loc[cam1_croise_suivant.groupby(['immat'])['created_y'].idxmin()]
        # on regroupe les attributs dedescription de type etde fiabilite de camera dans des listes (comme ça si 3 camera on pourra faire aussi
        cam1_fastest_next['l']=cam1_fastest_next.apply(lambda x:self. test_unicite_type([x['l_x'],x['l_y']],mode='1/2'), axis=1)
        cam1_fastest_next['fiability']=cam1_fastest_next.apply(lambda x: all(element > 50 for element in [x['fiability_x'],x['fiability_y']]), axis=1)
        #on ne garde que les passage le plus rapide devant la camera 2
        cam1_puis_cam2=cam1_fastest_next.loc[cam1_fastest_next.loc[:,'camera_id_y']==self.camera2]
        #on trie puis on ajoute un filtre surle temps entre les 2 camera.
        cam1_cam2_passages=cam1_puis_cam2.set_index('created_y').sort_index()
        cam1_cam2_passages_filtres=cam1_cam2_passages[self.date_debut:self.date_debut+pd.Timedelta(hours=self.temps_max_autorise)]
        #on ressort la colonne de tempsde l'index et on cree la colonne des differentiel de temps
        cam1_cam2_passages_filtres=cam1_cam2_passages_filtres.reset_index()
        cam1_cam2_passages_filtres['tps_parcours']=cam1_cam2_passages_filtres['created_y']-cam1_cam2_passages_filtres['created_x'] #creer la colonne des differentiel de temps
        
        return cam1_cam2_passages_filtres

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
            tps_parcours_bruts['pl_cluster']=pd.to_datetime('2018-01-01')+self.tps_pl_cluster
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
        """

        self.cameras_suivantes=cameras
        self.df, self.date_debut, self.duree, self.temps_max_autorise=df, date_debut, duree, temps_max_autorise
        self.nb_cam=len(cameras)
        
        #dico de resultats
        self.dico_traj_directs=self.liste_trajets_directs()
        self.df_transit=self.df_trajet_indirect()

    def liste_trajets_directs(self):
        """
        pour obtenir un dico contenant les instances de trajet_direct pour chaque trajet éleémentaires
        """  
        dico_traj_directs={}    
        #pour chaque couple de camera
        for indice,couple_cam in enumerate([[self.cameras_suivantes[i],self.cameras_suivantes[i+1]] for i in range(len(self.cameras_suivantes)-1)]) :
            #initialisation du nom de variables pour le dico resultat 
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
            if trajet.df_tps_parcours_pl_final.empty : #des qu'un des trajets elementaires est vide on s'arret, et on retourne le dico avec un vide
                return dico_traj_directs
        
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
    dates= pd.date_range(date_jour, periods=2, freq='H') #générer les dates par intervalle d'1h
    #parcourir les dates
    for date in dates : 
        date=date.strftime("%Y-%m-%d %H:%M:%S")
        #parcourir les trajets possibles
        for index, value in liste_trajets.iterrows() :
            o_d, carac_trajet=value[0],value[1]
            for dico_carac in carac_trajet : #carle json des trajets est de type record
                cameras=dico_carac['cameras']
                type_t=dico_carac['type_trajet']
                print(f"trajet : {o_d}, cameras : {cameras}")
                if type_t=='indirect' : # dans ce cas on appelle la classe correspondante
                    try :
                        trajet=trajet_indirect(df_journee,date, 60, 16, cameras)
                    except PasDePlError : #si pas de pl on boucle sur le trajet suivant
                        print("pas de pl")
                        continue 
                    df_trajet=trajet.df_transit#en deduire le total
                    if save_graphs : trajet.exporter_graph(r'Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\graphs',o_d) #si on exporte les graphs des trajets directs
                else :
                    try : 
                        trajet=trajet_direct(df_journee,date, 60, 16, cameras[0],cameras[1])
                    except PasDePlError :
                        print("pas de pl")
                        continue
                    df_trajet=trajet.df_tps_parcours_pl_final
                    if save_graphs : trajet.exporter_graph(r'Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\graphs',o_d,trajet.graph_prctl)    
                df_trajet['o_d']=o_d
                #stocker les resultats
                if 'dico_od' in locals() : #si la varible existe deja on la concatene avec le reste
                    dico_od=pd.concat([dico_od,df_trajet], sort=False)
                else : #sinon on initilise cette variable
                    dico_od=df_trajet

    return dico_od 
