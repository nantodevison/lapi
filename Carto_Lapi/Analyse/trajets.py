# -*- coding: utf-8 -*-
'''
Created on 21 juin 2019

@author: martin.schoreisz

Calcul des trajets relevant des O-D identifiées dans le module Import_Forme
'''



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
        temps_max_autorise -- le temps que l'on autorise pour trouver les vehicules passés par cam1.
        typeCluster -- pour typeTrajet='Global' : regroupement des temps de parcours par o_d ou parcameras parcourue 
    """
    
    def __init__(self,df,date_debut, duree, cameras,typeTrajet='Direct',df_filtre=None,temps_max_autorise=18, 
                 liste_trajet=liste_complete_trajet,typeVeh=1 ) :
        """
        Constrcuteur
        Attributs de sortie : 
           df_transit : pandas dataframe des trajets avec immat, date_cam_1, date_cam_2, tps_parcours (cameras en plus en Indirect ou Global)
           temps_parcours_max : : temps de parcours max mmoyen, issu de Cluster ou percentile
           tps_parcours_max_type : : source du tempsde parcours max
           df_passag_transit : pandas dataframe: uniquemnet pour global : liste des passages consideres comme du transit
        """
                #en prmeier on verifie que le typede trajet colle avec le nb de camera
        if ((len(cameras) >2 and typeTrajet != 'Indirect') or typeTrajet not in ['Direct','Global','Indirect'] 
            or len(cameras)<2 or (len(cameras)==2 and typeTrajet == 'Indirect')):
           raise TypeTrajet_NbCamera_Error(len(cameras),typeTrajet)
        
        #en fonction du df qui est passé on met la date de creation en index ou non
        if isinstance(df.index,pd.DatetimeIndex) :
            self.df=df
        else :
            self.df=df.set_index('created').sort_index()
            
        #on filtre sur le type de vehicule étudié
        self.df=self.df.loc[self.df['l']==typeVeh].copy()
        
        if self.df.empty:    
            raise PasDePlError()
        
        #attributs
        self.date_debut, self.duree, self.cameras_suivantes, self.temps_max_autorise=pd.to_datetime(date_debut), duree, cameras,temps_max_autorise
        self.typeTrajet = typeTrajet
        self.date_fin=self.date_debut+pd.Timedelta(minutes=self.duree)
        self.df_duree=self.df.loc[self.date_debut:self.date_fin] 
        
        #calcul des df
        if typeTrajet=='Direct' :
            self.df_transit=self.trajet_direct()
            self.timedelta_min,self.timedelta_max,self.timestamp_mini,self.timestamp_maxi,self.duree_traj_fut=self.temps_timedeltas_direct()
        elif typeTrajet=='Global' :
                self.df_transit, self.df_passag_transit=self.loc_trajet_global(df_filtre,liste_trajet)
        else : 
            self.dico_traj_directs=self.liste_trajets_directs()
            self.df_transit=self.df_trajet_indirect()
        
        #temps de parcours
        if typeTrajet in ['Direct', 'Indirect'] :
            try : 
                self.temps_parcours_max=temp_max_cluster(self.df_transit,800)[1]
                self.tps_parcours_max_type='Cluster'
            except ClusterError :
                self.temps_parcours_max=self.df_transit.tps_parcours.quantile(0.85)
                self.tps_parcours_max_type='85eme_percentile'
        else : 
            dico_tps_max={}
            dico_tps_max['date'], dico_tps_max['temps'], dico_tps_max['type'], dico_tps_max['o_d']=[],[],[],[]
            for od in self.df_transit.o_d.unique().tolist():
                try : 
                    t_ref=15 if self.date_debut.hour in [6,7,8,14,15,16,17,18,19] else 60 
                    delai_ref=param_cluster[(param_cluster.trajet.apply(lambda x : od in x)) & (param_cluster['temps_etudie']==t_ref)].delai.values[0]  
                    coef=param_cluster[(param_cluster.trajet.apply(lambda x : od in x)) & (param_cluster['temps_etudie']==t_ref)].nb_pt_min.values[0]
                    temps_parcours_max=temp_max_cluster(self.df_transit.loc[self.df_transit['o_d']==od],delai_ref,coef)[1]
                    tps_parcours_max_type='Cluster'
                except ClusterError :
                    temps_parcours_max=self.df_transit.loc[self.df_transit['o_d']==od].tps_parcours.quantile(0.85)
                    tps_parcours_max_type='85eme_percentile'
                dico_tps_max['date'].append(self.date_debut)
                dico_tps_max['temps'].append(temps_parcours_max)
                dico_tps_max['type'].append(tps_parcours_max_type)
                dico_tps_max['o_d'].append(od)    
            self.temps_parcours_max=pd.DataFrame(dico_tps_max)
        
    def trajet_direct(self):
        """
        Fonction de calcul des trajets de transit entre la camera 1 et la camera 2 du constructeur
        renvoi une pandas dataframe avec les attributs : 
            immat : immatriculation cry^ptée
            date_cam_1 : pandas datetime : date de passage devant la première camera
            date_cam_2 : pandas datetime : date de passage devant la deuxieme camera    
            tps_parcours : pandas timedelta : tempsde parcours entre les 2 cameras
        """
        #trouver tt les bagnoles passée par cam1 dont la 2eme camera est cam2
        #isoler camera 1
        cam1_puis_cam2=trouver_passages_consecutif(self.df, self.date_debut, self.date_fin,self.cameras_suivantes[0], self.cameras_suivantes[1])
        #pour la fiabilite on peut faire varier le critere. ici c'est 0 : tous le spassages sont pris
        cam1_puis_cam2['fiability']=cam1_puis_cam2.apply(lambda x: all(element > 0 for element in [x['fiability_x'],x['fiability_y']]), axis=1)
        #on trie puis on ajoute un filtre surle temps entre les 2 camera.
        cam1_cam2_passages=cam1_puis_cam2.set_index('created_y').sort_index()
        cam1_cam2_passages_filtres=cam1_cam2_passages[self.date_debut:self.date_debut+pd.Timedelta(hours=self.temps_max_autorise)]
        #on ressort la colonne de tempsde l'index et on cree la colonne des differentiel de temps
        cam1_cam2_passages_filtres=cam1_cam2_passages_filtres.reset_index()
        cam1_cam2_passages_filtres['tps_parcours']=cam1_cam2_passages_filtres['created_y']-cam1_cam2_passages_filtres['created_x'] #creer la colonne des differentiel de temps
        #isoler les passages fiables
        df_pl=cam1_cam2_passages_filtres.loc[cam1_cam2_passages_filtres.loc[:,'fiability']==True]
        df_tps_parcours_pl_final=df_pl[['immat','created_x', 'created_y','tps_parcours']].rename(columns=dico_renommage)
        df_tps_parcours_pl_final['cameras']=df_tps_parcours_pl_final.apply(lambda x : tuple(self.cameras_suivantes),axis=1)
        if df_tps_parcours_pl_final.empty :
            raise PasDePlError()
        
        return df_tps_parcours_pl_final
 
    def liste_trajets_directs(self):
        """
        pour obtenir un dico contenant les instances de trajet pour chaque trajet éleémentaires
        se base sur la liste des camerras du constcructeur
        en sortie : 
            dico_traj_direct : dictionnaire avec comme key une valeur 'trajet'+integer et en value une instance de la classe trajet issue du'un calcul de trajet Direct
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
        Assemblage des df du dico issu de liste_trajets_directs
        renvoi une pandas dataframe avec les attributs : 
            immat : immatriculation cry^ptée
            date_cam_1 : pandas datetime : date de passage devant la première camera
            date_cam_2 : pandas datetime : date de passage devant la deuxieme camera    
            tps_parcours : pandas timedelta : tempsde parcours entre les 2 cameras
            cameras : liste des cameras parcourue
        """
        dico_rename={'date_cam_1_x':'date_cam_1','date_cam_2_y':'date_cam_2'} #paramètres pour mise en forme donnees
        
        #on fait une jointure de type 'inner, qui ne garde que les lignes présentes dans les deux tables, en iterant sur chaque element du dico
        long_dico=len(self.dico_traj_directs)
        #print (self.dico_traj_directs)
        if long_dico<2 : #si c'est le cas ça veut dire une seule entree dans le dico, ce qui signifie que l'entree est empty, donc on retourne une df empty pour etre raccord avec le type de donnees renvoyee par trajet_direct dans ce cas
            raise PasDePlError()
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

        df_transit['cameras']=df_transit.apply(lambda x:tuple(self.cameras_suivantes), axis=1)

        return df_transit
        
    def loc_trajet_global(self,df_filtre,liste_trajet): 
        """
        fonction de detection des trajets pour tous les destinations possibles de la camera 1 du constrcuteur.
        Nécessite l'utilisation de la variable module liste_complete_trajet qui contient tous les trajets possible en entree-sortie
        En sortie : 
            df_transit : pandas dataframe conteant les mm colonnes que pour direct et indirects
            df_passag_transit : pandas dataframe conteant les passages considérés en transit
            
        """       
        groupe_pl,df_duree_cam1,df_duree_autres_cam=grouper_pl(self.df, self.date_debut, self.date_fin, self.cameras_suivantes[0],df_filtre)
        
        df_agrege=filtre_et_forme_passage(self.cameras_suivantes[0],groupe_pl, liste_trajet, df_duree_cam1)
        
        #pour obtenir la liste des passagesrelevant de trajets de transits :
        #limitation des données des cameras par jointures
        df_passag_transit=trajet2passage(df_agrege,df_duree_autres_cam)
        """df_joint_passag_transit=df_agrege.merge(df_duree_autres_cam.reset_index(), on='immat')
        df_passag_transit1=df_joint_passag_transit.loc[(df_joint_passag_transit.apply(lambda x : x['camera_id'] in x['cameras'], axis=1))]
        df_passag_transit=(df_passag_transit1.loc[df_passag_transit1.apply(lambda x : x['date_cam_1']<=x['created']<=x['date_cam_2'], axis=1)]
                        [['created','camera_id','immat','fiability','l_y','state_x']].rename(columns={'l_y':'l','state_x':'state'}))"""
        
        #et par opposition la liste des passages ne relevant pas des trajets de transit        
        return df_agrege,df_passag_transit
        
    def temps_timedeltas_direct(self):
        """
        Calcul des temps permettant de cahiner les trajets directs. Utile pour trajet indirects
        """
        
        timedelta_min=self.df_transit.tps_parcours.min()
        timedelta_max=self.df_transit.tps_parcours.max()
        timestamp_mini=self.date_debut+timedelta_min
        timestamp_maxi=self.date_fin+timedelta_max
        duree_traj_fut=math.ceil(((timestamp_maxi-timestamp_mini)/ np.timedelta64(1, 'm')))
        
        return timedelta_min,timedelta_max,timestamp_mini,timestamp_maxi,duree_traj_fut
        
    def graph(self):
        """
        Pour obtenir le graph des temps de parcours avec la ligne du temps de parcours max associé
        en entre : df_transit et temps_parcours_max issu du constructeur,
        en sortie :  graph_tps_parcours : char altair
        """
        copie_df=self.df_transit.copy()
        copie_df.tps_parcours=pd.to_datetime('2018-01-01')+copie_df.tps_parcours
        if self.typeTrajet !='Global' :
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
        else : 
            print (f"liste des o-d : {self.df_transit.o_d.unique()}")
            dico_graph={}
            copie_df=copie_df.merge(self.temps_parcours_max, on='o_d')
            copie_df.temps=pd.to_datetime('2018-01-01')+copie_df.temps

            for od in self.df_transit.o_d.unique() :
                copie_df_tps_max=copie_df.loc[copie_df['o_d']==od][['cameras', 'temps', 'type']].drop_duplicates('temps').copy()
                index_temps=pd.DataFrame(pd.DatetimeIndex(pd.date_range(start=self.date_debut, end=self.date_fin, periods=len(copie_df_tps_max))), columns=['date_cam_1'])
                prod_cartesien=copie_df_tps_max.assign(key=1).merge(index_temps.assign(key=1), on='key')
                points=alt.Chart(copie_df.loc[copie_df['o_d']==od]).mark_point().encode(
                                x='date_cam_1',
                                y='hoursminutes(tps_parcours):T',
                                color='cameras',
                                shape='cameras',
                                tooltip='hoursminutes(tps_parcours)').interactive()
                line = alt.Chart(copie_df.loc[copie_df['o_d']==od]).mark_line().encode(
                                x='date_cam_1',
                                y='hoursminutes(temps):T',color='type').interactive()
                dico_graph[od]=points+line
            return dico_graph