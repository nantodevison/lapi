# -*- coding: utf-8 -*-
'''
Created on 21 juin 2019

@author: martin.schoreisz

Calcul des trajets relevant des O-D identifi�es dans le module Import_Forme
'''
from Import_Forme import liste_complete_trajet, param_cluster
import pandas as pd
import numpy as np
import altair as alt
import math
from sklearn.cluster import DBSCAN

dico_renommage={'created_x':'date_cam_1', 'created_y':'date_cam_2'}

def trajet2passage(df_trajets, df_passage) : 
    """
    Fonction de passage d'une df des trajets vers la df des passages la composant
    en entree :
        df_trajets : df des trajets (formalisme selon class trajet)
        df_passage : df des passages (formalisme selon export depuis Bdd)
    en sortie : 
        df_passag_transit : df des passages compris dans les trajets
    """
    df_passages_transit=df_trajets.merge(df_passage.reset_index(), on='immat')
    df_passag_transit1=df_passages_transit.loc[(df_passages_transit.apply(lambda x : x['camera_id'] in x['cameras'], axis=1))]
    df_passag_transit=(df_passag_transit1.loc[df_passag_transit1.apply(lambda x : x['date_cam_1']<=x['created']<=x['date_cam_2'], axis=1)]
                [['created','camera_id','immat','fiability','l_y','state_x']].rename(columns={'l_y':'l','state_x':'state'}))
    return df_passag_transit

def filtrer_df(df_global,df_filtre): 
    df_global=df_global.reset_index().set_index(['created','immat'])
    df_filtre=df_filtre.reset_index().set_index(['created','immat'])
    df_global_filtre=df_global.loc[~df_global.index.isin(df_filtre.index)].reset_index().set_index('created')
    return df_global_filtre  

def trouver_passages_consecutif(df, date_debut, date_fin, camera_1, camera_2) : 
    """
    pour obtenir une df des immat passées par une camera puis de suite une autre
    en entree : 
        df : df des passages
        date_debut : string de type YYYY-MM-DD
        date_fin : string de type YYYY-MM-DD
        camera_1 : entier camera de debut
        camera_2 : entier camera de fin
    en sortie : 
        cam1_puis_cam2 : df des immat avec la camera 1 puis la seconde camera vue de suite apres
    """
    df_duree=df.loc[date_debut:date_fin] #limiter la df de base
    df_duree_cam1=df_duree.loc[df_duree.loc[:,'camera_id']==camera_1]#isoler camera 1
    df_duree_autres_cam=df.loc[(df.loc[:,'immat'].isin(df_duree_cam1.loc[:,'immat']))]#on retrouve ces immatriculation mais qui ne sont pas à la 1ere camera
    cam1_croise_autre_cam=df_duree_cam1.reset_index().merge(df_duree_autres_cam.reset_index(), on='immat')#on fait une jointure entre cam 1 et les autres cam pour avoir une correspondance entre le passage devan la 1ere cmaera et la seconde
    cam1_croise_suivant=cam1_croise_autre_cam.loc[(cam1_croise_autre_cam.loc[:,'created_x']<cam1_croise_autre_cam.loc[:,'created_y'])]#on ne garde que les passages à la 2ème caméra postérieur au passage à la première
    cam1_fastest_next=cam1_croise_suivant.loc[cam1_croise_suivant.groupby(['immat'])['created_y'].idxmin()]#on isole le passage le plus rapide devant cam suivante pour chaque immatriculation
    if cam1_fastest_next.empty : 
        raise PasDePlError()
    cam1_puis_cam2=cam1_fastest_next.loc[cam1_fastest_next.loc[:,'camera_id_y']==camera_2].copy()#on ne garde que les passage le plus rapide devant la camera 2
    return cam1_puis_cam2         

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
    #faire tourner la clusterisation et recupérer le label (i.e l'identifiant cluster) et le nombre de cluster
    try :
        clustering=DBSCAN(eps=delai, min_samples=len(liste_valeur)/coeff).fit(liste)
    except ValueError :
        raise ClusterError()
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters_== 0 :
        raise ClusterError()
    #mettre en forme au format pandas
    results = pd.DataFrame(pd.DataFrame([donnees_src.index,labels]).T)
    results.columns = ['index_base', 'cluster_num']
    results = pd.merge(results,df_pl_ok, left_on='index_base', right_index=True )
    #obtenir un timedelta unique
    temp_parcours_max=results.loc[results.loc[:,'cluster_num']!=-1].groupby(['cluster_num'])['tps_parcours'].max().min()
    
    return n_clusters_, temp_parcours_max

def grouper_pl(df,date_debut,date_fin,camera,df_filtre):
    """
    Regroupement des PL par immat en fonction des attributs de dates de debuts et de fin et de la camera1 de l'objet, 
    avec filtre des passages deja present dans un autre trajet. les cameras et date de passages devant sont stockées dans des tuples
    en entree : 
        df_filtre : df des passages dejas vu dans des trajets
    en sortie : 
        groupe_pl : df des immats groupées  
        df_duree_cam1 : df des passages à la camera 1, TV 
        df_duree_autres_cam : df des passages des immats passées à la cameras 1 camera 1, TV 
    """
    #on limite le nb d'objet entre les 2 heures de depart
    df_duree=df.loc[date_debut:date_fin]
    if df_duree.empty : 
        raise PasDePlError() 
    if isinstance(df_filtre,pd.DataFrame) : 
        df_duree=filtrer_df(df_duree,df_filtre)
    #on trouve les veh passés cameras 1
    df_duree_cam1=df_duree.loc[df_duree.loc[:,'camera_id']==camera]
    if df_duree_cam1.empty : 
        raise PasDePlError() 
    #on recupere ces immat aux autres cameras
    df_duree_autres_cam=df.loc[(df.loc[:,'immat'].isin(df_duree_cam1.loc[:,'immat']))]
    groupe=(df_duree_autres_cam.sort_index().reset_index().groupby('immat').agg({'camera_id':lambda x : tuple(x), 
                                                                                 'created':lambda x: tuple(x)}))
    if groupe.empty :
        raise PasDePlError()
    
    return groupe, df_duree_cam1,df_duree_autres_cam

def filtre_et_forme_passage(camera,groupe_pl_init, liste_trajet, df_duree_cam1):
    """
    filtre des pl groupes selon une liste de trajets prédéfinie,
    traitements des données filtrées pour ajouts attributs o_d et tps de parcours
    en entree : 
        groupe_pl : issues de grouper_pl
        liste_trajet : df des trajets : attribut du présent module : liste_complete_trajet, liste_trajet_incomplet
        df_duree_cam1 : df des passages à la camera 1, TV, issu de grouper_pl
    en sortie : 
        df_agrege : df des trajets filtres avec les attributs qui vont bien
    """
    def filtrer_passage(liste, df_liste_trajet,cam) :
        """
        Récuperer les cameras qui correpondent à un trajet
        en entre : liste : tuple des cameras associée à une immat
                   df_liste_trajet : dataframe des trajets pssibles issus de liste_complete_trajet
        en sortie : liste des cameras retenues dans le trajet
        """
        for liste_cams in [a for a in df_liste_trajet.cameras.tolist() if a[0]==cam] :
            if liste[:len(liste_cams)]==tuple(liste_cams):
                return liste[:len(liste_cams)]
        else : return liste
    
    def recuperer_date_cam2(liste,liste_created,df_liste_trajet,cam):
        """
        Récuperer les horaires de passage des cameras qui correpondent à un trajet
        en entre : liste : tuple des cameras associée à une immat
                   liste_created : tuple des horodate associées à une immat
                   df_liste_trajet : dataframe des trajets pssibles issus de liste_complete_trajet
        en sortie : liste des horodates retenues dans le trajet
        """
        for liste_cams in [a for a in df_liste_trajet.cameras.tolist() if a[0]==cam] :
            if liste[:len(liste_cams)]==tuple(liste_cams):
                return liste_created[len(liste_cams)-1]
        else : return liste_created[-1]
    groupe_pl=groupe_pl_init.copy()
    groupe_pl['camera_id']=groupe_pl.apply(lambda x : filtrer_passage(x['camera_id'],liste_trajet,camera),axis=1)#on filtre les cameras selon la liste des trajets existants
    groupe_pl['created']=groupe_pl.apply(lambda x : recuperer_date_cam2(x['camera_id'],x['created'],liste_trajet,camera),axis=1)#on recupère les datetimede passages correspondants
    df_ts_trajets=(groupe_pl.reset_index().merge(liste_trajet[['cameras','origine','destination']],right_on='cameras', left_on='camera_id').
                   rename(columns={'created':'date_cam_2'}).drop('camera_id',axis=1))#on récupère les infos par jointure sur les cameras
    if df_ts_trajets.empty :
        raise PasDePlError()
    df_ts_trajets['o_d']=df_ts_trajets.apply(lambda x : x['origine']+'-'+x['destination'],axis=1)
    df_agrege=df_duree_cam1.reset_index().merge(df_ts_trajets,on='immat').drop(['camera_id','fiability'],axis=1).rename(columns={'l_y':'l','created':'date_cam_1'})
    df_agrege['tps_parcours']=df_agrege.apply(lambda x : x.date_cam_2-x.date_cam_1, axis=1)
    df_agrege=df_agrege.loc[df_agrege['date_cam_2'] > df_agrege['date_cam_1']]#pour les cas bizarres de plaques vu a qq minutes d'intervalle au sein d'une même heure
    
    if df_agrege.empty :
        raise PasDePlError()
    
    return df_agrege

def cam_voisines(immat, date, camera, df) :
    """
    Retrouver les dates et camera de passages d'un vehicule avant et apres un passage donne
    en entree : 
        immat : string : immatribualtion
        date : string : date de passage
        camera : cam de passage
        df : df contenant tous les passages de immats concernees (df 3 semianes ou extraction)
    en sortie : 
        cam_suivant : entier : camera suivante
        date_suivant : Pd.Timestamp ou string type YYYY-MM-DD associe à cam suivant
        cam_precedent : entier : camera precedente
        date_precedent : Pd.Timestamp ou string type YYYY-MM-DD associe à cam precedent
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


class trajet():
    """
    classe permettant le calcul de trajet : 
    - direct (2 cameras : debut et fin uniquement) � un horaire de depart donne. le veh passe cam 1 puis de suite cam2 
    - indirect (plus de 2 cameras : toute les cameras parcourue) � un horaire de depart donne. Le veh passe a chacune des cameras, dans l'ordre
    - global (2 cameras : debut et fin uniquement) � un horaire de depart donne. le veh part de cam1 et arrive a cam2, selon une liste d etrajet possible. cf fichier XXXXX
    Attributs : 
        df -- une pandas df issue de ouvrir_fichier_lapi ou de tout autre slice ou copy
        date_debut -- string de type YYYY-MM-DD hh:mm:ss ou pandas datetime -- date de debut du passage � la 1ere cam
        duree -- integer -- duree en minute pour periode de passage devant la premi�re camera
        typeTrajet -- type de trajet -- Direct, Indirect, Global. le nombre de Camera est li� et fonction de ce param�tre
        df_filtre -- pandas dataframe -- permetde filtrer les donn�es selonles passages d�j� traites. en typeTrajet='Global unqiuement'
        temps_max_autorise -- le temps que l'on autorise pour trouver les vehicules pass�s par cam1.
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
        
        #en fonction du df qui est pass� on met la date de creation en index ou non
        if isinstance(df.index,pd.DatetimeIndex) :
            self.df=df
        else :
            self.df=df.set_index('created').sort_index()
            
        #on filtre sur le type de vehicule �tudi�
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
            immat : immatriculation cry^pt�e
            date_cam_1 : pandas datetime : date de passage devant la premi�re camera
            date_cam_2 : pandas datetime : date de passage devant la deuxieme camera    
            tps_parcours : pandas timedelta : tempsde parcours entre les 2 cameras
        """
        #trouver tt les bagnoles pass�e par cam1 dont la 2eme camera est cam2
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
        pour obtenir un dico contenant les instances de trajet pour chaque trajet �le�mentaires
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
            if indice==0 : # si c'est le premier tarjet on se base sur des param�tres classiques
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
            immat : immatriculation cry^pt�e
            date_cam_1 : pandas datetime : date de passage devant la premi�re camera
            date_cam_2 : pandas datetime : date de passage devant la deuxieme camera    
            tps_parcours : pandas timedelta : tempsde parcours entre les 2 cameras
            cameras : liste des cameras parcourue
        """
        dico_rename={'date_cam_1_x':'date_cam_1','date_cam_2_y':'date_cam_2'} #param�tres pour mise en forme donnees
        
        #on fait une jointure de type 'inner, qui ne garde que les lignes pr�sentes dans les deux tables, en iterant sur chaque element du dico
        long_dico=len(self.dico_traj_directs)
        #print (self.dico_traj_directs)
        if long_dico<2 : #si c'est le cas �a veut dire une seule entree dans le dico, ce qui signifie que l'entree est empty, donc on retourne une df empty pour etre raccord avec le type de donnees renvoyee par trajet_direct dans ce cas
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

        df_transit['cameras']=df_transit.apply(lambda x :tuple(self.cameras_suivantes), axis=1)

        return df_transit
        
    def loc_trajet_global(self,df_filtre,liste_trajet): 
        """
        fonction de detection des trajets pour tous les destinations possibles de la camera 1 du constrcuteur.
        N�cessite l'utilisation de la variable module liste_complete_trajet qui contient tous les trajets possible en entree-sortie
        En sortie : 
            df_transit : pandas dataframe conteant les mm colonnes que pour direct et indirects
            df_passag_transit : pandas dataframe conteant les passages consid�r�s en transit
            
        """       
        groupe_pl,df_duree_cam1,df_duree_autres_cam=grouper_pl(self.df, self.date_debut, self.date_fin, self.cameras_suivantes[0],df_filtre)
        
        df_agrege=filtre_et_forme_passage(self.cameras_suivantes[0],groupe_pl, liste_trajet, df_duree_cam1)
        
        #pour obtenir la liste des passagesrelevant de trajets de transits :
        #limitation des donn�es des cameras par jointures
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
        Pour obtenir le graph des temps de parcours avec la ligne du temps de parcours max associ�
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
        


class ClusterError(Exception):
    """Excpetion si pb dans la construction du cluster
    Attributs : 
        message -- message d'erreur -- text
        nb_cluster -- nombre de cluster -- int 
    """       
    def __init__(self):
        Exception.__init__(self,"Erruer de cluster : nb cluster = 0 ou ValueError sur le DBScan ou df d'entree vide")

class PasDePlError(Exception):  
    """
    Exception levee si le trajet direct ne comprend pas de pl
    """     
    def __init__(self):
        Exception.__init__(self,'pas de PL sur la période et les cameras visées')

class TypeTrajet_NbCamera_Error(Exception):  
    """
    Exception levee si le type de trajet ne correpond pas au nombre de camera ou si le type e trajet n'est pas Direct, Indirect, Global.
    """     
    def __init__(self, nb_cam, typeTrajet):
        Exception.__init__(self,f"le nb de camera ({nb_cam}) ne correspond pas au type de trajet, ou le type : {typeTrajet} n'est pas connu")