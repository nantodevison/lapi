# -*- coding: utf-8 -*-
'''
Created on 27 fev. 2019
@author: martin.schoreisz

Module de traitement des donnees lapi

'''

import matplotlib #pour éviter le message d'erreurrelatif a rcParams
import pandas as pd
import numpy as np
import Connexion_Transfert as ct
import altair as alt
import os,math,re, datetime as dt
from sklearn.cluster import DBSCAN
from sklearn import svm
from statistics import mode, StatisticsError

dico_renommage={'created_x':'date_cam_1', 'created_y':'date_cam_2'}

def mise_en_forme_dfs_trajets (fichier, type):
    """
    mise en forme des dfs de liste de trajets possibles à partir des json contenus ici :
    Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python
    en entree :
        fichier : raw string : le chemin du fichier
        type : string : 'complet' ou 'incomplet' : le type de fichier de trajet
    en sortie : 
        df_liste_trajets : la df des trajets
    """
    df_liste_trajets=pd.read_json(fichier, orient='index')
    df_liste_trajets['cameras']=df_liste_trajets.apply(lambda x : tuple(x['cameras']),axis=1)
    if type=='complet': 
        df_liste_trajets['tps_parcours_theoriq']=df_liste_trajets.apply(lambda x : pd.Timedelta(milliseconds=x['tps_parcours_theoriq']),axis=1)
    elif type=='incomplet' :
        df_liste_trajets['tps_parcours_theoriq']=df_liste_trajets.apply(lambda x : pd.Timedelta(x['tps_parcours_theoriq']),axis=1)
    df_liste_trajets.sort_values('nb_cams', ascending=False, inplace=True)
    return df_liste_trajets

#attributs de liste des trajets
liste_complete_trajet=mise_en_forme_dfs_trajets(r'Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\trajets_possibles.json','complet')
liste_trajet_incomplet=mise_en_forme_dfs_trajets(r'Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\liste_trajet_incomplet.json','incomplet')
liste_trajet_rocade=pd.read_json(r'Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\liste_trajet_rocade.json', orient='index')
param_cluster=pd.read_json(r'Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\param_cluster.json', orient='index')
#fichier des plaques, en df
plaques_europ=pd.read_csv(r'Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\plaques_europ.txt', sep=" ", header=None, names=['pays','re_plaque'])
#matrices des nb de jours
matrice_nb_jo=pd.read_json(r'Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\nb_jours_mesures.json',orient='index').pivot(
    index='origine', columns='destination',values='nb_jo').replace('NC',np.NaN)
matrice_nb_jo_samedi=pd.read_json(r'Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\nb_jours_mesures.json',orient='index').pivot(
    index='origine', columns='destination',values='nb_jo_samedi').replace('NC',np.NaN)
matrice_nb_j_tot=pd.read_json(r'Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\nb_jours_mesures.json',orient='index').pivot(
    index='origine', columns='destination',values='nb_j_tot').replace('NC',np.NaN)


    
def ouvrir_fichier_lapi_final(date_debut, date_fin) : 
    """ouvrir les donnees lapi depuis la Bdd 'lapi' sur le serveur partage GTI
    l'ouvertur se fait par appel d'une connexionBdd Python (scripts de travail ici https://github.com/nantodevison/Outils/blob/master/Outils/Martin_Perso/Connexion_Transfert.py)
    en entree : date_debut : string de type YYYY-MM-DD hh:mm:ss
                date_fin: string de type YYYY-MM-DD hh:mm:ss
    en sortie : dataframe pandas
    """
    with ct.ConnexionBdd('gti_lapi_final') as c : 
        requete_passage=f"select case when camera_id=13 or camera_id=14 then 13 when camera_id=15 or camera_id=16 then 15 else camera_id end::integer as camera_id , created, immatriculation as immat, fiability, l, state from data.te_passage where created between '{date_debut}' and '{date_fin}'"
        df_passage=pd.read_sql_query(requete_passage, c.sqlAlchemyConn)
        requete_plaque=f"select plaque_ouverte, chiffree from data.te_plaque_courte"
        df_plaque=pd.read_sql_query(requete_plaque, c.sqlAlchemyConn)
        requete_immat=f"select immatriculation,pl_siv,pl_3barriere,pl_2barriere,pl_1barriere,pl_mmr75,vul_siv,vul_mmr75,vl_siv,vl_mmr75  from data.te_immatriculation"
        df_immat=pd.read_sql_query(requete_immat, c.sqlAlchemyConn)
        return df_passage,df_plaque, df_immat

def supprimer_doublons(df_passage):
    """
    Suppression des doublons exact entre les attributs created et immat et suppresion des doublons proches (inf a 10s) entre created, camera_id
    et immat, avec conservation de la ligne avec la fiability la plus haute
    en entree : 
        df_passage : df issue de l'import des données de la bdd
    en sortie : 
        df_3semaines : df avec les doublons supprimes
    """
    #supprimer les doublons
    df_3semaines=df_passage.reset_index().drop_duplicates(['created','immat'])
    #doublons "proches" : même immat, même camera, passages écartés de moins de 10s
    df_3semaines=df_3semaines.sort_values(['immat','created','camera_id','fiability']).copy()
    df_3semaines['id']=(df_3semaines.created - df_3semaines.created.shift(1) > pd.Timedelta(seconds=10)).fillna(99999999).cumsum(skipna=False)
    df_3semaines=df_3semaines.sort_values(['immat','id','fiability'], ascending=False).copy().drop_duplicates(['immat','id']).set_index('created')
    return df_3semaines

def passages_proches(df):
    """
    Trouver les passages trop proches à des cameras différentes, à appliquer sur un groupe de type de veh (PL, VL...)
    en entre : 
        df : df des données issues de la Bdd
    en sortie :
        groupe_pl : pl groupes par immat
        groupe_pl_rappro : PL trop proche, groupe par immat avec tuple des created, fiability, camera_id
        
    """
    #fonction de test d'ecart entre les passages
    def ecart_passage(liste_passage, liste_camera,state) : 
        for i in range(len(liste_passage)-1):
            if (pd.to_datetime(liste_passage[i+1])-pd.to_datetime(liste_passage[i])<pd.Timedelta('00:05:00') and 
                pd.to_datetime(liste_passage[i])!=pd.to_datetime(liste_passage[i+1]) and state!='!!') : #on trouve l'enchainement en moins de 5minutes, sans prendre les doublons
                return True
        else : return False
    def conserver_state(liste_state):
        if '!!' in liste_state : 
            return '!!'
        else : 
            try : #si erreur de statistique car element represente avec le mm nombre
                return mode(liste_state) #element le plus represente
            except StatisticsError : 
                return liste_state[0]
    def liste_passage(liste_cam, liste_created) : 
        liste_passage=[]
        for i in range(len(liste_created)-1):
            if pd.to_datetime(liste_created[i+1])-pd.to_datetime(liste_created[i])<pd.Timedelta('00:05:00') :
                liste_passage.append(liste_cam[i])
                liste_passage.append(liste_cam[i+1])
        return tuple(liste_passage)
    def liste_created(liste_cam, liste_created) : 
        liste_created_fin=[]
        for i in range(len(liste_created)-1):
            if pd.to_datetime(liste_created[i+1])-pd.to_datetime(liste_created[i])<pd.Timedelta('00:05:00') :
                liste_created_fin.append(liste_created[i])
                liste_created_fin.append(liste_created[i+1])
        return tuple(liste_created_fin)
    def liste_fiability(liste_fiab, liste_created) : 
        liste_fiab_fin=[]
        for i in range(len(liste_created)-1):
            if pd.to_datetime(liste_created[i+1])-pd.to_datetime(liste_created[i])<pd.Timedelta('00:05:00') :
                liste_fiab_fin.append(liste_fiab[i])
                liste_fiab_fin.append(liste_fiab[i+1])
        return tuple(liste_fiab_fin)
    
    
    #on grouep les données et modifie les colonnes
    groupe=(df.sort_index().reset_index().groupby('immat').agg({'camera_id':lambda x : tuple(x), 
                                                                  'created':lambda x: tuple(x),'state':lambda x : conserver_state(list(x)),
                                                                  'fiability':lambda x: tuple(x)}))
    #on ajoute une colonne drapeau pour localiser le pb
    groupe['erreur_tps_passage']=groupe.apply(lambda x :  ecart_passage(x['created'], x['camera_id'], x['state']),axis=1)
    #et on extrait unqiement les passages problemetaique
    groupe_pl_rappro=groupe[groupe['erreur_tps_passage']].copy()
    groupe_pl_rappro['liste_passag_faux']=groupe_pl_rappro.apply(lambda x : liste_passage(x['camera_id'],x['created']),axis=1)
    groupe_pl_rappro['liste_created_faux']=groupe_pl_rappro.apply(lambda x : liste_created(x['camera_id'],x['created']),axis=1)
    groupe_pl_rappro['fiability_faux']=groupe_pl_rappro.apply(lambda x : liste_fiability(x['fiability'],x['created']),axis=1)
    
    return groupe_pl_rappro, groupe

def recalage_cam10(df):
    """
    retarder de 25 minutes les passages à la caméra 10
    en entrée : 
        df : df des passages sans doublons
    en sortie : 
        df_recale : df avec l'attribut created modifie
    """
    df_recale=df.reset_index().copy()
    df_recale.loc[df_recale['camera_id']==10,'created']=df_recale.apply(lambda x : x['created']-pd.Timedelta('25min'),axis=1)
    df_recale=df_recale.set_index('created').sort_index()
    return df_recale

def filtre_plaque_non_valable(df, df_plaques):
    """
    Filtrer les plaques non valable d'une df
    en entree : 
        df : df des passages
        df_plaques : df des plaques non cryptee
    en sortie : 
        df_passages_filtre_plaque : df des passages sans les plaques non valide
        plaque_a_filtrer : dfdes plaques ouvertes qui ont été filtrees
    """
    def check_valid_plaque(plque_ouvert, liste_plaque_europ):
        """
        Marqueur si la plque correspond à un type de plaque attendu, cf Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\plaques_europ.txt
        """
        if not re.match('^([0-9])+$|^([A-Z])+$',plque_ouvert) :
            return any([re.match(retest,plque_ouvert) for retest in plaques_europ.re_plaque.tolist()])
        else : return False
    #jointure avec les plaques ouvertes
    df_passages_plaque_ouverte=df.reset_index().merge(df_plaques, left_on='immat', right_on='chiffree')
    df_passages_plaque_ouverte['plaque_valide']=df_passages_plaque_ouverte.apply(lambda x : check_valid_plaque(x['plaque_ouverte'],plaques_europ),axis=1)
    #plaques à filtrer
    plaque_a_filtrer=(df_passages_plaque_ouverte.loc[(~df_passages_plaque_ouverte['plaque_valide'])].plaque_ouverte.
                      value_counts().reset_index().rename(columns={'index':'plaque_ouverte','plaque_ouverte':'nb_occurence'}))
    #filtre des plques non desirees
    df_passages_filtre_plaque=df_passages_plaque_ouverte.loc[~df_passages_plaque_ouverte.plaque_ouverte.isin(plaque_a_filtrer.plaque_ouverte.tolist())]
    df_passages_filtre_plaque=df_passages_filtre_plaque.set_index('created').sort_index()
    return df_passages_filtre_plaque, plaque_a_filtrer

def affecter_type(df_passage,df_immat ):
    """
    affecter le type de vehicule dans le df des passage selon les infos fournies dans df_immat
    en entree : 
       df_passage : df des passages issues de ouvrir_fichier_lapi_final
       df_immat : df des immatriculations issues de ouvrir_fichier_lapi_final
    en sortie : 
        df_passage : df des passages avec l'attribut 'l' modifié
    """
    #definir le type de veh dans df_immat
    def type_veh(pl_siv, pl_3barriere, pl_2barriere,pl_1barriere,pl_mmr75,vul_siv,vul_mmr75,vl_siv,vl_mmr75):
        if (pl_siv+pl_3barriere+pl_2barriere+pl_1barriere>0) or (pl_mmr75>0 and (vul_mmr75+vl_mmr75)==0) : 
            return 1
        elif (vl_siv>0) or (vl_mmr75>0 and (vul_mmr75+pl_mmr75)==0) :
            return 0
        elif vul_siv>0 or (vul_mmr75>0 and (vl_mmr75+pl_mmr75)==0) :
            return 2
        else :
            return -1
    #affecter selon l'immatriculation uniquement
    df_immat['type_veh']=df_immat.apply(lambda x : type_veh(x['pl_siv'], x['pl_3barriere'], x['pl_2barriere'],x['pl_1barriere'],
                                                            x['pl_mmr75'],x['vul_siv'],x['vul_mmr75'],x['vl_siv'],x['vl_mmr75']),axis=1)
    df_passage=df_passage.reset_index().merge(df_immat[['immatriculation','type_veh']], left_on='immat', right_on='immatriculation', how='left')
    df_passage['l']=df_passage['type_veh']
    df_passage=df_passage.set_index('created').sort_index()
    df_passage.drop(['type_veh','immatriculation'],axis=1,inplace=True)
    return df_passage

def affecter_type_nuit(df_passages_affectes, df_immat):
    """
    affecter le type à des immats vue de nuit, plque etrangere, sur un trajetde transit, non vu avant, fiabilite ok
    en entre : 
        df_passages_affectes : passages issus de affecter_type
        df_immat : df des immatriculations issues de ouvrir_fichier_lapi_final
    en sortie : 
        df_passages_affectes : df_passages_affectes avec attribut l modifié
    """
    #nb de passages avec un type inconnu
    df=df_passages_affectes.copy()
    passages_type_inconnu=df.loc[df['l']==-1].reset_index()
    #passages inconnu avec une fiabilite superieure a 75 sur cam autre que 1ou2 et fiab > 35 sr cam 1 et 2
    passages_type_inconnu_fiab_sup75=(passages_type_inconnu.loc[((passages_type_inconnu['fiability']>75) & (~passages_type_inconnu['camera_id'].isin([1,2]))) |
                                                            ((passages_type_inconnu['fiability']>35) & (passages_type_inconnu['camera_id'].isin([1,2])))])
    #grouper les immat, mettre les attributs en tuple ou set
    groupe=(passages_type_inconnu_fiab_sup75.set_index('created').sort_index().reset_index().groupby('immat').agg({'camera_id':lambda x : tuple(x), 
                                                                                 'created':lambda x: tuple(x), 
                                                                                 'state': lambda x : set(tuple(x))}))
    #filtrer selon les horaires compris entre 19h et 6h (attention biais possible sur journee différentes), le type de trajet, un seul pays, pas francçais
    groupe_filtre=groupe.loc[groupe.apply(lambda x: (all((pd.to_datetime(e).hour>19 or pd.to_datetime(e).hour<7) for e in x['created'])) &
                        (x['camera_id'] in liste_complete_trajet.cameras.tolist()) & (len(x['state'])==1) & 
                        (x['state']!=set(['FR',])) ,axis=1)].copy()
    #filtrer les pays non connus
    groupe_filtre=groupe_filtre.loc[~groupe_filtre.apply(lambda x: [(a) for a in x['state']]==['  '],axis=1)]
    #modifier la valeur de l
    df.loc[df_passages_affectes['immat'].isin(groupe_filtre.index.tolist()),'l']=1
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
                
def trouver_passages_consecutif(df, date_debut, date_fin, camera_1, camera_2) : 
    """
    pour obtenir une df des immat passées par une camera puis de suite une autre
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

def creer_liste_date(date_debut, nb_jours):
    """
    générer une liste de date à parcourir
    en entree : 
       date_debut : string : de type YYYY-MM-DD hh:mm:ss
        nb_jours : integer : nb de jours considéré (mini 1)
    en sortie : 
        liste_date : liste de liste de [date, ecart_temps] au format pd.timestamps et integer en minutes
    """
    liste_date=[]
    for date in pd.date_range(date_debut, periods=nb_jours*24, freq='H') : 
        if date.hour in [6,7,8,14,15,16,17,18,19] : 
            for date_15m in pd.date_range(date, periods=4, freq='15T') :
                liste_date.append([date_15m,15])
        else: 
            liste_date.append([date,60])
    return liste_date

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

def trajet_non_transit(df_transit, df_passage):
    """
    obtnir les immat groupees avec camera de passage et temps de passage des passages non concernes par le transit
    en entree : 
        df_transit : df des trajets de transit valides
        df_passage : df de tous les passages
    en sortie : 
        passages_non_transit : df des passages non transit regroupes par immat avec date de passage et camera_id dans des tuples tries par date
    """
    # trouver les passages correspondants aux trajets
    passages_transit=trajet2passage(df_transit,df_passage)
    #trouver les passages non compris dans passages transit
    passages_non_transit=df_passage.loc[
        ~df_passage.reset_index().set_index(['created', 'camera_id','immat']).index.isin(
        passages_transit.set_index(['created', 'camera_id','immat']).index.tolist())]
    return passages_non_transit.reset_index().sort_values('created').groupby('plaque_ouverte').agg(
        {'camera_id':lambda x : tuple(x),
         'created':lambda x: tuple(x)})
      
def transit_temps_complet(date_debut, nb_jours, df_3semaines,liste_trajet_loc=liste_complete_trajet):
    """
    Calcul des trajets et passages des poids lourds en transit, sur une période de temps minimale d'1j (peut etre regle et affiné dans la fonction selon date_debut, nb_jours et periode)
    en entree : 
        date_debut : string : de type YYYY-MM-DD hh:mm:ss
        nb_jours : integer : nb de jours considéré (mini 1)
        df_3semaines : pandas dataframe : issu de la fonction ouvrir_fichier_lapi nettoyée (suppr doublons et autres)
    en sortie :
        dico_od : pandas dataframe : liste des trajet de transit (cf Classe trajet, fonction loc_trajet_gloabl)
        dico_passag : pandas dataframe : liste des passgaes liés au trajet de transit (cf Classe trajet, fonction loc_trajet_gloabl)
    """


    for date, duree in creer_liste_date(date_debut, nb_jours) :
        if date.weekday()==5 : # si on est le semadi on laisse la journee de dimanche passer et le pl repart
            df_journee=df_3semaines.loc[date:date+pd.Timedelta(hours=32)]
        else : 
            df_journee=df_3semaines.loc[date:date+pd.Timedelta(hours=18)]
        if date.hour==0 : print(f"date : {date} debut_traitement : {dt.datetime.now()}")
        for cameras in zip([15,12,8,10,19,6],range(6)) : #dans ce mode peu importe la camera d'arrivée, elle sont toutes analysées
            try : 
                if 'dico_passag' in locals() : #si la varible existe deja on utilise pour filtrer le df_journee en enlevant les passages dejà pris dans une o_d (sinon double compte ente A63 - A10 et A660 -A10 par exemple 
                    donnees_trajet=trajet(df_journee,date,duree,cameras, typeTrajet='Global',df_filtre=dico_passag.loc[dico_passag['created']>=date].copy(),
                                          liste_trajet=liste_trajet_loc)
                else : 
                    donnees_trajet=trajet(df_journee,date,duree,cameras, typeTrajet='Global',liste_trajet=liste_trajet_loc)
                df_trajet, df_passag, df_tps_max=donnees_trajet.df_transit, donnees_trajet.df_passag_transit, donnees_trajet.temps_parcours_max
                
            except PasDePlError :
                continue
            
            if 'dico_passag' in locals() : #si la varible existe deja on la concatene avec le reste
                dico_passag=pd.concat([dico_passag,df_passag], sort=False)
                dico_od=pd.concat([dico_od,df_trajet], sort=False)
                dico_tps_max=pd.concat([dico_tps_max,df_tps_max], sort=False)
            else : #sinon on initilise cette variable
                dico_passag=df_passag
                dico_od=df_trajet 
                dico_tps_max=df_tps_max
    dico_tps_max=pd.DataFrame(dico_tps_max)
    return dico_od,  dico_passag, dico_tps_max

def param_trajet_incomplet(date_debut,df_od_corrige,df_3semaines,dico_passag):
    """
    Récupérer les paramètres necessaires à la fonction transit_trajet_incomplet
    en entree : 
        dico_passag : df des passages de transit issu des précédentes fonctions
        date_debut : string : de type YYYY-MM-DD hh:mm:ss : INUTILISE
        df_od_corrige df des trajet de  transit issu des précédentes fonctions
        df_3semaines : df de base de tous les passages
    en sortie 
        df_filtre_A63 : df 
        df_non_transit : df des passages qui ne sont pas dans dico_passag
        df_passage_transit : df des passages qui ne sont pas dans dico_passag mais qui ont une immat dans dico passage
    """
    #detreminer les passages non_transit (ne peut pas etre fait dans la fonction loc_trajet_global) 
    df_non_transit=df_3semaines.loc[(~df_3semaines.reset_index().set_index(['created','camera_id','immat']).index.isin(
                                dico_passag.set_index(['created','camera_id','immat']).index.tolist()))]
    
    #grouper les passage transit, associer le nombre de fois où ils ont passés puis ne conserver que ceux qui sont passé au moins 2 fois
    df_transit_nb_passage=df_od_corrige.groupby(['immat','o_d'])['l'].count().reset_index().rename(columns={'l':'Nb_occ'})
    df_immat_transit_nb_passage_sup2=df_transit_nb_passage.loc[df_transit_nb_passage['Nb_occ']>=2]
    
    #df des passages qui n'ont pas été identiiés comme transit, mais qui ont une immat qui a déjà fait du transit
    df_passage_transit=df_non_transit.loc[(df_non_transit.immat.isin(dico_passag.immat.unique()))]
    
    #identifier les doucblons : tel que présente le fichier présente bcp d'immat en double avec par exempele les o_d A660-N10 puis N10-A660.
    #or tout les trajets finissant par A660 ou A63 sont déja traites plus haut, donc on les vire
    df_filtre_A63=df_immat_transit_nb_passage_sup2.loc[df_immat_transit_nb_passage_sup2.apply(lambda x : x['o_d'].split('-')[1] not in ['A660','A63'],axis=1)]
    
    return df_filtre_A63, df_passage_transit, df_non_transit
    

def transit_trajet_incomplet(df_filtre_A63,df_passage_transit,date_debut,nb_jours, df_3semaines,liste_trajet_loc=liste_trajet_incomplet):
    """
    Extrapoler des trajest à partir des immats en transit,sur des trajets où il manque la camera de fin
    en entree : 
        df_filtre_A63 : df des immat de transit qui ne sont pas passées par A63. issu de param_trajet_incomplet()
        df_passage_transit : df des passages d'immatricluation identifiées en transit. issu de param_trajet_incomplet()
        date_debut : string : de type YYYY-MM-DD hh:mm:ss
        nb_jours : integer : nb de jours que l'on souhiate etudie depuis la date_debut
        df_3semaines : df des passages totaux filtres (cf supprimer_doublons() )
        liste_trajet_loc : df de filtre selon les trajets prévus, cf mise_en_forme_dfs_trajets
    """
    df_passage_transit_incomplet=None
    dico_od=None
    for date, duree in creer_liste_date(date_debut, nb_jours) :
        if date.hour==0 : print(f"date : {date} debut_traitement : {dt.datetime.now()}")
        date_fin=date + pd.Timedelta(minutes=duree)
        for cameras in [15,12,8,10,6,4,1,2,3,5] :
            #regrouper les pl
            try : 
                groupe_pl,df_duree_cam1,df_duree_autres_cam=grouper_pl(df_passage_transit
                                                        , date, date_fin, cameras, df_passage_transit_incomplet)
            except PasDePlError :
                continue 
            #le pb c'est qu epour le trajet qui s'arrete sur Rocade Ouest, le PL est susceptible d'aller soit sur A89 soit sur N10.
            #on doit donc faire une jointure sur l'immat pour connaitre ses o_d possibles
            #1. Trouver les immat de groupe PL concernées par du transit
            groupe_pl_transit=groupe_pl.merge(df_filtre_A63, left_index=True, right_on='immat').rename(columns={'o_d':'o_d_immat'})
            if groupe_pl_transit.empty : 
                continue
            #2.filtrer selon les trajets possibles
            try : 
                trajets_possibles=(filtre_et_forme_passage(cameras,groupe_pl_transit, liste_trajet_loc, df_duree_cam1).rename(columns={'o_d':'o_d_liste_trajet'}))
            except PasDePlError :
                continue 
            #3. ajouter les infos sur les cameras avant / apres le passage final
            cam_proches=trajets_possibles.apply(lambda x : cam_voisines(x['immat'], x['date_cam_2'],x['cameras'][-1], df_3semaines),axis=1, result_type='expand')
            cam_proches.columns=['cam_suivant','date_suivant','cam_precedent','date_precedent']
            cam_proches.drop(['cam_precedent','date_precedent'], axis=1, inplace=True)
            trajets_possible_enrichi=pd.concat([trajets_possibles,cam_proches],axis=1)
            #4.filtrer selon un critère de camera suivante qui est une des entrée du dispositif Lapi
            dico_filtre = {'destination':[15,6,8,10,12]}
            trajet_transit_incomplet=trajets_possible_enrichi.loc[trajets_possible_enrichi.apply(lambda x : (x['cam_suivant'] in dico_filtre['destination']) & (
                                                                        x['o_d_immat']==x['o_d_liste_trajet']),axis=1)].copy()
            trajet_transit_incomplet.rename(columns={'o_d_liste_trajet':'o_d'}, inplace=True)
            #POUR TEST !!!!!
            #trajet_transit_incomplet=trajets_possible_enrichi
            if trajet_transit_incomplet.empty : 
                continue
            #extrapolation des passages
            df_joint_passag_transit=trajets_possibles.merge(df_duree_autres_cam.reset_index(), on='immat')
            df_passag_transit1=df_joint_passag_transit.loc[(df_joint_passag_transit.apply(lambda x : x['camera_id'] in x['cameras'], axis=1))]
            df_passag_transit=(df_passag_transit1.loc[df_passag_transit1.apply(lambda x : x['date_cam_1']<=x['created']<=x['date_cam_2'], axis=1)]
                            [['created','camera_id','immat','fiability','l_y','state_x']].rename(columns={'l_y':'l','state_x':'state'}))
            #ajoute les passages concernes au dico_passage qui sert de filtre

            df_passage_transit_incomplet=pd.concat([df_passage_transit_incomplet,df_passag_transit])
            dico_od=pd.concat([dico_od,trajet_transit_incomplet], sort=False)
            
    return dico_od,df_passage_transit_incomplet



def pourcentage_pl_camera(df_pl,dico_passag, df_vl):
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
    df_synthese_vl_tot=df_vl.groupby('camera_id').resample('H').count()['immat'].reset_index().rename(columns={'immat':'nb_veh'})
    df_synthese_pl_transit=dico_passag.set_index('created').groupby('camera_id').resample('H').count()['immat'].reset_index().rename(
            columns={'immat':'nb_veh'})
    
    df_synthese_pl_tot['type']='PL total'
    df_synthese_vl_tot['type']='VL total'
    df_synthese_pl_transit['type']='PL transit'
    
    df_pct_pl_transit=df_synthese_pl_tot.merge(df_synthese_pl_transit, on=['camera_id','created']).rename(columns={'nb_veh_x':'nb_pl_tot',
                                                                                            'nb_veh_y':'nb_pl_transit'})
    print(f'pl_tot ={len(df_synthese_pl_tot)} , vl_tot={len(df_synthese_vl_tot)} , pl_transit={len(df_synthese_pl_transit)},pl_tot-joint-transit={len(df_pct_pl_transit)}')
    df_pct_pl_transit['pct_pl_transit']=df_pct_pl_transit.apply(lambda x : pct_pl(x['nb_pl_transit'],x['nb_pl_tot']) ,axis=1) 
    
    concat_tv=pd.concat([df_synthese_pl_tot,df_synthese_vl_tot,df_synthese_pl_transit], axis=0, sort=False).rename(columns={'0':'nb_veh'})
    jointure_pct_pl=concat_tv.merge(df_pct_pl_transit, on=['camera_id','created'], how='left')[['camera_id','created','nb_veh','type','pct_pl_transit']]
    #[['camera_id','created','pct_pl_transit']]
    return jointure_pct_pl
    
def filtrer_df(df_global,df_filtre): 
    df_global=df_global.reset_index().set_index(['created','immat'])
    df_filtre=df_filtre.reset_index().set_index(['created','immat'])
    df_global_filtre=df_global.loc[~df_global.index.isin(df_filtre.index)].reset_index().set_index('created')
    return df_global_filtre
    
def jointure_temps_reel_theorique(df_transit, df_tps_parcours, df_theorique,typeTrajet='complet'):
    """
    Création du temps de parcours et affectation d'un attribut drapeau pour identifier le trafci de transit
    en entree : 
        df_transit : df des o_d issu de transit_temps_complet
        df_tps_parcours : df des temps de parcours issu du lapi df_tps_parcours (transit_temps_complet)
        df_theorique : liste des trajets possibles etdes temps theoriques associés : liste_complete_trajet
        typeTrajet : string : si le trajet est issue de cameras de debut et fin connuen ou d'une camera de fin extrapolee. 
    en sortie : 
        df_transit_tps_parcours : df des o_d complété par un attribut drapeau sur le transit, et les temps de parcours, et le type de temps de parcours
    """
        
    def temps_pour_filtre(date_passage,tps_parcours, type_tps_lapi, tps_lapi, tps_theoriq):
        """pour ajouter un attribut du temps de parcours qui sert à filtrer les trajets de transit"""
        marge = 660 if date_passage.hour in [19,20,21,22,23,0,1,2,3,4,5,6] else 0  #si le gars passe la nuit, on lui ajoute 11 heure de marge
        if type_tps_lapi in ['Cluster','moyenne Cluster','predit']:
            return tps_lapi+pd.Timedelta(str(marge)+'min')
        else : 
            return tps_theoriq+pd.Timedelta(str(marge)+'min')   
        
    def periode_carac(date_passage) :
        """
        pour calculer la période de passage selon une date
        """
        if date_passage.hour in [6,7,8,14,15,16,17,18,19] : 
            return date_passage.floor('15min').to_period('15min')
        else : 
            return date_passage.to_period('H')
    
    if df_transit.empty :
        raise PasDePlError()
    df_transit=df_transit.copy()        
    df_transit['period']=df_transit.apply(lambda x : periode_carac(x['date_cam_1']),axis=1)
    df_tps_parcours['period']=df_tps_parcours.apply(lambda x : periode_carac(x['date']),axis=1)
    if typeTrajet=='complet' : #dans ce cas la jointure avec les temps theorique ne se fait que sur les cameras
        df_transit_tps_parcours=df_transit.merge(df_tps_parcours, on=['o_d','period'],how='left').merge(df_theorique[['cameras','tps_parcours_theoriq' ]], 
                                                                                                    on='cameras')
    else : #dans ce cas la jointure avec les temps theorique ne se sur les cameras et l'od, car doublons de cameras pour certains trajet (possibilité d'aller vers A89 ou N10 apres Rocade)
        df_transit_tps_parcours=df_transit.merge(df_tps_parcours, on=['o_d','period'],how='left').merge(df_theorique[['cameras','tps_parcours_theoriq','o_d']], 
                                                                                                    on=['cameras','o_d'])
        df_transit_tps_parcours['type']=df_transit_tps_parcours.type.fillna('85eme_percentile')
        df_transit_tps_parcours['temps']=df_transit_tps_parcours.temps.fillna(df_transit_tps_parcours['tps_parcours_theoriq'])
        df_transit_tps_parcours['date']=df_transit_tps_parcours.apply(lambda x : x['period'].to_timestamp(), axis=1)
    df_transit_tps_parcours['temps_filtre']=df_transit_tps_parcours.apply(lambda x : temps_pour_filtre(x['date_cam_1'],
                                                                    x['tps_parcours'], x['type'], x['temps'], x['tps_parcours_theoriq']), axis=1)
    
    return df_transit_tps_parcours


def identifier_transit(df_transit_temps, marge,nom_attribut_temps_filtre='temps_filtre',nom_attribut_tps_parcours='tps_parcours'): 
    """
    affecter un attribut drapeau d'identification du trafic de trabsit, selon une marge
    en entree : 
        df_transit_temps : df issue de jointure_temps_reel_theorique
        marge : integer: marge possible entre le temps theorique ou lapi et le temsp de passage. comme les camions doivent faire une pause de 45min toute les 4h30...
    en sortie : 
        df_transit_temps : df avec la'ttribut filtre_tps identifiant le trafic de trabsit (1) ou non (0)
    """
    def filtre_tps_parcours(temps_filtre,tps_parcours, marge) : 
        """pour ajouter un attribut drapeau sur le tempsde parcours, et ne conserver que les trajets de transit"""
        if tps_parcours <= temps_filtre+pd.Timedelta(str(marge)+'min') :
            return 1
        else: 
            return 0
        
    df_transit_temps_final=df_transit_temps.copy()
    df_transit_temps_final['filtre_tps']=df_transit_temps_final.apply(lambda x : filtre_tps_parcours(x[nom_attribut_temps_filtre],
                                                                    x[nom_attribut_tps_parcours],marge), axis=1)
    return df_transit_temps_final
        
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

def verif_doublons_trajet(dico_od, destination):
    """
    fonction de vérification qu'un passage contenu dans un trajet n'est pas contenu dans un autre
    en entrée : dico_od : le dico des passagesde transit issu de la fonction  transit_temps_complet
                destination : string : destination du trajet (parmi les destination possibles dans liste_complete_trajet
    en sortie : une dataframe avec les passages en doublons
    """
    df_depart=dico_od.loc[dico_od['destination']==destination].copy()
    jointure=df_depart.merge(dico_od, on='immat')
    jointure=jointure.loc[jointure.date_cam_1_x!=jointure.date_cam_1_y].copy()
    df_doublons=(jointure.loc[((jointure.date_cam_1_y>=jointure.date_cam_1_x) & (jointure.date_cam_1_y<=jointure.date_cam_2_x)) |
                  ((jointure.date_cam_2_y>=jointure.date_cam_1_x) & (jointure.date_cam_2_y<=jointure.date_cam_2_x))])
    return df_doublons
    
def cam_voisines(immat, date, camera, df) :
    """
    Retrouver les dates et camera de passages d'un vehicule avant et apres un passage donne
    en entree : 
        immat : string : immatribualtion
        date : string : date de passage
        camera : cam de passage
        df : df contenant tous les passages de immats concernees (df 3 semianes ou extraction)
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

def correction_trajet(df_3semaines, dico_od, voie_ref='A660', cam_ref_1=13, cam_ref_2=15, cam_ref_3=19) : 
    """
    Fonction qui va ré-assigner les origines-destinations à A63 si certanes conditions sont remplie : 
    cas 1 : vue a A660 puis dans l'autre sens sur A63, ou inversement
    cas 2 : vue sur A660 Nord-Sud, puis A660 Sud-Nord, avec plus de 1jd'écart entre les deux
    en entree : 
        df_3semaines : dataframe des passages
        dico_od : dataframe des o_d issue de transit_temps_complet
        voie_ref : string : nom de la voie que l'on souhaite changer
        cam_ref_1 : integer : camera de a changer pour le sens 1 (cas 1)
        cam_ref_2 : integer : camera de a changer pour le sens 2 (cas 1)
        cam_ref_3 : integer camera a changer dans les deux sens (cas2)
    en sortie : 
        dico_od_origine : dataframe des o_d issue de transit_temps_complet complétée et modifée
    """
    
    def MaJ_o_d(correctionType, o, d):
        """
        Fonction de mise à jour des o_d pour les trajets concernants A660 que l'on rabat sur A63
        """
        if correctionType : 
            if o=='A660' : 
                new_o, new_d, od='A63',d,'A63-'+d
            elif o!='A63': 
                new_o, new_d, od=o,'A63',o+'-A63'
            else : 
                new_o, new_d, od=o,d,o+'-'+d
        else : 
            new_o, new_d, od=o,d,o+'-'+d 
        return new_o, new_d, od
        
    #cas 1 : passer sur A660 et vu avant ou apres sur A63
    dico_od_origine=dico_od.copy()
    dico_od_copie=dico_od.loc[(dico_od['origine']==voie_ref) | (dico_od['destination']==voie_ref)].reset_index().copy() #isoler les o_d liées au points en question
    df_immats=df_3semaines.loc[df_3semaines.immat.isin(dico_od_copie.immat.unique().tolist())] #limiter le df_3semaines aux immats concernée   df_adj=dico_od_copie.apply(lambda x : t.cam_adjacente(x['immat'],x['date_cam_1'],x['date_cam_2'],x['o_d'],df_immats),axis=1, result_type='expand') #construire les colonnes de camera adjacente et de temps adjacent 
    #on travaille le trajet : date_cam_1 est relatif à l'origine. si l'origine est A660, alors ce qui nous interesse est le passage précédent
    df_adj_cam1=dico_od_copie.apply(lambda x : cam_voisines(x['immat'],x['date_cam_1'],x['cameras'][0],df_immats),axis=1, result_type='expand')  
    df_adj_cam1.columns=['cam_suivant','date_suivant','cam_precedent1','date_precedent1']
    df_adj_cam1.drop(['cam_suivant','date_suivant'], axis=1, inplace=True)
    #inversement : date_cam_2 est relatif à la destination. si la destination est A660, alors ce qui nous interesse est le passage suivant
    df_adj_cam2=dico_od_copie.apply(lambda x : cam_voisines(x['immat'],x['date_cam_2'],x['cameras'][-1],df_immats),axis=1, result_type='expand') #construire les colonnes de camera adjacente et de temps adjacent 
    df_adj_cam2.columns=['cam_suivant2','date_suivant2','cam_precedent','date_precedent']
    df_adj_cam2.drop(['cam_precedent','date_precedent'], axis=1, inplace=True)
    dico_od_copie_adj=pd.concat([dico_od_copie,df_adj_cam1,df_adj_cam2],axis=1)
    #on creer une df de correction 
    dico_od_a_corrige_s_n=dico_od_copie_adj.loc[(dico_od_copie_adj['origine']==voie_ref) & (dico_od_copie_adj['cam_precedent1']==cam_ref_1)].copy()#recherche des lignes pour lesquelles origine=A660 et camera adjacente = 13 ou destination=A660 et et camera_adjacente = 15
    dico_od_a_corrige_s_n['temps_passage']=dico_od_a_corrige_s_n['date_cam_1']-dico_od_a_corrige_s_n['date_precedent1']#calcul du timedelta
    dico_od_a_corrige_n_s=dico_od_copie_adj.loc[(dico_od_copie_adj['destination']==voie_ref) & (dico_od_copie_adj['cam_suivant2']==cam_ref_2)].copy()
    dico_od_a_corrige_n_s['temps_passage']=dico_od_a_corrige_n_s['date_suivant2']-dico_od_a_corrige_n_s['date_cam_2']
    dico_temp=pd.concat([dico_od_a_corrige_n_s,dico_od_a_corrige_s_n])
    dico_correction=dico_temp.loc[~dico_temp.temps_passage.isna()]#on ne conserve que les ligne qui ont un timedelta !=NaT 
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','cameras']).index.isin(dico_correction.set_index(['date_cam_1','immat','cameras']).index),
                       'correction_o_d']=True #mise à jour de l'attribut drapeau
    dico_od_origine.loc[(dico_od_origine['correction_o_d']) &
                                          (dico_od_origine['correction_o_d_type'].isna()),'correction_o_d_type']='correction_A63'
    #mise à jour des  3 attributs liées aux o_d
    dico_od_origine.loc[dico_od_origine['correction_o_d'],'origine']=dico_od_origine.loc[dico_od_origine['correction_o_d']].apply(lambda x : MaJ_o_d(x['correction_o_d'], x['origine'],x['destination'])[0],axis=1)
    dico_od_origine.loc[dico_od_origine['correction_o_d'],'destination']=dico_od_origine.loc[dico_od_origine['correction_o_d']].apply(lambda x : MaJ_o_d(x['correction_o_d'], x['origine'],x['destination'])[1],axis=1)
    dico_od_origine.loc[dico_od_origine['correction_o_d'],'o_d']=dico_od_origine.loc[dico_od_origine['correction_o_d']].apply(lambda x : MaJ_o_d(x['correction_o_d'], x['origine'],x['destination'])[2],axis=1) 
    
    #cas 2 : passer sur A660 Nord-Sud puis Sud-Nord avec au moins 1 jour d'écart
    dico_od_copie=dico_od_origine.loc[(dico_od_origine['destination']=='A660')].reset_index().copy()
    
    df_adj_cam2=dico_od_copie.apply(lambda x : cam_voisines(x['immat'],x['date_cam_2'],x['cameras'][-1],df_immats),axis=1, result_type='expand') #construire les colonnes de camera adjacente et de temps adjacent 
    df_adj_cam2.columns=['cam_suivant','date_suivant','cam_precedent','date_precedent']
    df_adj_cam2.drop(['cam_precedent','date_precedent'], axis=1, inplace=True)
    dico_od_copie_adj=pd.concat([dico_od_copie,df_adj_cam1,df_adj_cam2],axis=1)
    dico_od_a_corrige=dico_od_copie_adj.loc[dico_od_copie_adj['cam_suivant']==cam_ref_3].copy()#filtrer les résultats sur la cameras de fin
    dico_od_a_corrige['temps_passage']=dico_od_a_corrige['date_suivant']-dico_od_a_corrige['date_cam_2']#calcul du temps de passages entre les cameras
    dico_filtre=dico_od_a_corrige.loc[dico_od_a_corrige['temps_passage']>=pd.Timedelta('1 days')]
    
    #pour les lignes ayant 1 temps de passage sup à 1 jour, on va réaffecter d à A63
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','date_cam_2']).index.isin(
    dico_filtre.set_index(['date_cam_1','immat','date_cam_2']).index.tolist()),'destination']='A63'
    #on modifie o_d aussi
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','date_cam_2']).index.isin(
    dico_filtre.set_index(['date_cam_1','immat','date_cam_2']).index.tolist()),'o_d']=(dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','date_cam_2']).index.isin(
    dico_filtre.set_index(['date_cam_1','immat','date_cam_2']).index.tolist())].apply(lambda x : x['origine']+'-'+x['destination'], axis=1))
    #puis on met à jour correction_o_d
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','date_cam_2']).index.isin(
    dico_filtre.set_index(['date_cam_1','immat','date_cam_2']).index.tolist()),'correction_o_d']=True
    #pour les lignes ayant 1 temps de passage sup à 1 jour, on va réaffecter o à A63
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat']).index.isin(
    dico_filtre.set_index(['date_suivant','immat']).index.tolist()),'origine']='A63'
    #on modifie o_d aussi
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat']).index.isin(
    dico_filtre.set_index(['date_suivant','immat']).index.tolist()),'o_d']=(dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat']).index.isin(
    dico_filtre.set_index(['date_suivant','immat']).index.tolist())].apply(lambda x : x['origine']+'-'+x['destination'], axis=1))
    #puis on met à jour correction_o_d
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat']).index.isin(
    dico_filtre.set_index(['date_suivant','immat']).index.tolist()),'correction_o_d']=True

    return dico_od_origine

def corriger_df_tps_parcours (dico_tps_max):
    """ fonction de correction de la df_tps_parcours issue de transit_temps_complet.
    On moyenne les valuers de temps de parcours de type '85_percentile' si encadrer par des Cluster
    en entree : 
        dico_tps_max issu de transit_temps_complet
    en sortie :
        dico_tps_max modifiée
    """
    def moyenne_tps_85pct(type_la, type_avant, type_apres, tps,tpsla) : 
        if type_la=='85eme_percentile' and type_avant=='Cluster' and type_apres=='Cluster' : 
            return tps,'moyenne Cluster'
        else : return tpsla,type_la
    
    dico_tps_max2=dico_tps_max.reset_index().drop('index',axis=1).sort_values(['o_d','date']).copy()
    dico_tps_max2['tps']=(dico_tps_max2.temps.shift(1)+dico_tps_max2.temps.shift(-1)) / 2
    dico_tps_max2['type_tps_1']=dico_tps_max2.type.shift(1)
    dico_tps_max2['type_tps_2']=dico_tps_max2.type.shift(-1)
    dico_tps_max2['temps']=dico_tps_max2.apply(lambda x : moyenne_tps_85pct(x['type'], x['type_tps_1'],x['type_tps_2'], x['tps'], x['temps'])[0],axis=1)
    dico_tps_max2['type']=dico_tps_max2.apply(lambda x : moyenne_tps_85pct(x['type'], x['type_tps_1'],x['type_tps_2'], x['tps'], x['temps'])[1],axis=1)
    dico_tps_max2.drop(['tps','type_tps_1','type_tps_2'],axis=1, inplace=True)
    
    return dico_tps_max2

def corriger_tps_parcours_extrapole(dixco_tpsmax_corrige,df_transit_extrapole):
    """
    modifier le dico des temps de trajets max de transit pour y inserer les temps issus de l'extrapolation (fonction predire_type_trajet)
    en entree : 
        dixco_tpsmax_corrige : df issue de corriger_df_tps_parcours
        df_transit_extrapole df des trajets de transit extrapole predire_type_trajet
    en sortie : 
        correction_temps : df des temps max avec mise a jour des temps et type pour les periodes et o_d extrapolees
    """
    period_od_a_modif=df_transit_extrapole.loc[(df_transit_extrapole['type']=='predit')&(df_transit_extrapole['filtre_tps']==1)].sort_values('date_cam_1').copy()
    liste_pour_modif=period_od_a_modif[['period','o_d','temps']].drop_duplicates()#.merge(dixco_tpsmax_corrige, on=['period','o_d'], how='right')
    liste_pour_modif['type']='predit'
    correction_temps=dixco_tpsmax_corrige.merge(liste_pour_modif, on=['period','o_d'], how='left')
    correction_temps['temps_x']=np.where(pd.notnull(correction_temps['temps_y']),correction_temps['temps_y'],correction_temps['temps_x'])
    correction_temps['type_x']=np.where(pd.notnull(correction_temps['type_y']),correction_temps['type_y'],correction_temps['type_x'])
    correction_temps=correction_temps.rename(columns={'temps_x':'temps','type_x':'type'}).drop(['temps_y','type_y'],axis=1)
    
    return correction_temps

def passages_fictif_rocade (liste_trajet, df_od,df_passages_transit,df_pl):
    """
    Créer des passages pour les trajets de transit non vus sur la Rocade mais qui y sont passé
    en entree : 
        liste_trajet : df des trajets concernes, issu de liste_trajet_rocade
        df_od : df des trajetsde transit validé selon le temps de parcours
        df_passages_transit : df des passages concernés par un trajet de transit (issu du traitement o_d)
        df_pl : df de tout passages pl (issu simplement de l'import mise en forme)
    en sortie : 
        df_passage_transit_redresse : df des passages concernés par un trajet de transit (issu du traitement o_d) + passages fictifs
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
    #trouver ceux qui ne contiennent aucune référence uax camera de la Rocade
    trajets_rocade_non_vu=trajets_rocade.loc[trajets_rocade.apply(lambda x : all(e not in x['cameras'] for e in [1,2,3,4]),axis=1)].copy()
    #créer des passage fictif au niveau de la Rocade avec comme created la moyenne entre date_cam_1 et date_cam_2
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


def differencier_rocade(df_od) :
    """
    Séparer le trafic Rocade Est du Ouest ou inconnu, pour les trajets entre A10 ou N10 et A63 ou A660
    en entree : 
        df_od : une df des o_d issu de l'affectation
    en sortie : 
        pivot_type_rocade : pivot table avec en index les o_d, en column le cote de la rocade
    """
    def diff_type_rocade(cameras):
        """
        Rocade Est ou ouest ou ucune selon les cameras ayant vu les PL
        en entree : 
            cameras : liste des cameras, attribut de la df
        en sortie : string : Ouest, Est, Autre
        """
        if any(e in cameras for e in [4,3]) : 
            return 'Est'
        elif any(e in cameras for e in [1,2]) :
            return 'Ouest'
        else : return 'Autre'
    #Isoler les trajets concernes et affecte un type de Rocade
    df_od_concernees=df_od.loc[df_od.o_d.isin(['N10-A63', 'A10-A63', 'N10-A660', 'A10-A660', 'A63-N1O',
                                                          'A63-A10', 'A660-N10','A660-A10'])].copy()
    df_od_concernees['type_rocade']=df_od_concernees.apply(lambda x : diff_type_rocade(x['cameras']),axis=1)
    return pd.pivot_table(df_od_concernees,values='l', index='o_d', columns='type_rocade',aggfunc='count', margins=True)

def predire_type_trajet(df_trajet,o_d, date, gamma, C):
    """
    retravailler les trajets non soumis aux cluster pour avoir une différenciation trabsit / local plus pertinente
    en entree : 
        df_trajet : df des trajets issu de jointure_temps_reel_theorique
        o_d : string : origie destination que l'on souhaite etudier
        date : string : YYYY-MM-DD : date etudiee
        gamma : integer : paramètre de prediction, cf sklearn
        C : integer : paramètre de prediction, cf sklearn
    en sortie :
        df_trajet : df des trajets avec le type modifie et le filtre_tps aussi
    """
    #isoler les données : sur un jour pour une o_d
    test_predict=df_trajet.loc[(df_trajet['o_d']==o_d) &
                           (df_trajet.set_index('date_cam_1').index.dayofyear==pd.to_datetime(date).dayofyear)].copy()
    #ajouter des champsde ocnversion des dates en integer, limiter les valeusr sinon pb de mémoire avec sklearn
    test_predict['date_int']=((test_predict.date_cam_1 - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))/1000000
    test_predict['temps_int']=(((pd.to_datetime('2018-01-01')+test_predict.tps_parcours) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))/1000000
    #créer les données d'entrée du modele
    X=np.array([[a,b] for a,b in zip(test_predict.date_int.tolist(),test_predict.temps_int.tolist())])
    y=np.array(test_predict.filtre_tps.tolist())
    #créer le modele
    clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
    #alimenter le modele
    clf.fit(X, y)
    
    #MODIFIER UNIQUEMENT LES VALEUR A 0
    #isoler les donner à tester
    df_a_tester=test_predict.loc[(test_predict['filtre_tps']==0) & (test_predict['type']=='85eme_percentile')].copy()
    #liste à tester
    liste_a_tester=np.array([[a,b] for a,b in zip(df_a_tester.date_int.tolist(),df_a_tester.temps_int.tolist())])
    #df de résultats de prédiction
    df_type_predit=pd.DataFrame([[i, v] for i,v in zip(df_a_tester.index.tolist(),[clf.predict([x])[0] for x in liste_a_tester])], 
                                columns=['index_source','type_predit'])
    #mise à jourde la df source : on met à jour le type
    df_trajet.loc[df_trajet.index.isin(df_type_predit.index_source.tolist()),'type']='predit'
    df_trajet.loc[df_trajet.index.isin(df_type_predit.loc[df_type_predit['type_predit']==1].index_source.tolist()),'filtre_tps']=1
    
    #MODIFIER L VALEUR DU TEMPS DE PARCOURS DE REFERENCE SUR L'INTERVAL CONSIDERE
    df_trajet_extra=df_trajet.loc[(df_trajet['filtre_tps']==1)&(df_trajet['type']=='predit')].copy()
    nw_tps_parcours=df_trajet_extra.groupby('period')['tps_parcours'].max()
    df_trajet=df_trajet.merge(nw_tps_parcours.reset_index(), on='period', suffixes=('_1','_2'), how='left')
    df_trajet['temps']=np.where(pd.notnull(df_trajet['tps_parcours_2']),df_trajet['tps_parcours_2'],df_trajet['temps'])
    df_trajet.drop('tps_parcours_2',axis=1,inplace=True)
    df_trajet.rename(columns={'tps_parcours_1':'tps_parcours'},inplace=True)
    
    return df_trajet

def correction_temps_cestas(df_transit_extrapole,df_passages_immat_ok,dixco_tpsmax_corrige):
    """
    Déterminé sui trafic est en transit en fonction du temps entre Cestas et entree ou sortie LAPI plutot que sur l'intégralité du trajet des PL concernes par A63.
    Permet de s'affranchir des pauses sur aires A63
    en entree : 
        df_transit_extrapole : df de transit issus de predire_type_trajet
        df_passages_immat_ok : df des passages valides apres tri des aberrations (doublons, passages proches, plaques foireuses)
        dixco_tpsmax_corrige : df des temps de parcours max, issu de corriger_tps_parcours_extrapole
    en sortie : 
        df_transit_A63_redresse_tstps : df des trajets concernes, avec tous les attributs relatifs à Cestas, sans correction du filtre temps
    """
    
    def tps_parcours_cestas(od,date_cam_1, date_cam_2, date_cestas):
        """
        calculer le temps de paroucrs depuis ou vers Cestas pour les o_d concernees par A63
        en entree : 
           od : string : origine_destination du trajet
           date_cam_1 : pd.timestamp : date de passage du debut du trajet
           date_cam_2 : pd.timestamp : date de passage de fin du trajet 
           date_cestas : pd.timestamp : date de passage à la camera de cestas
        en sortie : 
            tps_parcours_cestas : pd.timedelta : tps de parcours entre le debut et cestas ou entre cestas et la fin du trajet
        """
        if od.split('-')[0]=='A63':
            return date_cam_2-date_cestas
        else : 
            return date_cestas-date_cam_1
    
    def temps_pour_filtre(date_passage,type_tps_lapi, tps_lapi, tps_theoriq):
        """pour ajouter un attribut du temps de parcours qui sert à filtrer les trajets de transit"""
        marge = 660 if date_passage.hour in [19,20,21,22,23,0,1,2,3,4,5,6] else 0  #si le gars passe la nuit, on lui ajoute 11 heure de marge
        if type_tps_lapi in ['Cluster','moyenne Cluster','predit']:
            return tps_lapi+pd.Timedelta(str(marge)+'min')
        else : 
            return tps_theoriq+pd.Timedelta(str(marge)+'min')
       
    #selectionner les trajets realtif à A63, qui ne sont pas identifies comme transit
    df_transit_A63_redresse=df_transit_extrapole.loc[(df_transit_extrapole['filtre_tps']==0)&(
        df_transit_extrapole.apply(lambda x : 'A63' in x['o_d'],axis=1))&(
        df_transit_extrapole.apply(lambda x : (18 in x['cameras'] or 19 in x['cameras']),axis=1))]
    #trouver les passages correspondants
    passage_transit_A63_redresse=trajet2passage(df_transit_A63_redresse,df_passages_immat_ok)
    #retrouver le passage correspondants à camera 18 ou 19
    df_transit_A63_redresse=df_transit_A63_redresse.merge(passage_transit_A63_redresse,on='immat')
    df_transit_A63_redresse=df_transit_A63_redresse.loc[df_transit_A63_redresse['created'].between(df_transit_A63_redresse['date_cam_1'],df_transit_A63_redresse['date_cam_2'])]
    df_transit_A63_redresse=df_transit_A63_redresse.loc[(df_transit_A63_redresse['camera_id'].isin([18,19])) & 
                                                        (df_transit_A63_redresse.apply(lambda x: x['camera_id'] in x['cameras'],axis=1))]
    df_transit_A63_redresse=df_transit_A63_redresse.rename(columns={'l_x':'l','state_x':'state','created':'date_cestas'}).drop(['l_y','state_y','fiability','camera_id'],axis=1)
    #affecter tps de parcours vers ou depuis Cestas
    df_transit_A63_redresse['tps_parcours_cestas']=df_transit_A63_redresse.apply(
        lambda x : tps_parcours_cestas(x['o_d'], x['date_cam_1'], x['date_cam_2'], x['date_cestas']),axis=1)
    
    #creer les temps de parcours theorique et reel de Cestas
    #nouvel attribut pour traduire l'o_d cestas et les cameras cestas
    df_transit_A63_redresse['o_d_cestas']=df_transit_A63_redresse.apply(
        lambda x : 'A660-'+x['o_d'].split('-')[1] if x['o_d'].split('-')[0]=='A63' else x['o_d'].split('-')[0]+'-A660',axis=1)
    df_transit_A63_redresse['cameras_cestas']=df_transit_A63_redresse.apply(
        lambda x : x['cameras'][1:] if x['o_d'].split('-')[0]=='A63' else x['cameras'][:-1],axis=1)
    #jointure avec temps theorique
    df_transit_A63_redresse_tpq_theoriq=df_transit_A63_redresse.merge(liste_complete_trajet[['cameras','tps_parcours_theoriq']]
        ,left_on=['cameras_cestas'],right_on=['cameras']).drop('cameras_y',axis=1).rename(
        columns={'tps_parcours_theoriq_y':'tps_parcours_theoriq_cestas'})
    #jointure avec temps reel
    df_transit_A63_redresse_tstps=(df_transit_A63_redresse_tpq_theoriq.merge(dixco_tpsmax_corrige, left_on=['o_d_cestas','period'],right_on=['o_d','period'],
        how='left').rename(columns={'cameras_x':'cameras','o_d_x':'o_d','date_x':'date','type_x':'type',
                                    'temps_x':'temps','tps_parcours_theoriq_x':'tps_parcours_theoriq',
                                    'temps_y':'temps_cestas','type_y':'type_cestas'}).drop(['date_y','o_d_y'],axis=1))
    #Maj de l'attribut temps_filtre_cestas
    df_transit_A63_redresse_tstps['temps_filtre_cestas']=df_transit_A63_redresse_tstps.apply(lambda x : temps_pour_filtre(x['date_cam_1'],
        x['type_cestas'], x['temps_cestas'], x['tps_parcours_theoriq_cestas']), axis=1)
    return df_transit_A63_redresse_tstps

def forme_df_cestas(df_transit_A63_redresse):  
    """
    suppression des attributs relatifs a cestas et creation attributs drapeau de correction
    en entree : 
        df_transit_A63_redresse : df des trajets  issu de correction_temps_cestas avec MaJ tps_filtre depuis identifier_transit
    en sortie : 
        df_transit_A63_attr_ok : df des trajets épurées avec attributs drapeau de suivi de modif : correction_o_d et correction_o_d_type
    """
    #Mise à jour structure table et ajout attribut drapeau de correction
    df_transit_A63_attr_ok=df_transit_A63_redresse.drop([attr_cestas for attr_cestas in df_transit_A63_redresse.columns.tolist() if attr_cestas[-7:]=='_cestas'],axis=1)
    df_transit_A63_attr_ok.loc[df_transit_A63_attr_ok['filtre_tps']==1,'correction_o_d']=True
    df_transit_A63_attr_ok.loc[df_transit_A63_attr_ok['filtre_tps']==0,'correction_o_d']=False
    df_transit_A63_attr_ok['correction_o_d_type']=df_transit_A63_attr_ok.apply(lambda x : 'temps_cestas' if x['correction_o_d'] else 'autre',axis=1)
    return df_transit_A63_attr_ok
    
    
    
    
    
    
    

    