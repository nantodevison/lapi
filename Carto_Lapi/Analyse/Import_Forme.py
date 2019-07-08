# -*- coding: utf-8 -*-
'''
Created on 21 juin 2019

@author: martin.schoreisz

Mise en forme des donn�es issues de la Bdd Lapi
Import des attributs de donn�es n�cessaires
'''

import pandas as pd
import numpy as np
import Connexion_Transfert as ct
from statistics import mode, StatisticsError
import re

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
liste_complete_trajet=mise_en_forme_dfs_trajets(r'C:\Users\martin.schoreisz\Desktop\LAPI_LOCAL\Traitements\python\trajets_possibles.json','complet')
liste_trajet_incomplet=mise_en_forme_dfs_trajets(r'C:\Users\martin.schoreisz\Desktop\LAPI_LOCAL\Traitements\python\liste_trajet_incomplet.json','incomplet')
liste_trajet_rocade=pd.read_json(r'C:\Users\martin.schoreisz\Desktop\LAPI_LOCAL\Traitements\python\liste_trajet_rocade.json', orient='index')
param_cluster=pd.read_json(r'C:\Users\martin.schoreisz\Desktop\LAPI_LOCAL\Traitements\python\param_cluster.json', orient='index')
#fichier des plaques, en df
plaques_europ=pd.read_csv(r'C:\Users\martin.schoreisz\Desktop\LAPI_LOCAL\Traitements\python\plaques_europ.txt', sep=" ", header=None, names=['pays','re_plaque'])
#matrices des nb de jours
matrice_nb_jo=pd.read_json(r'C:\Users\martin.schoreisz\Desktop\LAPI_LOCAL\Traitements\python\nb_jours_mesures.json',orient='index').pivot(
    index='origine', columns='destination',values='nb_jo').replace('NC',np.NaN)
matrice_nb_jo_samedi=pd.read_json(r'C:\Users\martin.schoreisz\Desktop\LAPI_LOCAL\Traitements\python\nb_jours_mesures.json',orient='index').pivot(
    index='origine', columns='destination',values='nb_jo_samedi').replace('NC',np.NaN)
matrice_nb_j_tot=pd.read_json(r'C:\Users\martin.schoreisz\Desktop\LAPI_LOCAL\Traitements\python\nb_jours_mesures.json',orient='index').pivot(
    index='origine', columns='destination',values='nb_j_tot').replace('NC',np.NaN)
#donnees de comptage gestionnaire
donnees_gest=pd.read_csv(r'C:\Users\martin.schoreisz\Desktop\LAPI_LOCAL\Traitements\python\Synthese_trafic_LAPI.csv')
donnees_gest['created']=pd.to_datetime(donnees_gest.created)


    
def ouvrir_fichier_lapi_final(date_debut, date_fin) : 
    """ouvrir les donnees lapi depuis la Bdd 'lapi' sur le serveur partage GTI
    l'ouvertur se fait par appel d'une connexionBdd Python (scripts de travail ici https://github.com/nantodevison/Outils/blob/master/Outils/Martin_Perso/Connexion_Transfert.py)
    en entree : date_debut : string de type YYYY-MM-DD hh:mm:ss
                date_fin: string de type YYYY-MM-DD hh:mm:ss
    en sortie : dataframe pandas des passages df_passage, des plaques d'immatriculation non cryptee df_plaque, des immats avec les attributs d'identification vl/pl
    """
    with ct.ConnexionBdd('gti_lapi_final') as c : 
        requete_passage=f"select case when camera_id=13 or camera_id=14 then 13 when camera_id=15 or camera_id=16 then 15 else camera_id end::integer as camera_id , created, immatriculation as immat, fiability, l, state from data.te_passage where created between '{date_debut}' and '{date_fin}'"
        df_passage=pd.read_sql_query(requete_passage, c.sqlAlchemyConn)
        requete_plaque=f"select plaque_ouverte, chiffree from data.te_plaque_courte"
        df_plaque=pd.read_sql_query(requete_plaque, c.sqlAlchemyConn)
        requete_immat=f"select immatriculation,pl_siv,pl_3barriere,pl_2barriere,pl_1barriere,pl_mmr75,vul_siv,vul_mmr75,vl_siv,vl_mmr75  from data.te_immatriculation"
        df_immat=pd.read_sql_query(requete_immat, c.sqlAlchemyConn)
        return df_passage,df_plaque, df_immat

def recalage_passage_1h(df_passage):
    """
    ajouter une heure à tous les passage
    """
    df_passage['created']=df_passage['created']+pd.Timedelta('1H')
    return df_passage

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
    def ecart_passage(liste_passage, state) : 
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
    def liste_created(liste_created) : 
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
    groupe['erreur_tps_passage']=groupe.apply(lambda x :  ecart_passage(x['created'],x['state']),axis=1)
    #et on extrait unqiement les passages problemetaique
    groupe_pl_rappro=groupe[groupe['erreur_tps_passage']].copy()
    groupe_pl_rappro['liste_passag_faux']=groupe_pl_rappro.apply(lambda x : liste_passage(x['camera_id'],x['created']),axis=1)
    groupe_pl_rappro['liste_created_faux']=groupe_pl_rappro.apply(lambda x : liste_created(x['created']),axis=1)
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
    def check_valid_plaque(plque_ouvert):
        """
        Marqueur si la plque correspond à un type de plaque attendu, cf Q:\DAIT\TI\DREAL33\2018\C17SI0073_LAPI\Traitements\python\plaques_europ.txt
        """
        if not re.match('^([0-9])+$|^([A-Z])+$',plque_ouvert) :
            return any([re.match(retest,plque_ouvert) for retest in plaques_europ.re_plaque.tolist()])
        else : return False
    #jointure avec les plaques ouvertes
    df_passages_plaque_ouverte=df.reset_index().merge(df_plaques, left_on='immat', right_on='chiffree')
    df_passages_plaque_ouverte['plaque_valide']=df_passages_plaque_ouverte.apply(lambda x : check_valid_plaque(x['plaque_ouverte']),axis=1)
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

def affecter_type_nuit(df_passages_affectes):
    """
    affecter le type à des immats vue de nuit, plque etrangere, sur un trajetde transit, non vu avant, fiabilite ok
    en entre : 
        df_passages_affectes : passages issus de affecter_type
    en sortie : 
        df : df_passages_affectes avec attribut l modifié
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

def import_et_mise_en_forme(type_veh=1):
    """
    fonction globale d'enchaiement des traitements du module
    en entree : 
        type_veh le type de veihcule sur leqel on veut travailler (par defaut 1=PL, sinon O=VL, sinon2=VUL sinon error)
    """
    #importer les données
    df_passages_source, df_plaques, df_immat=ouvrir_fichier_lapi_final('2019-01-22 23:00:00','2019-02-13 22:59:59')
    #recaler les passages d'1h
    df_passages_source=recalage_passage_1h(df_passages_source
                                           )
    #affecter le type de vehicule
    df_passages_source=affecter_type(df_passages_source,df_immat)
    df_passages_source=affecter_type_nuit(df_passages_source)
    #filtrer que les PL
    df_passages_pl=df_passages_source.loc[df_passages_source['l']==1].copy()
    #supprimer les doublons
    df_passages_pl=supprimer_doublons(df_passages_pl)
    #suppr passages proches
    df_passages_pl=recalage_cam10(df_passages_pl)
    #filtre des immatriculations
    df_passages_immat_ok, df_immat_suppr=filtre_plaque_non_valable(df_passages_pl, df_plaques)

    
    return df_passages_immat_ok, df_immat_suppr
    
    
    
    