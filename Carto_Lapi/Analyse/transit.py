# -*- coding: utf-8 -*-
'''
Created on 21 juin 2019

@author: martin.schoreisz

Module de d�termination du trafic de transit � partir de la classe des trajets du module trajet
'''

from Import_Forme import liste_complete_trajet, liste_trajet_incomplet
from trajets import trajet, PasDePlError, grouper_pl,filtre_et_forme_passage, cam_voisines, trajet2passage
from Correction_transit import forme_df_cestas
import pandas as pd
import datetime as dt

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

def transit_marge0(df_transit_extrapole,df_transit_A63_redresse):
    """
    creer la df de base de transit, sans marge
    en entree : 
        df_transit_extrapole : df issue de Correction_transit.predire_ts_trajets
        df_transit_A63_redresse : df issue de Correction_transit.correction_temps_cestas
    """
    dico_df_transit={}
    dico_df_transit['df_transit_airesA63_ss_filtre']=forme_df_cestas(identifier_transit(df_transit_A63_redresse, 15,'temps_filtre_cestas','tps_parcours_cestas'))#identifier le transit pour PL passé par Cestas
    dico_df_transit['df_transit_airesA63_avec_filtre']= dico_df_transit['df_transit_airesA63_ss_filtre'].loc[
         dico_df_transit['df_transit_airesA63_ss_filtre']['filtre_tps']==1].copy()
    dico_df_transit['df_transit_pas_airesA63_ss_filtre']=df_transit_extrapole.loc[~df_transit_extrapole.set_index(['date_cam_1','immat']).index.isin(
                df_transit_A63_redresse.set_index(['date_cam_1','immat']).index.tolist())].copy()
    
    dico_df_transit['df_transit_marge0_ss_filtre']=pd.concat([dico_df_transit['df_transit_airesA63_ss_filtre'],
                  dico_df_transit['df_transit_pas_airesA63_ss_filtre']],sort=False)
    dico_df_transit['df_transit_marge0_ss_filtre'].correction_o_d=dico_df_transit['df_transit_marge0_ss_filtre'].correction_o_d.fillna(False)
    dico_df_transit['df_transit_marge0_avec_filtre']=dico_df_transit['df_transit_marge0_ss_filtre'].loc[
        dico_df_transit['df_transit_marge0_ss_filtre']['filtre_tps']==1].copy()
    return dico_df_transit

    
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
            df_journee=df_3semaines.loc[date:date+pd.Timedelta(hours=552)]
        else : 
            df_journee=df_3semaines.loc[date:date+pd.Timedelta(hours=552)]
        if date.hour==0 : print(f"date : {date} debut_traitement : {dt.datetime.now()}")
        for cameras in zip([15,12,8,10,19,6],range(6)) : #dans ce mode peu importe la camera d'arrivée, elle sont toutes analysées
            if 'dico_passag' in locals() : #si la varible existe deja on utilise pour filtrer le df_journee en enlevant les passages dejà pris dans une o_d (sinon double compte ente A63 - A10 et A660 -A10 par exemple 
                try:
                    donnees_trajet=trajet(df_journee,date,duree,cameras, typeTrajet='Global',df_filtre=dico_passag.loc[dico_passag['created']>=date].copy(),
                                      liste_trajet=liste_trajet_loc)
                except PasDePlError :
                    continue
            else : 
                try:
                    donnees_trajet=trajet(df_journee,date,duree,cameras, typeTrajet='Global',liste_trajet=liste_trajet_loc)
                except PasDePlError:
                    continue
            df_trajet, df_passag, df_tps_max=donnees_trajet.df_transit, donnees_trajet.df_passag_transit, donnees_trajet.temps_parcours_max   
            
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

def param_trajet_incomplet(df_od_corrige,df_3semaines,dico_passag):
    """
    Récupérer les paramètres necessaires à la fonction transit_trajet_incomplet
    en entree : 
        dico_passag : df des passages de transit issu des précédentes fonctions
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
    
    #grouper les passage transit, associer le nombre de fois où ils ont passés puis ne conserver que ceux qui sont passé au moins 1 fois
    df_transit_nb_passage=df_od_corrige.groupby(['immat','o_d'])['l'].count().reset_index().rename(columns={'l':'Nb_occ'})
    df_immat_transit_nb_passage_sup2=df_transit_nb_passage.loc[df_transit_nb_passage['Nb_occ']>=1]
    
    #df des passages qui n'ont pas été identiiés comme transit, mais qui ont une immat qui a déjà fait du transit
    df_passage_transit=df_non_transit.loc[(df_non_transit.immat.isin(dico_passag.immat.unique()))]
    
    #identifier les doucblons : tel que présente le fichier présente bcp d'immat en double avec par exempele les o_d A660-N10 puis N10-A660.
    #or tout les trajets finissant par A660 ou A63 sont déja traites plus haut, donc on les vire
    df_filtre_A63=df_immat_transit_nb_passage_sup2.loc[df_immat_transit_nb_passage_sup2.apply(lambda x : x['o_d'].split('-')[1] not in ['A660','A63'],axis=1)]
    
    return df_filtre_A63, df_passage_transit, df_non_transit
    

def transit_trajet_incomplet(df_filtre_A63,df_passage_transit,date_debut,nb_jours, df_3semaines,liste_trajet_loc=liste_trajet_incomplet):
    """
    Extrapoler des trajest à partir des immats en transit,sur des trajets où il manque la camera de fin ou de debut
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
                groupe_pl,df_duree_cam1,df_duree_autres_cam=grouper_pl(df_passage_transit , date, date_fin, cameras, df_passage_transit_incomplet)
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
            #attention, là il y a des doublons si le PL a deja effectue plusieurs trajets
            # dans ce cas je fais un drop duplicates quiaffece aleatoirement le PL a un trajet, car le nb de PL dans ce casest très faible (<1/10000)
            dico_filtre = {'destination':[15,6,8,10,12]}
            trajet_transit_incomplet=trajets_possible_enrichi.loc[trajets_possible_enrichi.apply(lambda x : (x['cam_suivant'] in dico_filtre['destination']) & (
                                                                        x['o_d_immat']==x['o_d_liste_trajet']),axis=1)].copy()
            trajet_transit_incomplet.rename(columns={'o_d_liste_trajet':'o_d'}, inplace=True)
            trajet_transit_incomplet=trajet_transit_incomplet.drop_duplicates(['date_cam_1', 'immat'])
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
    affecter un attribut drapeau d'identification du trafic de trabsit, selon une marge (marge variable selon o_d)
    en entree : 
        df_transit_temps : df des transit
        marge : integer: marge possible entre le temps theorique ou lapi et le temsp de passage. comme les camions doivent faire une pause de 45min toute les 4h30...
    en sortie : 
        df_transit_temps : df avec la'ttribut filtre_tps identifiant le trafic de trabsit (1) ou non (0)
    """
    def filtre_tps_parcours(temps_filtre,tps_parcours, marge,o_d) : 
        """pour ajouter un attribut drapeau sur le tempsde parcours, et ne conserver que les trajets de transit"""
        if o_d in ['A10-A63', 'A63-A10', 'N10-A63', 'A63-N10'] and marge>15 : 
            if tps_parcours <= temps_filtre+pd.Timedelta(str(marge-15)+'min') :
                return 1
            else: 
                return 0
        elif 'A63' not in o_d : 
            marge=15
            if tps_parcours <= temps_filtre+pd.Timedelta(str(marge)+'min') :
                return 1
            else: 
                return 0
        else : 
            if tps_parcours <= temps_filtre+pd.Timedelta(str(marge)+'min') :
                return 1
            else: 
                return 0
        
    df_transit_temps_final=df_transit_temps.copy()
    df_transit_temps_final['filtre_tps']=df_transit_temps_final.apply(lambda x : filtre_tps_parcours(x[nom_attribut_temps_filtre],
                                                                    x[nom_attribut_tps_parcours],marge, x['o_d']), axis=1)
    return df_transit_temps_final
