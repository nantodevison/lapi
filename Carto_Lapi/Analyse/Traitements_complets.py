# -*- coding: utf-8 -*-
'''
Created on 23 juin 2019

@author: martin.schoreisz

Module de traitement complet des donn�es LAPI : import, mise en forme, trajets, transit.
obetntion d'une dataframe des trajets de transit 
'''

from trajets import trajet2passage
from Import_Forme import import_et_mise_en_forme, liste_complete_trajet, liste_trajet_incomplet
from transit import transit_temps_complet, jointure_temps_reel_theorique, identifier_transit, param_trajet_incomplet, transit_trajet_incomplet
from Correction_transit import corriger_df_tps_parcours, predire_ts_trajets,corriger_tps_parcours_extrapole,correction_temps_cestas,forme_df_cestas,correction_trajet
import pandas as pd
import os

def definir_transit():
    """
    regroupeement des fonctions permettant d'arriver au trafic de transit PL prenant en compte le temps de pause à Cestas
    """
    #importer les données et recuperer les données mise en forme
    df_passages_immat_ok=import_et_mise_en_forme()[0]
    #creer la datafrme de base des trajets susceptible d'etre en transit
    dico_od,  dico_passag, dico_tps_max=transit_temps_complet('2019-01-23 00:00:00',22,df_passages_immat_ok)
    #corriger le dico des temps de parcours
    dixco_tpsmax_corrige=corriger_df_tps_parcours(dico_tps_max)
    #joindre avec les temps theoriques
    df_transit_tps_ref=jointure_temps_reel_theorique(dico_od,dixco_tpsmax_corrige,liste_complete_trajet)
    #df des transit avec marge 0 ss extrapolation
    df_transit_marge0_ss_extrapolation=identifier_transit(df_transit_tps_ref, 0)
    #extrapole e, se basant sur du macine learning
    df_transit_extrapole=predire_ts_trajets(df_transit_marge0_ss_extrapolation)
    #mettre a jour e dico des temps max autorises
    dixco_tpsmax_corrige=corriger_tps_parcours_extrapole(dixco_tpsmax_corrige,df_transit_extrapole)
    #creation des attributs relatifs a Cestas, pour les PL sur une O-D liées à A63, non identifiés comme transit, et qui ont été vus à Cestas
    df_transit_A63_redresse=correction_temps_cestas(df_transit_extrapole,df_passages_immat_ok,dixco_tpsmax_corrige)
    return df_transit_A63_redresse, df_transit_extrapole, df_passages_immat_ok, dixco_tpsmax_corrige

def appliquer_marge(liste_marges,df_transit_A63_redresse, df_transit_extrapole):
    """
    retourner un dico des trajets de transit en prenant en compte une marge.
    chaque entree du dico correspond à une valeur de marge
    en entree : 
        liste_marges : list ed'entier >0 correspondant aux marge en minute
        df_transit_extrapole : df des trajets de susceptible d'etre en transit, apres extrapoation par machine learning
        df_transit_A63_redresse : df des trajets suscetible d'etre en transit apres prise en compte aires A63
    en sortie :
        dico_df_transit : df de tout les trajets suceotible d'etre en etransit (filtre_tps = 1 ou 0)
        dico_df_od_ok : df des trajets en transit (filtre_tps = 1)
    """
    #appliquer la martge sur les donnees issu de l'extrapolation et sur celle issues de la prise ene compte des aires
    dico_df_transit={}
    for i in liste_marges :
        dico_df_transit['df_transit_extrapole_marge'+str(i)]=identifier_transit(df_transit_extrapole, i)#identifier le transit pour tous les PL
        dico_df_transit['df_transit_airesA63_marge'+str(i)]=identifier_transit(df_transit_A63_redresse, 15,'temps_filtre_cestas','tps_parcours_cestas')#identifier le transit pour PL passé par Cestas
        dico_df_transit['df_transit_airesA63_marge'+str(i)]=forme_df_cestas(dico_df_transit['df_transit_airesA63_marge'+str(i)])
        dico_df_transit['df_transit_marge'+str(i)]=pd.concat([dico_df_transit['df_transit_airesA63_marge'+str(i)],
              dico_df_transit['df_transit_extrapole_marge'+str(i)]],sort=False)#attention cela crée des doublons car ceux present dans A63_redresse sont aussi dans extrapole
        dico_df_transit['df_transit_marge'+str(i)].correction_o_d=(dico_df_transit['df_transit_marge'+str(i)].
                                                               correction_o_d.fillna(False).copy()) 
        dico_df_transit['df_transit_marge'+str(i)]=dico_df_transit['df_transit_marge'+str(i)].sort_values(['date_cam_1', 'immat','filtre_tps']).copy() #tri
        dico_df_transit['df_transit_marge'+str(i)].drop_duplicates(['date_cam_1','immat'],keep='last', inplace=True)#puis suppression des doublons
    dico_df_od_ok={'df_od_ok_marge'+str(i):dico_df_transit['df_transit_marge'+str(i)].loc[dico_df_transit['df_transit_marge'+str(i)]['filtre_tps']==1].copy()
         for i in liste_marges}# on ne conserve que les PL en transit

    return dico_df_transit, dico_df_od_ok

def correction_A660(dico_df_od_ok,df_passages_immat_ok,liste_marges):
    """
    appliquer la corrction A660 pour un ensemble de marge
    en entree : 
        dico_df_od_ok : df des trajets en transit (filtre_tps = 1) cf appliquer_marge
        df_passages_immat_ok :df des passages, cf definir_transit
        liste_marges : list ed'entier >0 correspondant aux marge en minute
    en sortie : 
        dico_corr_A63_A660 : df des trajets de transit, avec certaines o_d odifiees
    """
    dico_corr_A63_A660={'corr_A63_A660'+str(i):correction_trajet(df_passages_immat_ok, dico_df_od_ok ['df_od_ok_marge'+str(i)])
                    for i  in liste_marges}
    return dico_corr_A63_A660

def extrapol_trajets_incomplets(dico_df_od_ok,df_passages_immat_ok,dico_corr_A63_A660,liste_marges, dixco_tpsmax_corrige):
    """
    extrapolation des trajets de trasit pour des véhicules deja vues sur des trajets complets, pour un ensemble de marge
    """
    dico_df_od_final={}
    # dico des df des passages avant correction A660
    dico_passag_avantcorr={'passag_avantcorr'+str(i):
                   trajet2passage(dico_df_od_ok ['df_od_ok_marge'+str(i)],df_passages_immat_ok) 
                   for i in liste_marges}
    #extrapolation trajets icomplets
    for i  in liste_marges:
        df_filtre_A63,df_passage_transit,df_non_transit=(param_trajet_incomplet(dico_corr_A63_A660['corr_A63_A660'+str(i)],df_passages_immat_ok,
            dico_passag_avantcorr['passag_avantcorr'+str(i)]))
        trajet_transit_incomplet2, passage2=transit_trajet_incomplet(df_filtre_A63,df_passage_transit,'2019-01-23 00:00:00',22, df_passages_immat_ok)
        #affectation des temps de parcours de reference
        df_transit_incomplet_tps_ref=jointure_temps_reel_theorique(trajet_transit_incomplet2,dixco_tpsmax_corrige,liste_trajet_incomplet,'incomplet')
        #Maj de l'attruibut drapeau et Maj des autres attruibuts
        df_transit_incomplet_tps_ref=identifier_transit(df_transit_incomplet_tps_ref, 0)
        #mettre en forme les attributs
        df_transit_incomplet_tps_ref=df_transit_incomplet_tps_ref[['date_cam_1', 'immat', 'state', 'l', 'date_cam_2', 'cameras', 'origine',
               'destination', 'o_d', 'tps_parcours', 'period', 'date', 'temps', 'type',
               'tps_parcours_theoriq','temps_filtre', 'filtre_tps']]
        #filtrer le df : 
        df_transit_incomplet_tps_ref_final=df_transit_incomplet_tps_ref.loc[df_transit_incomplet_tps_ref['filtre_tps']==1].copy()
        #ajouter l'attribut d'identification des trajets 
        df_transit_incomplet_tps_ref_final['correction_o_d']=True
        df_transit_incomplet_tps_ref_final['correction_o_d_type']='extrapole'
        #6. Ajouter au df des o_d précédents : 
        dico_df_od_final['df_od_final_marge'+str(i)]=pd.concat([dico_corr_A63_A660['corr_A63_A660'+str(i)],df_transit_incomplet_tps_ref_final],sort=False)
    return dico_df_od_final

def save_results(chemin,dico_df_od_final,liste_marges ):
    """ 
    sauvegardere les resultats pour plusieurs marges
    en entree : 
        chemin : raw string du chemin vers le dossier de sauvergarde
    """
    for i in liste_marges:
        nomfichier=f'marge{i}min.json'
        dico_df_od_final['df_od_final_marge'+str(i)].reset_index().drop(['level_0','index'],axis=1).to_json(
            os.path.join(chemin,nomfichier),orient='index')