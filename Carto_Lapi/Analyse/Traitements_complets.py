# -*- coding: utf-8 -*-
'''
Created on 23 juin 2019

@author: martin.schoreisz

Module de traitement de regroupement de certaines fcontions
obetntion d'une dataframe des trajets de transit 
'''

from trajets import trajet2passage
from Import_Forme import  liste_trajet_incomplet
from transit import  jointure_temps_reel_theorique, identifier_transit, param_trajet_incomplet, transit_trajet_incomplet
from Correction_transit import correction_trajet
import pandas as pd
import os


def appliquer_marge(liste_marges,df_transit_airesA63, df_transit_pas_airesA63):
    """
    retourner un dico des trajets de transit en prenant en compte une marge.
    chaque entree du dico correspond à une valeur de marge
    en entree : 
        liste_marges : list ed'entier >0 correspondant aux marge en minute
        df_transit_pas_airesA63 : df des trajets non concernes par les aires A63
        df_transit_airesA63 : df concernes par aires A63 (pas de marge sur ceux là)
    en sortie :
        dico_df_transit : df de tout les trajets suceotible d'etre en etransit (filtre_tps = 1 ou 0)
        dico_df_od_ok : df des trajets en transit (filtre_tps = 1)
    """
    #appliquer la martge sur les donnees issu de l'extrapolation et sur celle issues de la prise ene compte des aires
    dico_transit_avec_marge={}
    for i in liste_marges :
        dico_transit_avec_marge['df_transit_pas_airesA63_marge'+str(i)]=(identifier_transit(df_transit_pas_airesA63, i))
        dico_transit_avec_marge['df_transit_marge'+str(i)+'_ss_filtre']=pd.concat([df_transit_airesA63,
                                            dico_transit_avec_marge['df_transit_pas_airesA63_marge'+str(i)]],sort=False)
        dico_transit_avec_marge['df_transit_marge'+str(i)+'_ss_filtre'].correction_o_d=dico_transit_avec_marge['df_transit_marge'+str(i)+'_ss_filtre'].correction_o_d.fillna(False)
        dico_transit_avec_marge['df_transit_marge'+str(i)+'_avec_filtre']=(dico_transit_avec_marge['df_transit_marge'+str(i)+'_ss_filtre'].loc[
            dico_transit_avec_marge['df_transit_marge'+str(i)+'_ss_filtre']['filtre_tps']==1]).copy()
    return dico_transit_avec_marge

def correction_A660(dico_transit_avec_marge,df_passages_immat_ok,liste_marges):
    """
    appliquer la corrction A660 pour un ensemble de marge
    en entree : 
        dico_transit_avec_marge : df des trajets en transit (filtre_tps = 1) cf appliquer_marge
        df_passages_immat_ok :df des passages, cf definir_transit
        liste_marges : list ed'entier >0 correspondant aux marge en minute
    en sortie : 
        dico_corr_A63_A660 : df des trajets de transit, avec certaines o_d odifiees
    """
    dico_corr_A63_A660={'corr_A63_A660'+str(i):correction_trajet(df_passages_immat_ok, dico_transit_avec_marge ['df_transit_marge'+str(i)+'_avec_filtre'])
                    for i  in liste_marges}
    return dico_corr_A63_A660

def extrapol_trajets_incomplets(dico_transit_avec_marge,df_passages_immat_ok,dico_corr_A63_A660,liste_marges, dixco_tpsmax_corrige):
    """
    extrapolation des trajets de trasit pour des véhicules deja vues sur des trajets complets, pour un ensemble de marge
    """
    dico_df_od_final={}
    # dico des df des passages avant correction A660
    dico_passag_avantcorr={'passag_avantcorr'+str(i):
               trajet2passage(dico_transit_avec_marge ['df_transit_marge'+str(i)+'_avec_filtre'],df_passages_immat_ok) 
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