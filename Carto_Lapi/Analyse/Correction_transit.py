# -*- coding: utf-8 -*-
'''
Created on 21 juin 2019

@author: martin.schoreisz

Module d'affinage des donn�es calcul�e dans le module transit
'''

from transit import cam_voisines, trajet2passage
from Import_Forme import liste_complete_trajet
import pandas as pd
import numpy as np
from sklearn import svm

#liste des o_d ok pour predire_ts_trajets
liste_od_ok=['A660-A62','A62-A63','A63-A62','A660-N10', 'A660-A10','N10-A63']

def correction_trajet(df_3semaines, dico_od, voie_ref='A660', cam_ref_1=13, cam_ref_2=15, cam_ref_3=19) : 
    """
    Fonction qui va re-assigner les origines-destinations � A63 si certanes conditions sont remplie : 
    cas 1 : vue a A660 puis dans l'autre sens sur A63, ou inversement
    cas 2 : vue sur A660 Nord-Sud, puis A660 Sud-Nord, avec plus de 1jd'�cart entre les deux
    en entree : 
        df_3semaines : dataframe des passages
        dico_od : dataframe des o_d issue de transit_temps_complet
        voie_ref : string : nom de la voie que l'on souhaite changer
        cam_ref_1 : integer : camera de a changer pour le sens 1 (cas 1)
        cam_ref_2 : integer : camera de a changer pour le sens 2 (cas 1)
        cam_ref_3 : integer camera a changer dans les deux sens (cas2)
    en sortie : 
        dico_od_origine : dataframe des o_d issue de transit_temps_complet compl�t�e et modif�e
    """
    
    def MaJ_o_d(correctionType, o, d):
        """
        Fonction de mise � jour des o_d pour les trajets concernants A660 que l'on rabat sur A63
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
    dico_od_copie=dico_od.loc[(dico_od['origine']==voie_ref) | (dico_od['destination']==voie_ref)].reset_index().copy() #isoler les o_d li�es au points en question
    df_immats=df_3semaines.loc[df_3semaines.immat.isin(dico_od_copie.immat.unique().tolist())] #limiter le df_3semaines aux immats concern�e   df_adj=dico_od_copie.apply(lambda x : t.cam_adjacente(x['immat'],x['date_cam_1'],x['date_cam_2'],x['o_d'],df_immats),axis=1, result_type='expand') #construire les colonnes de camera adjacente et de temps adjacent 
    #on travaille le trajet : date_cam_1 est relatif � l'origine. si l'origine est A660, alors ce qui nous interesse est le passage pr�c�dent
    df_adj_cam1=dico_od_copie.apply(lambda x : cam_voisines(x['immat'],x['date_cam_1'],x['cameras'][0],df_immats),axis=1, result_type='expand')  
    df_adj_cam1.columns=['cam_suivant','date_suivant','cam_precedent1','date_precedent1']
    df_adj_cam1.drop(['cam_suivant','date_suivant'], axis=1, inplace=True)
    #inversement : date_cam_2 est relatif � la destination. si la destination est A660, alors ce qui nous interesse est le passage suivant
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
                       'correction_o_d']=True #mise � jour de l'attribut drapeau
    dico_od_origine.loc[(dico_od_origine['correction_o_d']) &
                                          (dico_od_origine['correction_o_d_type'].isna()),'correction_o_d_type']='correction_A63'
    #mise � jour des  3 attributs li�es aux o_d
    dico_od_origine.loc[dico_od_origine['correction_o_d'],'origine']=dico_od_origine.loc[dico_od_origine['correction_o_d']].apply(lambda x : MaJ_o_d(x['correction_o_d'], x['origine'],x['destination'])[0],axis=1)
    dico_od_origine.loc[dico_od_origine['correction_o_d'],'destination']=dico_od_origine.loc[dico_od_origine['correction_o_d']].apply(lambda x : MaJ_o_d(x['correction_o_d'], x['origine'],x['destination'])[1],axis=1)
    dico_od_origine.loc[dico_od_origine['correction_o_d'],'o_d']=dico_od_origine.loc[dico_od_origine['correction_o_d']].apply(lambda x : MaJ_o_d(x['correction_o_d'], x['origine'],x['destination'])[2],axis=1) 
    
    #cas 2 : passer sur A660 Nord-Sud puis Sud-Nord avec au moins 1 jour d'�cart
    dico_od_copie=dico_od_origine.loc[(dico_od_origine['destination']=='A660')].reset_index().copy()
    
    df_adj_cam2=dico_od_copie.apply(lambda x : cam_voisines(x['immat'],x['date_cam_2'],x['cameras'][-1],df_immats),axis=1, result_type='expand') #construire les colonnes de camera adjacente et de temps adjacent 
    df_adj_cam2.columns=['cam_suivant','date_suivant','cam_precedent','date_precedent']
    df_adj_cam2.drop(['cam_precedent','date_precedent'], axis=1, inplace=True)
    dico_od_copie_adj=pd.concat([dico_od_copie,df_adj_cam1,df_adj_cam2],axis=1)
    dico_od_a_corrige=dico_od_copie_adj.loc[dico_od_copie_adj['cam_suivant']==cam_ref_3].copy()#filtrer les r�sultats sur la cameras de fin
    dico_od_a_corrige['temps_passage']=dico_od_a_corrige['date_suivant']-dico_od_a_corrige['date_cam_2']#calcul du temps de passages entre les cameras
    dico_filtre=dico_od_a_corrige.loc[dico_od_a_corrige['temps_passage']>=pd.Timedelta('1 days')]
    
    #pour les lignes ayant 1 temps de passage sup � 1 jour, on va r�affecter d � A63
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','date_cam_2']).index.isin(
    dico_filtre.set_index(['date_cam_1','immat','date_cam_2']).index.tolist()),'destination']='A63'
    #on modifie o_d aussi
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','date_cam_2']).index.isin(
    dico_filtre.set_index(['date_cam_1','immat','date_cam_2']).index.tolist()),'o_d']=(dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','date_cam_2']).index.isin(
    dico_filtre.set_index(['date_cam_1','immat','date_cam_2']).index.tolist())].apply(lambda x : x['origine']+'-'+x['destination'], axis=1))
    #puis on met � jour correction_o_d
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat','date_cam_2']).index.isin(
    dico_filtre.set_index(['date_cam_1','immat','date_cam_2']).index.tolist()),'correction_o_d']=True
    #pour les lignes ayant 1 temps de passage sup � 1 jour, on va r�affecter o � A63
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat']).index.isin(
    dico_filtre.set_index(['date_suivant','immat']).index.tolist()),'origine']='A63'
    #on modifie o_d aussi
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat']).index.isin(
    dico_filtre.set_index(['date_suivant','immat']).index.tolist()),'o_d']=(dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat']).index.isin(
    dico_filtre.set_index(['date_suivant','immat']).index.tolist())].apply(lambda x : x['origine']+'-'+x['destination'], axis=1))
    #puis on met � jour correction_o_d
    dico_od_origine.loc[dico_od_origine.set_index(['date_cam_1','immat']).index.isin(
    dico_filtre.set_index(['date_suivant','immat']).index.tolist()),'correction_o_d']=True

    return dico_od_origine

def corriger_df_tps_parcours (dico_tps_max):
    """ fonction de correction de la df_tps_parcours issue de transit_temps_complet.
    On moyenne les valuers de temps de parcours de type '85_percentile' si encadrer par des Cluster
    en entree : 
        dico_tps_max issu de transit_temps_complet
    en sortie :
        dico_tps_max modifi�e
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
        df_transit_extra_filtr_tps_modif : df issue de corriger_df_tps_parcour ave le filtre temps selon les temps de de parcours max issu de la prediction
    """    
    period_od_a_modif=df_transit_extrapole.loc[(df_transit_extrapole['type']=='predit')&(df_transit_extrapole['filtre_tps']==1)].sort_values('date_cam_1').copy()
    liste_pour_modif=period_od_a_modif[['date','o_d','temps']].drop_duplicates()#.merge(dixco_tpsmax_corrige, on=['period','o_d'], how='right')
    liste_pour_modif['type']='predit'
    correction_temps=dixco_tpsmax_corrige.merge(liste_pour_modif, on=['o_d','date'], how='left')
    correction_temps['temps_x']=np.where(pd.notnull(correction_temps['temps_y']),correction_temps['temps_y'],correction_temps['temps_x'])
    correction_temps['type_x']=np.where(pd.notnull(correction_temps['type_y']),correction_temps['type_y'],correction_temps['type_x'])
    correction_temps=correction_temps.rename(columns={'temps_x':'temps','type_x':'type'}).drop(['temps_y','type_y'],axis=1)
    jointure_temps=df_transit_extrapole.merge(correction_temps, on=['date','o_d'], how='left')
    jointure_temps['temps_filtre']=np.where(jointure_temps['type_y']=='85eme_percentile',jointure_temps['temps_filtre'],jointure_temps['temps_y'] )
    jointure_temps['type_x']=jointure_temps['type_y']
    jointure_temps=jointure_temps.drop(['temps_y','type_y'],axis=1).rename(columns={'temps_x':'temps', 'type_x':'type'})
    
    return jointure_temps, correction_temps

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
    test_predict['temps_int']=(((pd.to_datetime('2018-01-01')+test_predict.tps_parcours) - pd.Timestamp("1970-01-01")) //
                                pd.Timedelta('1s'))/1000000
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
    df_trajet_extra=df_trajet.loc[(df_trajet['filtre_tps']==1)&(df_trajet['type']=='predit')
                                  &(df_trajet['o_d']==o_d)].copy()
    nw_tps_parcours=df_trajet_extra.groupby(['date','o_d'])['tps_parcours'].max()
    df_trajet=df_trajet.merge(nw_tps_parcours.reset_index(), on=['date','o_d'], suffixes=('_1','_2'), how='left')
    df_trajet['temps']=np.where(pd.notnull(df_trajet['tps_parcours_2']),df_trajet['tps_parcours_2'],df_trajet['temps'])
    df_trajet.drop('tps_parcours_2',axis=1,inplace=True)
    df_trajet.rename(columns={'tps_parcours_1':'tps_parcours'},inplace=True)
    
    return df_trajet

def predire_ts_trajets(df_transit_marge0_ss_extrapolation):
    df_transit_extrapole=df_transit_marge0_ss_extrapolation.copy()
    for od in [x for x in  df_transit_marge0_ss_extrapolation.o_d.unique().tolist() if x not in liste_od_ok ] : 
        print(f'o_d en cours : {od}')
        for date in [a.strftime('%Y-%m-%d') for a in pd.date_range(start='2019-01-23',end='2019-02-13')]:
                try : 
                    df_transit_extrapole=predire_type_trajet(df_transit_extrapole, od,date,600,35)
                except ValueError : 
                    continue
    return df_transit_extrapole
 
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
        df_transit_extrapole.apply(lambda x : (18 in x['cameras'] or 19 in x['cameras']),axis=1))].copy()
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
    df_transit_A63_redresse_tstps=(df_transit_A63_redresse_tpq_theoriq.merge(dixco_tpsmax_corrige, left_on=['o_d_cestas','date'],right_on=['o_d','date'],
        how='left').rename(columns={'cameras_x':'cameras','o_d_x':'o_d','type_x':'type',
                                    'temps_x':'temps','tps_parcours_theoriq_x':'tps_parcours_theoriq',
                                    'temps_y':'temps_cestas','type_y':'type_cestas'}).drop('o_d_y',axis=1))
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

