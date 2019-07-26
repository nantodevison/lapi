# -*- coding: utf-8 -*-
'''
Created on 1 juil. 2019

@author: martin.schoreisz
Moduel avec des fonctions pour traiter les resultats : import / exports en json, pourcentage PL par acm, passage fictif rocade...
'''
import pandas as pd
from Import_Forme import matrice_nb_jo_sup_31, matrice_nb_jo, matrice_nb_jo_inf_31, dico_correspondance, donnees_horaire



def pourcentage_pl_camera(df_pl,dico_passag):
    """
    fonction de regroupement des nb de vl, pl, et pl en trasit, par heure et par camera
    en entree : 
        df_pl : df des passages pl
        dico_passag : dico des passages PL de transit
        df_vl : df des passages vl
    en sortie : 

    """
    def pct_pl(a,b):
            try :
                return round(a*100/b)
            except ZeroDivisionError : 
                return 0
    
        
    df_synthese_pl_tot=df_pl.groupby('camera_id').resample('H').count()['immat'].reset_index().rename(columns={'immat':'nb_veh'})
    df_synthese_pl_transit=dico_passag.set_index('created').groupby('camera_id').resample('H').count()['immat'].reset_index().rename(
            columns={'immat':'nb_veh'})
    df_synthese_pl_tot['heure']=df_synthese_pl_tot.created.dt.hour
    df_synthese_pl_transit['heure']=df_synthese_pl_transit.created.dt.hour
    df_synthese_pl_tot['type']='PL total'
    df_synthese_pl_transit['type']='PL transit'
    df_concat_pl=pd.concat([df_synthese_pl_tot,df_synthese_pl_transit],sort=False)
    df_concat_pl_jo=df_concat_pl.loc[df_concat_pl.set_index('created').index.dayofweek < 5].copy()
    df_concat_pl_jo=df_concat_pl_jo.groupby(['camera_id', 'type','heure']).mean().reset_index()
    df_pct_pl_transit=df_concat_pl_jo.loc[df_concat_pl_jo['type']=='PL total'].merge(df_concat_pl_jo.loc[df_concat_pl_jo['type']=='PL transit'],on=['camera_id','heure'])
    df_pct_pl_transit['pct_pl_transit']=df_pct_pl_transit.apply(lambda x : pct_pl(x['nb_veh_y'],x['nb_veh_x']) ,axis=1)
    return df_concat_pl_jo,df_pct_pl_transit

def ajout_cam_n10(df_concat_pl,df_pct_pl_transit ) : 
    """
    ajouter les camera 20 et 21 pour représenter les variations horaires sur N10 seules
    """
    jointure_cross=df_pct_pl_transit.merge(df_pct_pl_transit, on='heure')
    jointure_cross_5_11=jointure_cross.loc[(jointure_cross['camera_id_x']==5)&(jointure_cross['camera_id_y']==11)]
    cam20=pd.DataFrame({'camera_id':[20]*24,'type_x':['PL total']*24,'heure':[a for a in range(24)], 
                  'nb_veh_x':(jointure_cross_5_11.nb_veh_x_x-jointure_cross_5_11.nb_veh_x_y).tolist(),
                    'type_y':['PL transit']*24,
                  'nb_veh_y':(jointure_cross_5_11.nb_veh_y_x-jointure_cross_5_11.nb_veh_y_y).tolist()})
    cam20['pct_pl_transit']=cam20.nb_veh_y/cam20.nb_veh_x*100
    jointure_cross_6_12=jointure_cross.loc[(jointure_cross['camera_id_x']==6)&(jointure_cross['camera_id_y']==12)]
    cam21=pd.DataFrame({'camera_id':[21]*24,'type_x':['PL total']*24,'heure':[a for a in range(24)], 
                  'nb_veh_x':(jointure_cross_6_12.nb_veh_x_x-jointure_cross_6_12.nb_veh_x_y).tolist(),
                    'type_y':['PL transit']*24,
                  'nb_veh_y':(jointure_cross_6_12.nb_veh_y_x-jointure_cross_6_12.nb_veh_y_y).tolist()})
    cam21['pct_pl_transit']=cam21.nb_veh_y/cam21.nb_veh_x*100
    df_pct_pl_transit_f=pd.concat([df_pct_pl_transit,cam20,cam21], axis=0, sort=False)
    df_concat_pl_f=pd.concat([df_concat_pl,
                    pd.concat([pd.concat([cam20[['camera_id','type_x', 'heure','nb_veh_x']].rename(columns={'type_x':'type','nb_veh_x':'nb_veh'}),
                    cam20[['camera_id','type_y', 'heure','nb_veh_y']].rename(columns={'type_y':'type','nb_veh_y':'nb_veh'})],axis=0, sort=False),
                    pd.concat([cam21[['camera_id','type_x', 'heure','nb_veh_x']].rename(columns={'type_x':'type','nb_veh_x':'nb_veh'}),
                    cam21[['camera_id','type_y', 'heure','nb_veh_y']].rename(columns={'type_y':'type','nb_veh_y':'nb_veh'})],axis=0, sort=False)], axis=0, 
                              sort=False)],axis=0, sort=False)
    return df_concat_pl_f, df_pct_pl_transit_f


def repartition_rocade_od(df_od_final):
    """
    repartition du pourcentage de PL par Rocade Est ou Ouest selon l'OD
    en entree : 
        df_od_final : df des trajets de transit
    en sortie : 
        repartition_pl_rocade_mjo : df des pourcentage par o_d
    """
    #Composer les données
    #répartitionRocade Est / Rocade Ouest
    #trouver les passages concernés
    passages_rocades=df_transit_propre_jo(df_od_final).loc[
        df_transit_propre_jo(df_od_final).apply(lambda x : any([a in x['cameras'] for a in [1,2,3,4]]),axis=1)]
    #affectation du site de Rocade
    passages_rocades['rocade_site']=passages_rocades.apply(lambda x : 'Ouest' if any([a in x['cameras'] for a in [1,2]]) else 'Est',axis=1 )
    #regroupement, moyenne par jour ouvre et jointure avec pct_pl
    passg_rocade_grp_tot_jo=(passages_rocades.groupby('o_d')['immat'].count()/10).reset_index().rename(columns={'immat':'nb_pl_tot'})
    repartition_pl_rocade_mjo=(passages_rocades.groupby(['o_d','rocade_site'])['immat'].count()/10).reset_index().rename(columns={'immat':'nb_pl_rocade'}).merge(
    passg_rocade_grp_tot_jo)
    repartition_pl_rocade_mjo['pct_pl']=round(repartition_pl_rocade_mjo.nb_pl_rocade*100/repartition_pl_rocade_mjo.nb_pl_tot)
    return repartition_pl_rocade_mjo

def donnees_sankey(df_apres31):
    """
    donnees pour les graphs de Sankey
    en entree : 
        df_apres31 : df des trajets de transit apres le 31/01
    en sortie : 
        sankey_rocade_f_s_n : df pour sankey, uniquement les trajets concernes par la separation en Rocade, sens S->N
        sankey_rocade_f_n_s : df pour sankey, uniquement les trajets concernes par la separation en Rocade, sens n->S
        sankey_direct_tot : df pour sankey, uniquement les trajets non concernes par Rocade
        df_sankey : df pour sankey, sans prise en compte Rocade
    """
    #sankey avec plotly, nécessite de rédémarrer jupyter notebook
    mat_jo_apre_31=matrice_transit(df_transit_propre_jo(df_apres31),type_j='jo_sup_31')
    df_sankey=mat_jo_apre_31.unstack().reset_index().sort_values(['origine','destination']).dropna().reset_index().drop('index',axis=1)
    df_sankey.columns=['destination', 'origine', 'nb_pl']
    df_sankey['o_d']=df_sankey.origine+'-'+df_sankey.destination
    
    repartition_pl_rocade_mjo=repartition_rocade_od(df_apres31)
    
    #on rappatrie sur la base du graph de sankey
    sankey_rocade=df_sankey.merge(repartition_pl_rocade_mjo[['o_d','pct_pl','rocade_site']],on='o_d').sort_values('o_d')
    #on isole les trajets qui ne se séparent pas à la rocade des autres
    sankey_direct=df_sankey.loc[~df_sankey.o_d.isin(sankey_rocade.o_d.tolist())]
    sankey_direct_tot=pd.concat([sankey_direct,sankey_rocade.loc[sankey_rocade['pct_pl']==100][['destination', 'origine', 'nb_pl','o_d']]],sort=False)
    sankey_rocade=sankey_rocade.loc[sankey_rocade['pct_pl']!=100].copy()
    
    #traitement de a partie entre origine et rocade
    sankey_rocade_org=sankey_rocade.copy()
    sankey_rocade_org['nb_pl']=sankey_rocade_org.apply(lambda x : x['nb_pl']*x['pct_pl']*0.01 if x['pct_pl']!=100 else x['nb_pl'],axis=1)
    sankey_rocade_org['destination']='Rocade '+sankey_rocade_org.rocade_site
    #traitement de la partie entre Rocade et destination
    sankey_rocade_dest=sankey_rocade_org.copy()
    sankey_rocade_dest['origine']=sankey_rocade_dest.destination
    sankey_rocade_dest['destination']=sankey_rocade_dest.apply(lambda x : x['o_d'].split('-')[1],axis=1)
    
    #fusion des résultats
    sankey_rocade_f=pd.concat([sankey_rocade_org,sankey_rocade_dest], axis=0, sort=False)[['destination', 'origine', 'nb_pl','o_d']]
    sankey_rocade_f=sankey_rocade_f.loc[sankey_rocade_f.apply(lambda x : 'A660' not in x['o_d'],axis=1)]
    
    
    #resultats (direct ou par sens si Rocade)
    sankey_rocade_f_s_n=sankey_rocade_f.loc[(~sankey_rocade_f['origine'].isin(['A10','N10']))&(sankey_rocade_f['destination']!='A63')].copy()
    sankey_rocade_f_n_s=sankey_rocade_f.loc[(~sankey_rocade_f['origine'].isin(['A63']))&(~sankey_rocade_f['destination'].isin(['A10','N10']))].copy()
    sankey_direct_tot=sankey_direct_tot.loc[sankey_direct_tot.apply(lambda x : 'A660' not in x['o_d'],axis=1)]
    df_sankey=df_sankey.loc[df_sankey.apply(lambda x : 'A660' not in x['o_d'],axis=1)]  
    return sankey_rocade_f_s_n, sankey_rocade_f_n_s, sankey_direct_tot, df_sankey

def indice_confiance_cam(df_pct_pl_transit,df_concat_pl_jo,cam):
    """
    df pour créer le graph des intervalles de confiance
    """
    if len(cam)>1 :
        traf_dira_rocade_cam=donnees_horaire.loc[donnees_horaire['camera'].isin(cam)].groupby('heure').agg(
            {'nb_pl':'sum','nb_pl_total':'sum','nb_tv':'sum'}).reset_index()
        df_pct_pl_transit_cam=df_pct_pl_transit.loc[df_pct_pl_transit['camera_id'].isin(cam)].groupby(['heure']).agg({'nb_veh_x':'sum','nb_veh_y':'sum'}).reset_index()
        df_pct_pl_transit_cam['pct_pl_transit']=df_pct_pl_transit_cam['nb_veh_y']/df_pct_pl_transit_cam['nb_veh_x']*100
        df_concat_pl_jo_cam=df_concat_pl_jo.loc[df_concat_pl_jo['camera_id'].isin(cam)&(df_concat_pl_jo['type']=='PL total')].groupby('heure')['nb_veh'].sum().reset_index()
        df_concat_pl_jo_cam['type']='PL total'
    else : 
        traf_dira_rocade_cam=donnees_horaire.loc[donnees_horaire['camera'].isin(cam)].copy()
        df_concat_pl_jo_cam=df_concat_pl_jo.loc[(df_concat_pl_jo['camera_id'].isin(cam))&(df_concat_pl_jo['type']=='PL total')].copy()
        df_pct_pl_transit_cam=df_pct_pl_transit.loc[df_pct_pl_transit['camera_id'].isin(cam)].copy()
    
    #jointure des données
    lien_traf_gest_traf_lapi=traf_dira_rocade_cam.merge(df_concat_pl_jo_cam, on='heure')
    lien_traf_gest_traf_lapi=lien_traf_gest_traf_lapi.merge(df_pct_pl_transit_cam[['heure','pct_pl_transit']], on='heure')
    
    #création des attributs
    lien_traf_gest_traf_lapi.rename(columns={'nb_pl':'nb_pl_siredo','nb_veh':'nb_pl_lapi','nb_pl_total':'nb_pl_siredo_total','nb_tv':'nb_tv_siredo'},inplace=True)
    lien_traf_gest_traf_lapi['nb_pl_transit_lapi']=lien_traf_gest_traf_lapi['nb_pl_lapi']*lien_traf_gest_traf_lapi['pct_pl_transit']*0.01
    lien_traf_gest_traf_lapi['pct_detec_lapi']=lien_traf_gest_traf_lapi['nb_pl_lapi']/lien_traf_gest_traf_lapi['nb_pl_siredo']
    lien_traf_gest_traf_lapi['nb_pl_lapi_recale']=lien_traf_gest_traf_lapi['nb_pl_siredo']*(1-lien_traf_gest_traf_lapi['pct_detec_lapi'])
    lien_traf_gest_traf_lapi['pct_pl_transit_max']=((lien_traf_gest_traf_lapi['nb_pl_transit_lapi']+lien_traf_gest_traf_lapi['nb_pl_lapi_recale']) / 
                                                    (lien_traf_gest_traf_lapi['nb_pl_lapi']+lien_traf_gest_traf_lapi['nb_pl_lapi_recale']))*100
    lien_traf_gest_traf_lapi['pct_pl_transit_min']=(lien_traf_gest_traf_lapi['nb_pl_transit_lapi']/ 
                                                    (lien_traf_gest_traf_lapi['nb_pl_lapi']+lien_traf_gest_traf_lapi['nb_pl_lapi_recale']))*100
                                                    
    #graphique
    pour_graph_synth_pl_lapi=df_concat_pl_jo_cam[['heure','nb_veh','type']].copy()
    pour_graph_synth_pl_lapi=pour_graph_synth_pl_lapi.loc[pour_graph_synth_pl_lapi['type']=='PL total'].copy()
    pour_graph_synth_pl_lapi['type']='LAPI'
    pour_graph_synth_pl_siredo_recale=traf_dira_rocade_cam[['heure', 'nb_pl']].rename(columns={'nb_pl':'nb_veh'}).copy()
    pour_graph_synth_pl_siredo_recale['type']='SIREDO recale'
    pour_graph_synth_pl_siredo_brut=traf_dira_rocade_cam[['heure', 'nb_pl_total']].rename(columns={'nb_pl_total':'nb_veh'}).copy()
    pour_graph_synth_pl_siredo_brut['type']='SIREDO brut'
    pour_graph_synth=pd.concat([pour_graph_synth_pl_lapi,pour_graph_synth_pl_siredo_recale,pour_graph_synth_pl_siredo_brut],sort=False)
    return  pour_graph_synth, lien_traf_gest_traf_lapi                      
                               
    
def PL_transit_dir_jo_cam(df_pct_pl_transit, cam):
    """
    graph de synthese du nombre de pl en trasit par heure. Base nb pl dir et pct_pl_transit lapi
    en entree : 
        df_pct_pl_transit : df du pct de pl en transit, issus de resultat.pourcentage_pl_camera
        cam : list integer : numeros de la camera etudiee. on peut en passer plsueiurs et obtenir une somme des nb veh et une moyenne des %P
    en sortie : 
        concat_dir_trafic : df avec heure, nb_pl et type_pl, par jour ouvre sur le(s) cameras desirees
    """
    if len(cam)>1 :
    #si le nombre de camera est sup à 1, il faut recalculer le %pl_transit pour l'ensemble des deux cameras, avant de l'appliquer à la somme des deux de la dir
        df_pct_pl_transit_multi_cam=df_pct_pl_transit.loc[df_pct_pl_transit['camera_id'].isin(cam)].groupby(['heure']).agg(
            {'nb_veh_x':'sum','nb_veh_y':'sum'}).reset_index()
        df_pct_pl_transit_multi_cam['pct_pl_transit']=df_pct_pl_transit_multi_cam['nb_veh_y']/df_pct_pl_transit_multi_cam['nb_veh_x']*100
        traf_dira_rocade_grp=donnees_horaire.loc[donnees_horaire['camera'].isin(cam)].groupby('heure').agg(
                {'nb_pl':'sum','nb_pl_total':'sum','nb_tv':'sum'}).reset_index()
        dira_pct_pl_lapi=traf_dira_rocade_grp.merge(df_pct_pl_transit_multi_cam, on=['heure'])
        dira_pct_pl_lapi['nb_pl_transit']=dira_pct_pl_lapi.nb_pl*dira_pct_pl_lapi.pct_pl_transit*0.01                             
    else :
        df_pct_pl_transit_multi_cam=df_pct_pl_transit.loc[df_pct_pl_transit['camera_id'].isin(cam)] 
        dira_pct_pl_lapi=donnees_horaire.loc[donnees_horaire['camera'].isin(cam)].merge(
            df_pct_pl_transit,left_on=['camera','heure'], right_on=['camera_id','heure'])
        dira_pct_pl_lapi['nb_pl_transit']=dira_pct_pl_lapi.nb_pl*dira_pct_pl_lapi.pct_pl_transit*0.01
    diratotal,diratransit, dira_tv=(dira_pct_pl_lapi[['heure','nb_pl']].copy(),
                                    dira_pct_pl_lapi[['heure','nb_pl_transit']].rename(columns={'nb_pl_transit':'nb_pl'}).copy(),
                                    dira_pct_pl_lapi[['heure','nb_tv']].copy().rename(columns={'nb_tv':'nb_pl'}))
    diratotal['type']='Tous PL'
    diratransit['type']='PL en transit'
    dira_tv['type']='Tous Vehicules'
    concat_dir_trafic=pd.concat([diratotal,diratransit,dira_tv], axis=0, sort=False)
    return concat_dir_trafic, df_pct_pl_transit_multi_cam

def passages_fictif_rocade (liste_trajet, df_od,df_passages_transit,df_pl):
    """
    Cr�er des passages pour les trajets de transit non vus sur la Rocade mais qui y sont pass�
    en entree : 
        liste_trajet : df des trajets concernes, issu de liste_trajet_rocade
        df_od : df des trajetsde transit valid� selon le temps de parcours
        df_passages_transit : df des passages concern�s par un trajet de transit (issu du traitement o_d)
        df_pl : df de tout passages pl (issu simplement de l'import mise en forme)
    en sortie : 
        df_passage_transit_redresse : df des passages concern�s par un trajet de transit (issu du traitement o_d) + passages fictifs
        df_pl_redresse : df de tout passages pl + passages fictifs
        trajets_rocade_non_vu : df des passgaes fictifs
    """
    def camera_fictive(origine, destination) : 
        """
        Connaitre la camera a affectee selon le trajet parcouru
        """
        if origine in ['A63','A62','A660'] and destination in ['N10','A10','A89'] : 
            return 4
        elif origine in ['A10','A89','N10'] and destination in ['A63','A62','A660'] :
            return 3
        else : 
            return -1
    #rechercher les trajets dans le dico des o_d
    trajets_rocade=df_od.loc[df_od.o_d.isin(liste_trajet.trajets.tolist())]
    #trouver ceux qui ne contiennent aucune r�f�rence uax camera de la Rocade
    trajets_rocade_non_vu=trajets_rocade.loc[trajets_rocade.apply(lambda x : all(e not in x['cameras'] for e in [1,2,3,4]),axis=1)].copy()
    #cr�er des passage fictif au niveau de la Rocade avec comme created la moyenne entre date_cam_1 et date_cam_2
    trajets_rocade_non_vu['created_fictif']=trajets_rocade_non_vu.apply(lambda x : x['date_cam_1']+((x['date_cam_2']-x['date_cam_1'])/2),axis=1)
    trajets_rocade_non_vu['camera_fictif']=trajets_rocade_non_vu.apply(lambda x : camera_fictive(x['origine'],x['destination']),axis=1)
    #virere clolonne inutiles
    trajets_rocade_non_vu=trajets_rocade_non_vu.drop(['date_cam_1', 'id', 'date_cam_2',
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

def passage_fictif_od(df_od,df_passage_transit_redresse,df_passages_immat_ok,dico_correspondance=dico_correspondance):
    """
    creerdes passages fictifs pour les trajets incomplets ou les trajets de A660 corrigés vers A63
    la date de reference du passage correspond à l'heure en cours
    en entree : 
        dico_correspondance : liste de liste qui contiennent : le type de camera (origine ou destination), le nom de l'origine ou destination, le camera_id
        df_od : df des trajetsde transit valide selon le temps de parcours
        df_passage_transit_redresse : df des passages concernes par un trajet de transit (issu du passages_fictif_rocade)
        df_passages_immat_ok : df de tous les passages PL (transit ou non)
    en sortie : 
        df_passage_trsnit_final : df_passage_transit_redresse avec ajout des passages fictif o_d
    """
    passage_redress=df_passage_transit_redresse.copy()
    passage_total = df_passages_immat_ok.copy()
    for i,params in enumerate(dico_correspondance) :
        df_filtre=df_od.loc[(df_od[params[0]]==params[1])].copy()
        trajets_rocade_non_vu=df_filtre.loc[df_filtre.apply(lambda x : params[2] not in x['cameras'],axis=1)].copy()
        if trajets_rocade_non_vu.empty:
            continue
        trajets_rocade_non_vu['created_fictif']=trajets_rocade_non_vu.apply(lambda x : x['date_cam_1'].floor('H'), axis=1)
        trajets_rocade_non_vu['camera_fictif']=params[2]
        trajets_rocade_non_vu.drop(['date_cam_1', 'id', 'date_cam_2','chiffree',
               'cameras', 'origine', 'destination', 'o_d', 'tps_parcours', 'period',
               'date', 'temps', 'type', 'tps_parcours_theoriq', 'filtre_tps'],axis=1,inplace=True)
        trajets_rocade_non_vu.rename(columns={'created_fictif':'created','camera_fictif':'camera_id'},inplace=True)
        trajets_rocade_non_vu['fictif']=params[0]
        trajets_rocade_non_vu['fiability']=999
        passage_transit_redress=pd.concat([trajets_rocade_non_vu,passage_redress]
                                        ,axis=0,sort=False) if i==0 else pd.concat([trajets_rocade_non_vu,passage_transit_redress],axis=0,sort=False)
        passages_tot_redresse=(pd.concat([trajets_rocade_non_vu.set_index('created'),passage_total],axis=0,sort=False) if i==0 else
                               pd.concat([trajets_rocade_non_vu.set_index('created'),passages_tot_redresse],axis=0,sort=False))
    return passage_transit_redress, passages_tot_redresse

def save_donnees(df, fichier):
    """
    sauvegarder les df dans des fichiers.
    en entree : 
        df : la dtatfarme � sauvegarder
        fichier : raw string du nom de fichier
    """
    "reset index si c'est un datetime[ns]"
    if df.index.dtype=='<M8[ns]' : 
        fichier_export=df.reset_index().copy()
    else : 
        fichier_export=df.copy()
    if not fichier_export.index.is_unique :
        fichier_export.reset_index(inplace=True)
    #suppr les colonnes qui servent   rine si elles existent
    if 'index' in fichier_export.columns :
        fichier_export.drop('index',axis=1, inplace=True)
    if 'level_0' in fichier_export.columns :
        fichier_export.drop('level_0',axis=1, inplace=True)
    #passer les datetime et texte lisible par pd.to_datetime
    list_attr_datetime=[attr for attr in fichier_export.columns if fichier_export[attr].dtypes in ['<m8[ns]','<M8[ns]']]
    for attr in list_attr_datetime : 
        fichier_export[attr]=fichier_export[attr].apply(lambda x : str(x))
    
    fichier_export.to_json(fichier, orient='index')
        
def ouvrir_donnees(fichier):
    """
    ouvrir les donées suvegradees avec  save_donnees 
    en entree : 
        fichier : raw string du fichier à ouvrir
    en sortie : 
        df : dataframe avec les attributs aux formats datetime et timedelta
    """
    df=pd.read_json(fichier,orient='index')
    for attr in ['date_cam_1','date_cam_2','tps_parcours','date','temps','tps_parcours_theoriq','temps_filtre','created'] : 
        if attr in df.columns : 
            if attr in ['date_cam_1','date_cam_2','date','created'] : 
                df[attr]=df[attr].apply(lambda x : pd.to_datetime(x))
            else :
                df[attr]=df[attr].apply(lambda x : pd.Timedelta(x))
    
    return df   
        
def decoupe_df_avant_apres_31(df_transit):
    """
    obtrnir2 df à partir de la df_od_final : la df des trajets avant le 31/01, la df des trajets après le 31/01
    en entree : 
        df_transit : df finales des PL en transit (o_d, pas passages)
    en sortie 
        df_transit
        df_avant31 : df des PL en transit avant le 31/01
        df_apres31 : la df des PL en transit à partirdu 31
    """
    df_avant31=df_transit.loc[df_transit['date_cam_1']<pd.to_datetime('2019-01-31 00:00:00')]
    df_apres31=df_transit.loc[df_transit['date_cam_1']>pd.to_datetime('2019-01-30 23:59:59')]
    return df_transit, df_avant31, df_apres31

def filtrer_jour_non_complet(df_transit):
    """
    Enlever les trajets ayant eu lieu lors de jours non complet pour obtenir des resultats cohrents sur le tmjo ou tmja
    AMELIORABLE si on pouvait passer un dico avec comme cle la camera et comme value une liste avec la date et si la cam est au debuit ou fin
    en entree : 
        df_transit : df finales des PL en transit (o_d, pas passages)
    en sortie : 
        df_transit_propre : la df_transit sans les trajets sur les dates à enlever
    """
    df_propre=df_transit.loc[df_transit.apply(lambda x : 
        not (x['origine']=='N10' and x['date_cam_1'].day==pd.to_datetime('2019-01-29').day) and
        not (x['origine']=='N10' and x['date_cam_1'].day==pd.to_datetime('2019-01-30').day) and 
        not (x['destination']=='A10' and x['date_cam_1'].day==pd.to_datetime('2019-02-01').day) and 
        not (x['destination']=='A62' and x['date_cam_1'].day==pd.to_datetime('2019-01-24').day) and
        not (x['origine']=='A63' and x['date_cam_1'].day==pd.to_datetime('2019-01-31').day) and 
        not (x['destination']=='A89' and x['date_cam_1'].day==pd.to_datetime('2019-02-06').day) and 
        not (x['destination']=='A63' and x['date_cam_1'].day==pd.to_datetime('2019-02-01').day) and
        not (x['origine']=='A89' and x['date_cam_1'].day==pd.to_datetime('2019-02-06').day),axis=1)]
    df_filtree=df_transit.loc[~df_transit.set_index(['date_cam_1','immat']).index.isin(df_propre.set_index(['date_cam_1','immat']).index.to_list())]
    
    return df_propre, df_filtree
        
def df_transit_propre_jo(df_transit_propre, type_j='jo'):    
    """ 
    filtre de la matrcie cree par  filtrer_jour_non_complet selon les jours ouvresou autre
    """
    if type_j=='jo' :
        df_transit_filtre=df_transit_propre.loc[df_transit_propre.set_index('date_cam_1').index.dayofweek<5]
    
    return df_transit_filtre

def matrice_transit(df_transit_filtre, type_j='jo_sup_31'):
    """
    crée les matrice de nb de PL selon le type de jours
    """
    if type_j=='jo_sup_31' :
        return round(pd.pivot_table(df_transit_filtre,values='l', index='origine', columns='destination',aggfunc='count')/matrice_nb_jo_sup_31,0)
    elif type_j=='jo' : 
        return round(pd.pivot_table(df_transit_filtre,values='l', index='origine', columns='destination',aggfunc='count')/matrice_nb_jo,0)
    elif  type_j=='jo_inf_31' :
        return round(pd.pivot_table(df_transit_filtre,values='l', index='origine', columns='destination',aggfunc='count')/matrice_nb_jo_inf_31,0)
        
