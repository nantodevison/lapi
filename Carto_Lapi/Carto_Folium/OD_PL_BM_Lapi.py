'''
Created on 8 oct. 2018

@author: Martin
'''

import folium
import pathlib as p
from Martin_Perso import Ogr_Perso, Connexion_Transfert as conn
from osgeo import ogr, osr
"""
#CREER UN GEOJSON DEPUIS UN SHAPE AVEC OGR (donnees exemple sur Lyon)
#definir la reprojection
epsg4326 = osr.SpatialReference() 
epsg2154 = osr.SpatialReference()
epsg4326.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
epsg2154.ImportFromProj4('+proj=lcc +lat_1=49 +lat_2=44 +lat_0=46.5 +lon_0=3 +x_0=700000 +y_0=6600000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs')
transform = osr.CoordinateTransformation(epsg2154, epsg4326)
#recuperer les données de bases
fichier=Ogr_Perso.DonneesShapefile(r'E:\Boulot\rapportage\sortie_CeremaEst\DF1548_069_E3.shp')
# Create the output Driver
outDriver = ogr.GetDriverByName('GeoJSON')
outDataSource = outDriver.CreateDataSource(r'E:\Boulot\python3\folium\testjson.geojson')
outLayer = outDataSource.CreateLayer(r'E:\Boulot\python3\folium\testjson.geojson',geom_type=ogr.wkbMultiLineString)
#creer les champs dans le transparent d'arrivee
for i in range (0, fichier.layerDef.GetFieldCount()):
    fieldDefn = fichier.layerDef.GetFieldDefn(i)
    outLayer.CreateField(fieldDefn)
outLayerDefn=outLayer.GetLayerDefn()
#ajouter les features
for feat in fichier.layer :
    outFeature = ogr.Feature(outLayerDefn)
    #ajouter les attributs
    for i in range(0, outLayerDefn.GetFieldCount()):
        fieldDefn = outLayerDefn.GetFieldDefn(i)
        outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(),feat.GetField(i))
    # Set geometry as centroid
    geom = feat.GetGeometryRef()
    try : 
        geom.Transform(transform)
    except AttributeError :
        pass
    outFeature.SetGeometry(geom)
    # Add new feature to output Layer
    outLayer.CreateFeature(outFeature)    
outDataSource=None
"""

#CREER UNE CARTE CENTREE SUR LYON AVEC LES VOIES PRECEDENTES
#repertoire de stockage des données
dossier_source=p.PurePath(r'E:\Boulot\python3\folium')
#fcihier geojson de lignes 
fichier_geojson='testjson.geojson'
fichierLigne=str(dossier_source.joinpath(fichier_geojson))
#fichier resultat
nom_fichier_carte='test1.html'
fichiersave=str(dossier_source.joinpath(nom_fichier_carte))

#initialiser la carte sur Bordeaux
carte=folium.Map(location=[44.420311, -0.820133])
"""
#passer le Geojson
folium.GeoJson(
    fichierLigne,
    name='ligne_test'
).add_to(carte)

#afficher un controle des couches
folium.LayerControl().add_to(carte)
"""





connexion_bdd=conn.ConnexionBdd('lapi')
connexion_bdd.creerConnexion()
curseur=connexion_bdd.curs
curseur.execute("WITH decomp_multi AS (SELECT id,st_transform((St_dump(geom)).geom,4326) AS geom, (St_dump(geom)).path AS path1 FROM public.courbe_n230_iter5),points AS ( SELECT id,((St_dumpPoints(geom)).geom), (St_dumpPoints(geom)).path, path1 FROM decomp_multi),coord_points AS ( SELECT id,round(st_x(geom)::numeric,6)::double precision AS x, round(st_y(geom)::numeric,6)::double precision AS y, path, path1 FROM points),groupe As ( SELECT ARRAY[y,x] as point , id, path, path1 FROM coord_points)select array_agg(point), id from groupe group by id")
for a in curseur :
    print ([points for points in a[0]], a[1])
    if a[1] == 1:
        folium.PolyLine([points for points in a[0]], color="blue", weight=20, opacity=1).add_to(carte)
    else :
        folium.PolyLine([points for points in a[0]], color="red", weight=20, opacity=1).add_to(carte)
folium.RegularPolygonMarker(location=(44.420311, -0.820133), fill_color='blue', number_of_sides=3, radius=10, rotation=-90).add_to(carte)

carte.save(fichiersave)
