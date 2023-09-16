import sys
sys.path.append('..')

import geopandas
import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
import math
import dash
import ast
from dash import dcc, html
import plotly.graph_objects as go

from shapely.geometry import Point
from shapely import wkt

from NETX_Functions.GraphOperations import projectPointOnEdge, get_SP_distance, get_SP_distance_TIME
from NETX_Functions.PrintStuff import addGraph, addEdge
from NETX_Functions.MathOperations import computeLengthLinestring

from SETTINGS_Analyses import *


def csvs2graph(path_nodes, path_edges, project = True):
    edges = pd.read_csv(path_edges, sep=";", low_memory = False) #PEMPEM: 14960, OE: 1100, EO: 8000
    nodes = pd.read_csv(path_nodes, sep=";", index_col = 0, low_memory = False) #PEMPEM: 14960, OE: 1100, EO: 8000
    # print(edges.driven.unique())
    
    for SPV in ['Velocity_00', 'Velocity_01', 'Velocity_10', 'Velocity_11', 'Velocity_20', 'Velocity_21', 'Average velocity']:
        edges[SPV] = edges[SPV].replace(np.nan, edges[SPV].mean())
        
    edges['highway'] = edges['highway'].replace(np.nan, "None")
    nodes['geometry'] = nodes['geometry'].apply(wkt.loads)
    edges['geometry'] = edges['geometry'].apply(wkt.loads)

    gdf_nodes = geopandas.GeoDataFrame(nodes, geometry=nodes.geometry, crs="+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +type=crs")
    gdf_edges = geopandas.GeoDataFrame(edges, geometry=edges.geometry, crs="+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +type=crs")
    gdf_edges.set_index(['u', 'v', 'key'], inplace=True)

    # Selection of attributes
    #gdf_nodes = gdf_nodes[['x','y', 'geometry']]
    #gdf_edges = gdf_edges[['geometry', 'u', 'v', 'key', 'length']]
    G = ox.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs = {'crs': "+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +type=crs", 'simplified': True})
    if project: G = ox.projection.project_graph(G)# to_crs="+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +type=crs"
    return G

""" ------------------  Load graph  ------------------ """
G_original = csvs2graph(path_nodes = PATH_TO_GRAPH+"Nodes_ENHANCED.csv", path_edges = PATH_TO_GRAPH+"Edges_ENHANCED.csv", project = False)
used_crs = "+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +type=crs"
""" ------------------------------------------------- """

# Add the traveltimes
for u, v, key, attrs in G_original.edges(data=True, keys=True):
    try:
        attrs['TravelTime_00'] = (attrs['length']/(attrs['Velocity_00']/3.6))/60 #minutes
        attrs['TravelTime_01'] = (attrs['length']/(attrs['Velocity_01']/3.6))/60 #minutes
        attrs['TravelTime_10'] = (attrs['length']/(attrs['Velocity_10']/3.6))/60 #minutes
        attrs['TravelTime_11'] = (attrs['length']/(attrs['Velocity_11']/3.6))/60 #minutes
        attrs['TravelTime_20'] = (attrs['length']/(attrs['Velocity_20']/3.6))/60 #minutes
        attrs['TravelTime_21'] = (attrs['length']/(attrs['Velocity_21']/3.6))/60 #minutes
        attrs['TravelTime_AVG'] = (attrs['length']/(attrs['Average velocity']/3.6))/60 #minutes
    except:
        attrs['TravelTime_00'] = None
        attrs['TravelTime_01'] = None
        attrs['TravelTime_10'] = None
        attrs['TravelTime_11'] = None
        attrs['TravelTime_20'] = None
        attrs['TravelTime_21'] = None
        attrs['TravelTime_AVG'] = None
        

# Note that if you use the csv's without projecting, you need to specify the crs explicitly
print(used_crs)
        
# If you want to view the nodes/edges
nodes, edges = ox.graph_to_gdfs(G_original)
print(nodes)
print(edges)

# Load from/to combinations
fromto_points = pd.read_csv(PATH_TO_FROMTO_COMBINATIONS, sep = ";", index_col = 0, low_memory = False)
# Project point on edge (check if close enough for projection)
from_points = geopandas.GeoDataFrame(fromto_points, geometry = geopandas.points_from_xy(fromto_points.Longitude_from, fromto_points.Latitude_from), crs="EPSG:4326")
from_points = ox.project_gdf(from_points, to_crs = used_crs)
from_points = from_points.geometry
to_points = geopandas.GeoDataFrame(fromto_points, geometry = geopandas.points_from_xy(fromto_points.Longitude_to, fromto_points.Latitude_to), crs="EPSG:4326")
to_points = ox.project_gdf(to_points, to_crs = used_crs)
to_points = to_points.geometry

def elementList(list_with_elements, list_to_check):
    if type(list_with_elements) is list:
        for element in list_with_elements:
            if element in list_to_check: return True
    elif list_with_elements[0] == "[":
        l = ast.literal_eval(list_with_elements)
        for element in l:
            if element in list_to_check: return True
    else: 
        if list_with_elements in list_to_check: return True
    return False

# Pre-process the graph, restrict the graphs to only the driveable roads and project it
used_road_types = ['trunk','primary','secondary','tertiary', 'unclassified', 'residential', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'] # Roads + Road links
EL = [(u,v,k) for u,v,k,d in G_original.edges(keys = True, data=True) if (elementList(d['highway'], used_road_types) | d['driven'])]
G_original = G_original.edge_subgraph(EL)
G_original_projected = ox.project_graph(G_original, to_crs='epsg:4326')  # Project the graph
#EDGES_ORIGINAL = G_original.edges(data=True, keys = True) # All edges of the graph 
#EDGES_ORIGINAL = [(u, v, k, {'geometry': data.get('geometry', None), 'oneway': data.get('oneway', None)}) for u, v, k, data in G_original.edges(keys=True, data=True) if 'geometry' in data and 'oneway' in data]

#SPEED_VARIABLE = "Velocity_00"
EDGES_ORIGINAL = [(u, v, k, data.get('geometry', None), data.get("Velocity_00", None), data.get("Velocity_01", None),data.get("Velocity_10", None), data.get("Velocity_11", None), data.get("Velocity_20", None), data.get("Velocity_21", None), data.get("Average velocity", None)) for u, v, k, data in G_original.edges(keys=True, data=True)]


# print(list(EDGES_ORIGINAL)[0])
# stop

def get_2_nearest_edges(edgs, point, return_dist=False):
    edge_distances = [(edge, point.distance(edge[3])) for edge in edgs]
    edge_distances_sorted = sorted(edge_distances, key = lambda x: x[1])
    return edge_distances_sorted[0:10] # For computation reasons
def plot_path(G, origin_point, destination_point, paths):
    # Create figure
    fig = go.Figure()

    # Draw basemap (OSM)
    fig.update_layout(mapbox1 = dict(center = dict(lon= fromto_points.Longitude_from[0],lat =fromto_points.Latitude_from[0]), accesstoken = mapbox_accesstoken, zoom = 13), margin = dict(t=10, b=0, l=10, r=10),showlegend=False,mapbox_style="light")
    
    # Only print the relevant part of the network
    nodes_used_extended = [[t[0]] + [t[1]] for t in paths]
    nodes_used_extended = [item for sublist in nodes_used_extended for item in sublist]
    
    # Print the graph
    fig = addGraph(fig, G.subgraph(nodes_used_extended), ['red', 'red'], include_existing = True)
    
    # Add all the individual edge parts of the shortest path
    for e in paths:
        e = (e[0], e[1], e[2], ox.projection.project_geometry(e[3], crs=used_crs, to_crs='epsg:4326')[0])
        fig = addEdge(fig, e)
    
    # Print the origin and destination point
    origin_point = ox.projection.project_geometry(Point(origin_point), crs = used_crs,to_latlong=True)[0]
    fig.add_trace(go.Scattermapbox(mode='markers', lat=[origin_point.y], lon=[origin_point.x], visible = True, text = "Origin", marker = {'size' : 15, 'opacity': 1, 'color': 'red', 'allowoverlap': True}))
    destination_point = ox.projection.project_geometry(Point(destination_point), crs = used_crs,to_latlong=True)[0]
    fig.add_trace(go.Scattermapbox(mode='markers', lat=[destination_point.y], lon=[destination_point.x], visible = True, text = "Destination", marker = {'size' : 15, 'opacity': 1, 'color': 'red', 'allowoverlap': True}))
    
    ################################################################ ADD PROJ_POINT OF AN EDGE (RED) ###############################################################
    #proj_point = ox.projection.project_geometry(Point(205734.13828451364, -39024.15897507633), crs = used_crs,to_latlong=True)[0]
    #fig.add_trace(go.Scattermapbox(mode='markers', lat=[proj_point.y], lon=[proj_point.x], visible = True, text = "Projection", marker = {'size' : 15, 'opacity': 1, 'color': 'red', 'allowoverlap': True}))
    ################################################################################################################################################################
    #proj_point = ox.projection.project_geometry(Point(204646.862610691, -39775.039426025236), crs = used_crs,to_latlong=True)[0]
    #fig.add_trace(go.Scattermapbox(mode='markers', lat=[proj_point.y], lon=[proj_point.x], visible = True, text = "Projection", marker = {'size' : 15, 'opacity': 1, 'color': 'red', 'allowoverlap': True}))
    ################################################################################################################################################################
    
    # Add the nodes, because it looks nicer
    fig = addGraph(fig, G.subgraph(nodes_used_extended), ['red', 'red'], include_existing = True, only_nodes = True)
        
    # Launch app
    app = dash.Dash(__name__)
    app.layout = html.Div([html.Div(id = 'fig2', children=[dcc.Graph(id='fig',figure=fig, style={"height" : "95vh"})], style={"height" : "80vh"})], className = "container" )
    if __name__ == '__main__':
        app.run_server(debug=False)
    ################################################################################################################################################################
def determine_possible_fromto_points(edges, point, MAX_PROJECTION, MAX_DISTANCE_OPPOSITE_EDGE):
    # Define the possible from_points 
    possible_points = []
    closestEdges = get_2_nearest_edges(edges, point, return_dist = True)
    minimal_projection_distance_OLD = closestEdges[0][1]
    if closestEdges[0][1] < MAX_PROJECTION:
        possible_points += [((point.x, point.y), closestEdges[0][0])]
    else: return possible_points, closestEdges, minimal_projection_distance_OLD
    if len(closestEdges) == 1: return possible_points, closestEdges, minimal_projection_distance_OLD
    
    i = 1
    while (closestEdges[i][1] < MAX_PROJECTION) & ((closestEdges[i][1] - closestEdges[0][1]) < MAX_DISTANCE_OPPOSITE_EDGE):
        possible_points += [((point.x, point.y), closestEdges[i][0])]
        if i == len(closestEdges)-1: break
        else: i += 1
    return possible_points, closestEdges, minimal_projection_distance_OLD

"""_______________________________________________________________________________________"""
"""_____________________________ CREATE THE ROUTING DATAFRAME ____________________________"""
"""_______________________________________________________________________________________"""
# Loop through all from and to points
df = pd.DataFrame(columns = ['From', 'To', "SP", 'Minimal_projection_distance_from', "From_projected", 'From_projected_distance', 'Minimal_projection_distance_to', "To_projected", 'To_projected_distance', 'tt_00', 'tt_01', 'tt_10', 'tt_11', 'tt_20', 'tt_21', 'tt_AVG', 'tt_00_optimal', 'tt_01_optimal', 'tt_10_optimal', 'tt_11_optimal', 'tt_20_optimal', 'tt_21_optimal', 'tt_AVG_optimal'])
if (SELECTION is not None):
    from_points = from_points[SELECTION]
    to_points = to_points[SELECTION]
idn = 0

for from_point, to_point in zip(from_points, to_points):   
    # Print progress
    # if idn<643:
    #     idn+=1
    #     continue
    # else: print(idn)
    try:
        if idn%(int(len(from_points)/1000))==0: print("Working on point: " + str(idn) + ", " + str(idn/(int(len(from_points)/100)))+"%", end="\r")
    except: print("Working on point: " + str(idn))
    
    # Define the possible from and to points
    possible_from_points, closestEdges_FROM, minimal_projection_distance_from_OLD = determine_possible_fromto_points(edges = EDGES_ORIGINAL, point= from_point, MAX_PROJECTION=MAX_PROJECTION, MAX_DISTANCE_OPPOSITE_EDGE=MAX_DISTANCE_OPPOSITE_EDGE)
    possible_to_points, closestEdges_TO, minimal_projection_distance_to_OLD = determine_possible_fromto_points(edges = EDGES_ORIGINAL, point= to_point, MAX_PROJECTION=MAX_PROJECTION, MAX_DISTANCE_OPPOSITE_EDGE=MAX_DISTANCE_OPPOSITE_EDGE)
    
    ###############################################################################
    ################### Determine the shortest path (DISTANCE) ####################
    ###############################################################################
    SP_length_OLD, SP_edges_OLD = float('inf'), []
    from_point_projected, to_point_projected, pp_from, pp_to = None, None, float('inf'), float('inf')
    ind_from = 0
    for fp in possible_from_points:
        ind_to = 0
        for tp in possible_to_points:
            a, b, from_point_projected_temp, to_point_projected_temp = get_SP_distance(G_original, fp, tp)
            if a < SP_length_OLD:
                pp_from = closestEdges_FROM[ind_from][1]
                pp_to = closestEdges_TO[ind_to][1]
                SP_length_OLD, SP_edges_OLD = a, b
                from_point_projected, to_point_projected = from_point_projected_temp, to_point_projected_temp        
            ind_to += 1
        ind_from += 1
    
    P = SP_length_OLD
    # print(SP_length_OLD)
    ###############################################################################
    # Check shortest path length
    # tt = 0
    # for e in SP_edges_OLD:
    #     length = computeLengthLinestring(e[3], method = 'euclidean')#m
    #     tt += length#computeLengthLinestring(e[3], method = 'euclidean')#m
    #     # print(p)
    #     #t+=p[4]
    # print(tt)
    ###############################################################################
    ###############################################################################
    
    ###############################################################################
    ########## Compute the travel time under the different circumstances ##########
    ###############################################################################
    traveltimes = [0,0,0,0,0,0,0]
    try:
        for e in SP_edges_OLD:
            dist = computeLengthLinestring(e[3], method = 'euclidean')#m
            ind = 0
            for SPV in ['Velocity_00', 'Velocity_01', 'Velocity_10', 'Velocity_11', 'Velocity_20', 'Velocity_21', 'Average velocity']:
                speed = G_original.edges[e[0],e[1],e[2]][SPV]/3.6 #m/s
                traveltimes[ind] += (dist/speed)/60
                ind+=1
    except:
        traveltimes = ["-", "-", "-", "-", "-", "-", "-"]
    # print(traveltimes)
    ###############################################################################
    ###############################################################################
 
    ###############################################################################
    ##################### Determine the shortest path (TIME) ######################
    ###############################################################################
    optimal_traveltimes = [0,0,0,0,0,0,0]
    # try: 
    for SPV in range(0,7): #'Velocity_00', 'Velocity_01', 'Velocity_10', 'Velocity_11', 'Velocity_20', 'Velocity_21', 'Average velocity'
        SP_length_OLD, SP_edges_OLD = float('inf'), []
        #from_point_projected, to_point_projected, pp_from, pp_to = None, None, float('inf'), float('inf')
        ind_from = 0
        for fp in possible_from_points:
            ind_to = 0
            # print(fp)
            for tp in possible_to_points:
                # print(tp)
                a, b, from_point_projected_temp, to_point_projected_temp = get_SP_distance_TIME(G_original, fp, tp, SPV)
                if a < SP_length_OLD:
                    #pp_from = closestEdges_FROM[ind_from][1]
                    #pp_to = closestEdges_TO[ind_to][1]
                    SP_length_OLD, SP_edges_OLD = a, b
                    #from_point_projected, to_point_projected = from_point_projected_temp, to_point_projected_temp        
                ind_to += 1
            ind_from += 1
        optimal_traveltimes[SPV] = SP_length_OLD
    # except:
    #     optimal_traveltimes = ["-", "-", "-", "-", "-", "-"]
    # print(optimal_traveltimes)
    ###############################################################################
    # Check shortest path traveltime (only for the last circumstance)
    # traveltime = 0
    # for e in SP_edges_OLD:
    #     dist = computeLengthLinestring(e[3], method = 'euclidean')#m
    #     speed = G_original.edges[e[0],e[1],e[2]]['Velocity_21']/3.6 #m/s
    #     traveltime += (dist/speed)/60
    # print(traveltime)
    ###############################################################################
    ###############################################################################
    
    if PLOTPATH: print("SP length in the graph: " + str(SP_length_OLD)) 
    if PLOTPATH: plot_path(G_original_projected, from_point, to_point, SP_edges_OLD)
    """----------------------------------------------------------------------------------"""
    """----------------------------------------------------------------------------------"""
    
    # Add to the dataframe
    df.loc[len(df)] = [from_point, to_point, P, minimal_projection_distance_from_OLD, from_point_projected, pp_from, minimal_projection_distance_to_OLD, to_point_projected, pp_to] + traveltimes + optimal_traveltimes
    
    idn += 1
    if idn % 100 == 0:
        if not os.path.exists("RoutingResults"): os.makedirs("RoutingResults")
        df.to_csv('RoutingResults/' + FNAME, sep = ";")
       
if not os.path.exists("RoutingResults"): os.makedirs("RoutingResults")
df.to_csv('RoutingResults/' + FNAME, sep = ";")
"""_______________________________________________________________________________________"""
"""_______________________________________________________________________________________"""