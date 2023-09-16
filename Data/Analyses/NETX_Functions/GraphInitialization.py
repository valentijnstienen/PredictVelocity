# Date       : 11-01-2021
# Environment: conda activate ox
# Location   : cd "Desktop/NETX/NETX_Functions"
# Run        : python GraphInitialization.py
# Package info: /Users/valentijnstienen/anaconda3/envs/ox/lib/python3.8/site-packages

# Load settings
#exec(open("../SETTINGS.py").read())

import osmnx as ox

import geopandas as gp
import pandas as pd
import networkx as nx
import pickle
import numpy as np
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from shapely.geometry import MultiPoint, Point, Polygon, LineString

from NETX_Functions.PrintStuff import create_shapefile
from NETX_Functions.MathOperations import computeLengthLinestring
from NETX_Functions.TransformGraph import to_undirected

def graph_from_place(query,network_type="all_private",simplify=True,retain_all=False,truncate_by_edge=False,which_result=None,buffer_dist=None,clean_periphery=True,custom_filter=None):
    """
    Adjusted from: Boeing, G. 2017. "OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks."
    Computers, Environment and Urban Systems 65, 126-139. doi:10.1016/j.compenvurbsys.2017.05.004

    Create graph from OSM within the boundaries of some geocodable place(s).
    The query must be geocodable and OSM must have polygon boundaries for the
    geocode result. If OSM does not have a polygon for this place, you can
    instead get its street network using the graph_from_address function, which
    geocodes the place name to a point and gets the network within some distance
    of that point. Alternatively, you might try to vary the which_result
    parameter to use a different geocode result. For example, the first geocode
    result (ie, the default) might resolve to a point geometry, but the second
    geocode result for this query might resolve to a polygon, in which case you
    can use graph_from_place with which_result=2. which_result=None will
    auto-select the first multi/polygon among the geocoding results.
    Parameters
    ----------
    query : string or dict or list
        the query or queries to geocode to get place boundary polygon(s)
    network_type : string
        what type of street network to get if custom_filter is None. One of
        'walk', 'bike', 'drive', 'drive_service', 'all', or 'all_private'.
    simplify : bool
        if True, simplify the graph topology with the simplify_graph function
    retain_all : bool
        if True, return the entire graph even if it is not connected.
        otherwise, retain only the largest weakly connected component.
    truncate_by_edge : bool
        if True, retain nodes outside boundary polygon if at least one of
        node's neighbors is within the polygon
    which_result : int
        which geocoding result to use. if None, auto-select the first
        multi/polygon or raise an error if OSM doesn't return one.
    buffer_dist : float
        distance to buffer around the place geometry, in meters
    clean_periphery : bool
        if True, buffer 500m to get a graph larger than requested, then
        simplify, then truncate it to requested spatial boundaries
    custom_filter : string
        a custom network filter to be used instead of the network_type presets,
        e.g., '["power"~"line"]' or '["highway"~"motorway|trunk"]'. Also pass
        in a network_type that is in settings.bidirectional_network_types if
        you want graph to be fully bi-directional.
    Returns
    -------
    G : networkx.MultiDiGraph
    Notes
    -----
    You can configure the Overpass server timeout, memory allocation, and
    other custom settings via ox.config().
    """
    # create a GeoDataFrame with the spatial boundaries of the place(s)
    if isinstance(query, (str, dict)):
        # if it is a string (place name) or dict (structured place query), then
        # it is a single place
        gdf_place = ox.geocoder.geocode_to_gdf(
            query, which_result=which_result, buffer_dist=buffer_dist
        )
    elif isinstance(query, list):
        # if it is a list, it contains multiple places to get
        gdf_place = ox.geocoder.geocode_to_gdf(query, buffer_dist=buffer_dist)
    else:
        raise TypeError("query must be dict, string, or list of strings")

    # extract the geometry from the GeoDataFrame to use in API query
    polygon = gdf_place["geometry"].unary_union
    ox.utils.log("Constructed place geometry polygon(s) to query API")

    # create graph using this polygon(s) geometry
    G = ox.graph.graph_from_polygon(
        polygon,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
        clean_periphery=clean_periphery,
        custom_filter=custom_filter,
    )

    ox.utils.log(f"graph_from_place returned graph with {len(G)} nodes and {len(G.edges)} edges")
    return G, polygon

if area[0][-4:] == '.shp': # initial graph based on shapefile
    """ ----------------------------------------------------------------------------"""
    """ ---------- Shapefile to GeoDF with each linestring added separately ------- """
    """ ----------------------------------------------------------------------------"""
    def segments(curve):
        return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))
    def shape2gdf_full(p, make_full):
        """
        Parameters
        ----------
        p : str, File path - allowed formats geojson and ESRI Shapefile and other formats Fiona can read and write
    
        """
        # Load shapefile into GeoDataFrame. Each line piece is now in a row of this geodataframe
        gdf_old = gp.read_file(p)
        if not gdf_old.crs.is_projected: gdf_old = ox.project_gdf(gdf_old, to_crs= None, to_latlong=False)
   
        if make_full: # If the geometries in the shapefile are linestrings of complete edges, we first need to create a gdf with only the line pieces (much larger)
            gdf_old = gdf_old.dropna(subset=['geometry'])
            gdf = gdf_old.iloc[0:0]
            print("Total amount of roads that needs to be transformed to points: " + str(len(gdf_old)))
            for i in range(0, len(gdf_old)):
                if i%(int(len(gdf_old)/100))==0:  
                    print("Road: " + str(i) + ", " + str(i/(int(len(gdf_old)/100)))+"%", end="\r")
                
                # Find the segments of the linestring
                segments_line = segments(gdf_old.loc[i,'geometry'])
                # Extract geometry
                try: old_geom = list(gdf.geometry)
                except: old_geom = []
                # Make a row for each edge piece and add the corresponding geometry
                gdf = gdf.append([gdf_old.iloc[i,:]]*len(segments_line),ignore_index=True)
                gdf.geometry = old_geom + segments_line
        else: gdf = gdf_old
        # Save new geodataframe (use pickle)
        if not os.path.exists("Results/_TEMP/GraphInitialization"): os.makedirs("Results/_TEMP/GraphInitialization")
        with open("Results/_TEMP/GraphInitialization/gdf_new_"+CASENAME+".pickle", "wb") as file: pickle.dump(gdf, file)
    """ ----------------------------------------------------------------------------"""

    """ ----------------------------------------------------------------------------"""
    """ ------------ GeoDF to Graph with each linestring added separately ----------"""
    """ ----------------------------------------------------------------------------"""
    def convert_gdf2graph(make_G_bidi = True):
        """
        Converts geoDF to routable networkx graph. GeoDF must have edges for each line PIECE!
    
        Parameters
        ----------
        p : str, File path - geodatafrae
        make_G_bidi : bool, if True, assumes linestrings are bidirectional
    
        Returns
        -------
        G : graph
        """
        # Load previously saved geodataframe
        with open("Results/_TEMP/GraphInitialization/gdf_new_"+CASENAME+".pickle", "rb") as input_file: gdf = pickle.load(input_file)
    
        # Compute the start- and end-position based on linestring 
        gdf['Start_pos'] = gdf.geometry.apply(lambda x: x.coords[0])
        gdf['End_pos'] = gdf.geometry.apply(lambda x: x.coords[-1])
        gdf['length'] = [computeLengthLinestring(x, method = 'euclidean') for x in gdf['geometry']]
    
        gdf = gdf.fillna(value=np.nan)
    
        # Create Series of unique nodes and their associated position
        s_points = gdf.Start_pos.append(gdf.End_pos).reset_index(drop=True)
        s_points = s_points.drop_duplicates()
    
        # Add index of start node of linestring to geopandas DataFrame
        df_points = pd.DataFrame(s_points, columns=['Start_pos'])
        df_points['FNODE_'] = df_points.index
        gdf = pd.merge(gdf, df_points, on='Start_pos', how='inner')
    
        # Add index of end node of linestring to geopandas DataFrame
        df_points = pd.DataFrame(s_points, columns=['End_pos'])
        df_points['TNODE_'] = df_points.index
        gdf = pd.merge(gdf, df_points, on='End_pos', how='inner')

        # Bring nodes and their position in form needed for osmnx (give arbitrary osmid (index) despite not osm file)
        df_points.columns = ['pos', 'osmid']
        df_points[['x', 'y']] = df_points['pos'].apply(pd.Series)

        # Create Graph Object
        G = nx.MultiDiGraph(crs=gdf.crs)
    
        # Add nodes to graph
        df_node_xy = df_points.drop('pos', 1)
        for node, data in df_node_xy.T.to_dict().items():
            G.add_node(node, **data)
    
        attributes_edges = ['FNODE_', 'TNODE_', 'ROAD_CODE', 'ROAD_CODE_','sinuosity', 'straightdi', 'Start_pos', 'End_pos', 'osm_id', 'full_id', 'bridge', 'osm_type', 'old_name', 'layer', 'accuracy', 'name:etymo', 'email', 'descriptio', 'loc_name', 'PFM:RoadID', 'PFM:garmin', 'smoothness', 'est_width', 'lanes', 'lanes:back', 'lanes:forw', 'tracktype', 'bicycle', 'foot', 'horse', 'sac_scale', 'trail_visi', 'mtb:scale', 'ford', 'maxheight', 'motor_vehi', 'width', 'barrier', 'waterway', 'check_date', 'import', 'cutting', 'segregated', 'wheelchair', 'abandoned:', 'abandone_1', 'cycleway', 'sidewalk', 'embankment', 'lit','Observatio', 'seasonal', 'wikidata', 'ref', 'start_date', 'alt_name', 'name:en', 'mtb:scale:', 'covered', 'intermitte', 'noname', 'crossing','footway', 'bridge:str', 'official_n', 'man_made', 'incline','informal']
    
        # Add edges to graph
        for i, row  in gdf.iterrows():
            dict_row  = row.to_dict()
            if 'geometry' in dict_row: del dict_row['geometry']
            u = dict_row['FNODE_']
            v = dict_row['TNODE_']   
            for attr in attributes_edges:
                if attr in dict_row: del dict_row[attr]
            G.add_edge(u_for_edge = u, v_for_edge = v, **dict_row)
            # Add the reverse edge to the graph
            if make_G_bidi: G.add_edge(u_for_edge = v, v_for_edge = u, **dict_row)
    
        with open("Results/_TEMP/GraphInitialization/graph_"+CASENAME+".pickle", "wb") as file: pickle.dump(G, file)
    """ ----------------------------------------------------------------------------"""

    """ ----------------------------------------------------------------------------"""
    """ -------------- Simplify the graph and add relevant attributes  -------------"""
    """ ----------------------------------------------------------------------------"""
    def create_final_initial_graph():
        # Load previously saved graph
        with open("Results/_TEMP/GraphInitialization/graph_"+CASENAME+".pickle", "rb") as input_file: G = pickle.load(input_file)
    
        # Simplify graph
        G = ox.simplification.simplify_graph(G, strict=True, remove_rings=False)

        # Add edge attributes not in OSM (used in the algorithm)
        nx.set_edge_attributes(G, "-", name = 'close_to_point_start')
        nx.set_edge_attributes(G, "-", name = 'close_to_point_end')
        nx.set_edge_attributes(G, False, name = 'oneway')
        nx.set_edge_attributes(G, False, name = 'new')
        nx.set_edge_attributes(G, None, name = 'highway')
    
        # Fix geometry issues for the new graph
        edges = ox.graph_to_gdfs(G, nodes = False)
        vallist = list(edges.geometry)
        nx.set_edge_attributes(G, True, "geometry")
        ind = 0
        for edge in G.edges:
            nx.set_edge_attributes(G, {edge: {'geometry': vallist[ind]}})
            ind += 1
    
        # Save the final initial graph
        if not os.path.exists("Results/"+CASE+"/"+CASENAME): os.makedirs("Results/"+CASE+"/"+CASENAME)
        with open("Results/"+CASE+"/"+CASENAME+"/graph_0-0.pickle", "wb") as file: pickle.dump(G, file)
    """ ----------------------------------------------------------------------------"""
    
    # Transform shapefile to graph object
    shape2gdf_full(area[0], make_full = True)
    convert_gdf2graph(make_G_bidi = True)
    create_final_initial_graph()

    # Find the polygon object when the area is specified by a region (as a string)
    _, ch_polygon_1 = graph_from_place(area[1], simplify = True, retain_all = True, truncate_by_edge = True, clean_periphery = True)

    # Save the polygon information
    with open("Results/"+CASE+"/"+CASENAME+"/polygon_0-0.pickle", "wb") as file: pickle.dump(ch_polygon_1, file)
    with open("Results/"+CASE+"/"+CASENAME+"/PolygonInfo.txt", "w") as output: output.write(str(ch_polygon_1))

    # Plot the polygon(s) that is used
    fig = go.Figure()
    try: # when the area consists of multiple polygons
        for poly in ch_polygon_1:
            x, y = poly.exterior.coords.xy
            fig.add_trace(go.Scattermapbox(mode='lines', lat=y.tolist(), lon=x.tolist(), visible = True,  marker = {'size' : 15, 'color': 'pink', 'allowoverlap': True}))
    except: # when the area is a single polygon
        x, y = ch_polygon_1.exterior.coords.xy
        fig.add_trace(go.Scattermapbox(mode='lines', lat=y.tolist(), lon=x.tolist(), visible = True,  marker = {'size' : 15, 'color': 'pink', 'allowoverlap': True}))
    # Focus map on a random polygon point
    fig.update_layout(mapbox1 = dict(center = dict(lat=y[0], lon=x[0]), accesstoken = mapbox_accesstoken, zoom = 10),margin = dict(t=10, b=0, l=10, r=10),showlegend=False,mapbox_style="satellite")
    # Save the map
    fig.write_html("Results/"+CASE+"/"+CASENAME+"/Visual.html")

    # Save the graph as shapefile  
    with open("Results/"+CASE+"/"+CASENAME+"/graph_0-0.pickle", "rb") as input_file: G = pickle.load(input_file) #
    if two_way: G = to_undirected(G)
    nodes, edges = ox.graph_to_gdfs(G)
    if not os.path.exists("Results/"+CASE+"/"+CASENAME+"/_Shapefiles (Start)"): os.makedirs("Results/"+CASE+"/"+CASENAME+"/_Shapefiles (Start)")
    create_shapefile(nodes, "Results/"+CASE+"/"+CASENAME+"/_Shapefiles (Start)/"+CASENAME[0]+"_nodes.shp")
    create_shapefile(edges, "Results/"+CASE+"/"+CASENAME+"/_Shapefiles (Start)/"+CASENAME[0]+"_edges.shp")
    """ ----------------------------------------------------------------------------"""
else: # Initial graph is not a shapefile
    # We start creating the initial graph using the OSMnx package. We also want to have a polygon that can be used for checking whether a GPS trace is within the scope of the case. 
    if type(area) == str: # Area is specified by a region (as a string)
        # Create initial graph
        initial_G, ch_polygon_1 = graph.graph_from_place(area, simplify = True, retain_all = True, truncate_by_edge = True, clean_periphery = True)
    elif type(area[0]) == tuple: # Area is a polygon
        # Create Polygon object
        ch_polygon_1 = Polygon([Point(x) for x in area])
        # Create initial graph
        initial_G = ox.graph.graph_from_polygon(ch_polygon_1, simplify = True, retain_all = True, truncate_by_edge = True, clean_periphery = False)#,network_type = 'drive') 
    else: # Area is a rectangle
        # Define corner points of the rectangle (the area does not represent actual points)
        lu = Point(area[0], area[3])
        ld = Point(area[0], area[2])
        ru = Point(area[1], area[3])
        rd = Point(area[1], area[2])
        # Create Polygon object
        ch_polygon_1 = MultiPoint([(lu.x, lu.y), (ld.x, ld.y), (ru.x, ru.y), (rd.x, rd.y)]).convex_hull.buffer(0)
        # Create initial graph
        initial_G = ox.graph.graph_from_bbox(lu.y, ld.y, lu.x, ru.x, simplify = True, retain_all = True, truncate_by_edge = True, clean_periphery = False) #,network_type='drive')
        
    # Add edge attributes not in OSM (used in the algorithm)
    nx.set_edge_attributes(initial_G, "-", name = 'close_to_point_start')
    nx.set_edge_attributes(initial_G, "-", name = 'close_to_point_end')
    nx.set_edge_attributes(initial_G, False, name = 'new')
    nx.set_edge_attributes(initial_G, False, name = 'driven')
    nx.set_edge_attributes(initial_G, "", name = 'DatesVelocities')
    nx.set_edge_attributes(initial_G, None, name = 'length_OLD')
     
    # Project the graph to the crs in which the centroid of the initial graph lies
    initial_G = ox.project_graph(initial_G, to_crs = None)
    
    for e in initial_G.edges(data = 'geometry', keys = True): 
        initial_G.edges[e[0], e[1], e[2]]['length_OLD'] = initial_G.edges[e[0], e[1], e[2]]['length']
        initial_G.edges[e[0], e[1], e[2]]['length'] = computeLengthLinestring(e[3], method = 'euclidean')
            
    # Save the graph object
    if not os.path.exists("Results/"+CASE+"/"+CASENAME): os.makedirs("Results/"+CASE+"/"+CASENAME)
    with open("Results/"+CASE+"/"+CASENAME+"/graph_0-0.pickle", "wb") as file: pickle.dump(initial_G, file)
        
    # Save the polygon information 
    with open("Results/"+CASE+"/"+CASENAME+"/polygon_0-0.pickle", "wb") as file: pickle.dump(ch_polygon_1, file)
    with open("Results/"+CASE+"/"+CASENAME+"/PolygonInfo.txt", "w") as output: output.write(str(ch_polygon_1))
    # Plot the polygon(s) that is used
    fig = go.Figure()
    try: # when the area consists of multiple polygons
        for poly in ch_polygon_1:
            x, y = poly.exterior.coords.xy
            fig.add_trace(go.Scattermapbox(mode='lines', lat=y.tolist(), lon=x.tolist(), visible = True,  marker = {'size' : 15, 'color': 'pink', 'allowoverlap': True}))
    except: # when the area is a single polygon
        x, y = ch_polygon_1.exterior.coords.xy
        fig.add_trace(go.Scattermapbox(mode='lines', lat=y.tolist(), lon=x.tolist(), visible = True,  marker = {'size' : 15, 'color': 'pink', 'allowoverlap': True}))
    # Focus map on a random polygon point
    fig.update_layout(mapbox1 = dict(center = dict(lat=y[0], lon=x[0]), accesstoken = mapbox_accesstoken, zoom = 10),margin = dict(t=10, b=0, l=10, r=10),showlegend=False,mapbox_style="satellite")
    # Save the map
    fig.write_html("Results/"+CASE+"/"+CASENAME+"/Visual.html")
         
    # Create and save as shapefiles
    nodes, edges = ox.graph_to_gdfs(initial_G)
    if not os.path.exists("Results/"+CASE+"/"+CASENAME+"/_Shapefiles (Start)"): os.makedirs("Results/"+CASE+"/"+CASENAME+"/_Shapefiles (Start)")
    create_shapefile(nodes, "Results/"+CASE+"/"+CASENAME+"/_Shapefiles (Start)/start_nodes.shp")
    create_shapefile(edges, "Results/"+CASE+"/"+CASENAME+"/_Shapefiles (Start)/start_edges.shp") 
    """ ----------------------------------------------------------------------------"""


