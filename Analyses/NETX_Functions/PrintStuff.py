import plotly.graph_objects as go
import osmnx as ox
import numpy as np

def addGraph(fig_graph, G, color, only_nodes = False, include_existing = False):
    """
    Extend the map on fig_graph with the nodes and edges of graph G.

    Parameters
    ----------
    fig_graph : figure
        the figure onto which the new plot needs to be placed
    G : graph (nx)
        graph object that contains the edges and nodes
    only_nodes : Boolean
        indicates whether you want to print all edges or only the nodes. 
    include_existing : Boolean 
        indicates whether you want to include all existing OSM edges/nodes in the plot
        (NOTE: if True, it may take a while, if even possible, to load the graph)
        
    Returns
    -------
    fig_graph : figure
        the updated figure, which now includes the nodes and edges of graph G
        (nodes are red and edges are blue)
        
    """
    # Obtain the nodes and edges of graph G
    if only_nodes:
        nodes = ox.graph_to_gdfs(G, edges = False, node_geometry= True)
        
        # Add the new nodes
        fig_graph.add_trace(go.Scattermapbox(mode='markers', lat=list(nodes.geometry.y), lon=list(nodes.geometry.x), text = nodes.index, visible = True, marker = {'size' : 7, 'color': 'red', 'allowoverlap': True}))
        
    else:
        nodes, edges = ox.graph_to_gdfs(G)
    
        # Split edges/nodes into new/old edges/nodes
        old_edges = edges.loc[~edges.new]
        new_edges = edges.loc[edges.new]# & (edges['length'] > 75)]
        new_nodes = nodes.loc[list(set(list(new_edges.u) + list(new_edges.v)))]
    
        # Print the new edges
        print("The newly added (explicitly) edges:")
        print(new_nodes)
        print(new_edges)
        print("Total length added explicitly by the algorithm: " + str(new_edges['length'].sum()))

    
        # Create lists of all coordinates of all the new/old edges
        edge_points_list_old = old_edges.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
        edge_points_list_new = new_edges.apply(lambda x: [y for y in x['geometry'].coords], axis=1)
    
        # Add the old edges (in blue)
        if include_existing & (len(old_edges)>0):
            # Initialize parameters 
            edge_latitudes_OLD, edge_longitudes_OLD, edge_notes_OLD, ind = [], [], [], 0
            for e in edge_points_list_old:
                lons, lats = zip(*e)
                edge_longitudes_OLD = edge_longitudes_OLD + list(lons)
                edge_latitudes_OLD = edge_latitudes_OLD + list(lats)
                edge_notes_OLD = edge_notes_OLD + len(lats)*[ind]
                edge_longitudes_OLD.append(None)
                edge_latitudes_OLD.append(None)
                edge_notes_OLD.append(None)
                ind = ind + 1

        # Add the new edges (in red)
        if len(edge_points_list_new)>0: 
            # Initialize parameters 
            edge_latitudes_NEW, edge_longitudes_NEW, edge_notes_NEW, ind = [], [], [], 0
            for e in edge_points_list_new:
                lons, lats = zip(*e)
                edge_longitudes_NEW = edge_longitudes_NEW + list(lons)
                edge_latitudes_NEW = edge_latitudes_NEW + list(lats)
                edge_notes_NEW = edge_notes_NEW + len(lats)*[ind]
                edge_longitudes_NEW.append(None)
                edge_latitudes_NEW.append(None)
                edge_notes_NEW.append(None)
                ind = ind + 1
    
        # Add the edges to the graph
        if not only_nodes:
            # Add the old existing edges 
            if include_existing & (len(old_edges)>0): fig_graph.add_trace(go.Scattermapbox(mode='lines', lat=edge_latitudes_OLD, lon=edge_longitudes_OLD, text = edge_notes_OLD, visible = True, opacity = 0.5, marker = {'size' : 10, 'color': 'blue', 'allowoverlap': True}))
            # Add the new edges
        if len(new_edges) > 0: fig_graph.add_trace(go.Scattermapbox(mode='lines', lat=edge_latitudes_NEW, lon=edge_longitudes_NEW, text = edge_notes_NEW, visible = True, marker = {'size' : 10, 'color': 'red', 'allowoverlap': True}))
    
        # Add the existing nodes to the graph
        if include_existing & (len(old_edges)>0):
            fig_graph.add_trace(go.Scattermapbox(mode='markers', lat=list(nodes.geometry.y), lon=list(nodes.geometry.x), text = nodes.index, visible = True, marker = {'size' : 7, 'color': 'blue', 'opacity': 0.5, 'allowoverlap': True}))
    
        # Add the new nodes
        fig_graph.add_trace(go.Scattermapbox(mode='markers', lat=list(new_nodes.geometry.y), lon=list(new_nodes.geometry.x), text = new_nodes.index, visible = True, marker = {'size' : 7, 'color': 'red', 'allowoverlap': True}))
    
    return fig_graph




def create_shapefile(gdf, filename):
    
    try: gdf['osmid'] = [str(l) for l in gdf['osmid']]
    except: a = 1
    
    try:
        a = gdf.x
        attributes = ['x','y', 'lon', 'lat', 'ref', 'highway']
        for attr in attributes:
            try: gdf[attr] = [str(l) for l in gdf[attr]]
            except: a = 1
    except: 
        attributes = ['Start_pos', 'End_pos', 'osm_type', 'highway', 'oneway', 'surface', 'access', 'junction', 'name', 'old_name', 'bridge', 'layer', 'accuracy', 'name:etymo', 'email', 'descriptio', 'loc_name', 'PFM:RoadID', 'PFM:garmin', 'smoothness', 'est_width', 'lanes', 'maxspeed','lanes:back', 'lanes:forw', 'tracktype', 'bicycle', 'foot', 'horse', 'sac_scale', 'trail_visi', 'mtb:scale', 'ford', 'maxheight','motor_vehi', 'width', 'barrier', 'service', 'waterway', 'check_date', 'import', 'cutting', 'segregated', 'wheelchair', 'abandoned:','abandone_1', 'cycleway', 'sidewalk', 'embankment', 'lit', 'tunnel','Observatio', 'seasonal', 'wikidata', 'ref', 'start_date', 'alt_name','name:en', 'mtb:scale:', 'covered', 'intermitte', 'noname', 'crossing','footway', 'bridge:str', 'official_n', 'man_made', 'incline','informal']
        for attr in attributes:
            try: gdf[attr] = [str(l) for l in gdf[attr]]
            except: a = 1
        
        try: gdf = gdf.drop(['close_to_point_start', 'close_to_point_end'], axis=1)
        except: a = 1

    gdf.to_file(filename)


def addEdge(fig_graph, edge):
    """
    Extend the map on fig_graph with the nodes and edges of graph G.

    Parameters
    ----------
    fig_graph : figure
        the figure onto which the new plot needs to be placed
    G : graph (nx)
        graph object that contains the edges and nodes
        
    Returns
    -------
    fig_graph : figure
        the updated figure, which now includes the nodes and edges of graph G
        (nodes are red and edges are blue)
        
    """
    try: edge_points_list = [y for y in edge['geometry'].coords]
    except: edge_points_list = [y for y in edge[3].coords]
    

    # Add new edges (in red)
    edge_latitudes, edge_longitudes, edge_notes, ind = [], [], [], 0
    for e in edge_points_list:
        lons, lats = zip(e)
        edge_longitudes = edge_longitudes + list(lons)
        edge_latitudes = edge_latitudes + list(lats)
        edge_notes = edge_notes + len(lats)*[ind]
        ind = ind + 1
    
    
    fig_graph.add_trace(go.Scattermapbox(mode='lines', lat=edge_latitudes, lon=edge_longitudes, text = edge_notes, visible = True, marker = {'size' : 30, 'color': 'yellow', 'allowoverlap': True}))
    
    #fig_graph.add_trace(go.Scattermapbox(mode='markers', lat=list(nodes.geometry.y), lon=list(nodes.geometry.x), text = nodes.index, visible = True, marker = {'size' : 7, 'color': 'blue', 'allowoverlap': True}))
    
    return fig_graph
      
def addCloseEdges(fig_graph, G, close_to_edges, steps, backwards):
    """
    Extend the map on fig_graph with the edges that are within [steps] steps from
    the [close_to_edges] when going backwards/forward.

    Parameters
    ----------
    fig_graph : figure
        the figure onto which the new plot needs to be placed
    G : graph (nx)
        graph object that contains the edges and nodes
    close_to_edges : list of tuples
        containing the edges to which we want to be close. An edge contains (at least):
        the following 4 items (in the first four spots of the tuple):
        from_node, to_node, key, geometry (e.g., (7676251282, 5777901393, 0, 
        <shapely.geometry.linestring.LineString object at 0x7fbe00160b20>, 0.3844871533782418))
    steps : number
        the number of steps we can be away from the close_to_edges.
    backwards : boolean
        Indicates whether we are looking backwards (True) or forward (False)
        
    Returns
    -------
    fig_graph : figure
        the updated figure, which now includes the nodes and edges of graph G that 
        are close to the close_to_edges (colored yellow)
     
    """
    closeEdges = getCloseEdges(G, close_to_edges, steps, backwards)
    print(closeEdges)
    edge_points_list = [item[3].coords for item in closeEdges]
    
    edge_latitudes, edge_longitudes, ind = [], [], 0
    for e in edge_points_list:
        lons, lats = zip(*e)
        edge_longitudes = edge_longitudes + list(lons)
        edge_latitudes = edge_latitudes + list(lats)
        edge_longitudes.append(None)
        edge_latitudes.append(None)
        ind = ind +1
    
    fig_graph.add_trace(go.Scattermapbox(mode='lines', lat=edge_latitudes, lon=edge_longitudes, visible = True, marker = {'color': 'yellow', 'allowoverlap': True}))
    
    return fig_graph
    
    
    