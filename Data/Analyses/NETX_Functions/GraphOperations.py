import osmnx as ox
import pandas as pd
from shapely.geometry import Point, LineString
from NETX_Functions.MathOperations import dist, computeAngularDifference, computeLengthLinestring, computeBearing
from math import sqrt
import networkx as nx

"""----------------------------------------- GRAPH CHANGING FUNCTIONS --------------------------------------"""
"""---------------------------------------------------------------------------------------------------------"""
def add_datvels(current_string, new_string):
    new_datvel = current_string
    temp = new_string.strip("|").split("|")
    # There should at least be one speed registration point
    if len(temp)> 1: 
        for p in range(0,len(temp),2):
            candidate_datvel = temp[p] +"|"+ temp[p+1]
            if not (candidate_datvel in new_datvel): new_datvel += "|" + candidate_datvel
    return new_datvel

#TODO DONE
def ensure_point_in_network(G_proj, projectedPoint, edge, be, name, settings, two_way, temp_last_point = None, start_point = None, do_print = False):
    """
    Ensure the new connection point [projectedPoint] is in the network. This is done, either by merging the point 
    with an exisitng node, or adding the node explicityly. The point is on line piece [be] on [edge]. 
    
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    projectedPoint : tuple (x,y)
        the point that will be inserted into the edge.
    edge : tuple
        (u, v, key, geometry)
    be : float
        edge piece id at which the point should be
    name : String
        (part of) the name of the new node
    settings : List
        Algorithm settings
    two_way : Boolean
        Indicates whether we want to add the point in two ways (if possible).
    temp_last_point/start_point : String
        Node names that may change due to this ensuring
    
    Returns
    -------
    current_point : String
        name of the node that is representing the new connection point (could have been merged)
    point_new : Boolean
        This boolean indicates whether the point was added explicitly or not. 
    splitted_edges : List
        represent possible edges that are used for edge_old, such as splitted edges that result from 
        adding the point to the edge. 
    temp_last_point/start_point : Strings
        Adjusted names (if needed)
    edge : tuple
        (u, v, key, geometry) If point is merged, adjust the edge for further reference. 
    
    """
    # If an edge is splitted, due to adding a point, keep the two splitted edges. We may need them later on.. 
    splitted_edges = None
    #merged_in_edge = False

    # We add the projectedPoint (first check whether it can be merged with an existing point) This node will be referenced to as currentPoint
    node_to_be_merged = mergeNode(G_proj, (projectedPoint[1], projectedPoint[0]), max_dist_merge = settings[2], indmax = settings[4])
    
    # Merge the point with an existing node
    if node_to_be_merged is not None:
        if do_print: print("Node is merged with: " + str(node_to_be_merged))
        currentPoint = str(node_to_be_merged) + '/' + name
        # Adjust the name of the newly merged node
        nx.relabel_nodes(G_proj, {node_to_be_merged: currentPoint}, copy = False) 
        # Check if node must be inserted in the edge (only when merged with not one of the end_points). It is merged with a node that is not in [edge]
        if (node_to_be_merged not in [edge[0], edge[1]]): 
            if do_print: print("Merge " + str(currentPoint) + " in the edge: {0}".format(edge))
            _, projectedPoint_Edge, be = projectPointOnEdge(edge, (Point(G_proj.nodes[currentPoint]['x'], G_proj.nodes[currentPoint]['y']),-1), alpha_bar = None, max_dist = float('inf'))        

            # Adjust all incoming and outgoing edges of currentPoint.
            incoming_edges = list(G_proj.in_edges(currentPoint, keys = True, data = True))
            outgoing_edges = list(G_proj.out_edges(currentPoint, keys = True, data = True))
            for e in incoming_edges:
                # Adjust geometry
                geom = e[3]['geometry']
                edge_points_list = [y for y in geom.coords]
                edge_points_list = sorted(set(edge_points_list), key = edge_points_list.index)
                if len(edge_points_list) <= 1: new_geom = LineString([projectedPoint_Edge] + [projectedPoint_Edge]) 
                else: new_geom = LineString(edge_points_list[:-1] + [projectedPoint_Edge]) 
                G_proj.edges[e[0], e[1], e[2]]['geometry'] = new_geom
                G_proj.edges[e[0], e[1], e[2]]['length'] = computeLengthLinestring(new_geom, method = 'euclidean')
            for e in outgoing_edges:
                # Adjust geometry
                geom = e[3]['geometry']
                edge_points_list = [y for y in geom.coords]
                edge_points_list = sorted(set(edge_points_list), key = edge_points_list.index)
                if len(edge_points_list) <= 1: new_geom = LineString([projectedPoint_Edge] + [projectedPoint_Edge]) 
                else: new_geom = LineString([projectedPoint_Edge] + edge_points_list[1:]) 
                G_proj.edges[e[0], e[1], e[2]]['geometry'] = new_geom
                G_proj.edges[e[0], e[1], e[2]]['length'] = computeLengthLinestring(new_geom, method = 'euclidean')
            
            # Adjust the node 
            new_point = Point(projectedPoint_Edge)
            G_proj.nodes[currentPoint]['geometry'] = new_point
            G_proj.nodes[currentPoint]['x'] = new_point.x
            G_proj.nodes[currentPoint]['y'] = new_point.y
            
            # Merge the existing node in [edge]
            currentPoint, splitted_edges = add_point_expl(G_proj, point = projectedPoint_Edge, edge = edge, be = be, node_name = currentPoint, settings = settings, merge = True, two_way = False, do_print = do_print)  
        
        if splitted_edges is None: 
            splitted_edges = [(currentPoint, currentPoint, 0, LineString([(G_proj.nodes[currentPoint]['x'], G_proj.nodes[currentPoint]['y']), (G_proj.nodes[currentPoint]['x'], G_proj.nodes[currentPoint]['y'])])), None]
        
        # Adjust temp_last_point and start_point, may have been changed by the merging process
        edge, temp_last_point = relabel(edge, temp_last_point, node_to_be_merged, currentPoint)
        _, start_point = relabel(None, start_point, node_to_be_merged, currentPoint)
        
        # Update parameters
        point_new = False # point is not added explicitly
    else: # Node will not be merged
        if do_print: print("Node is not merged. Add explicitly to the network and split the corresponding edge: {0}".format(edge))
        currentPoint = name
        currentPoint, splitted_edges = add_point_expl(G_proj, point = projectedPoint, edge = edge, be = be, node_name = currentPoint, settings = settings, merge = False, two_way = two_way, do_print = do_print)
        # Update parameters 
        point_new = True # point is added explicitly
    """ _________________________________________________________________________________________________________"""
    
    return currentPoint, point_new, splitted_edges, temp_last_point, start_point, edge

#TODO DONE
def include_point_in_newedge(G, edge, be, point, old_point, points_to_be_adjusted, settings, crs, max_actual_dist, temp_new, do_print = False):
    """
    Include [point] into the geometry of [edge]. Specifically, this [point] 
    is added on the line piece [be]. 
    
    Difficulty:
    - When added on the first/last line piece
        -> relocate the connection point (remove, if possible, the old connection point)
   
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    edge : tuple
        (u, v, key, geometry)
    be : float 
        number of corresponding edge piece where [point] will be included
    point : tuple
        (y, x, -1)
    old_point : tuple
        (projectedPoint_OLD (x,y), edge_old (u, v, key, geometry)) Can be used when adding a point 
        before the first GPS point. In this case, we are updating the close to point start, as we are now
        getting closer to the "real" connection point. 
    points_to_be_adjusted: List
        If this absorption results is followed by connecting a newly created edge, some point may change 
        names. These points are adjusted. 
    settings: -
        rest of the algorithm settings.
    crs: string
        Coordinate Reference System
    max_actual_dist : List
        MDC and MLDC for the current point. Note that this is only used when we use the previous point, 
        for projecting a new starting connection point. 
    temp_new : Boolean
        This boolean indicates whether the first point of a newly created edge was added explicitly or not. 
        Note that when creating a new edge and, due to this inclusion, and if a new connection point is merged with 
        our starting point of our newly created edge, we assume that this point is no longer new (as it now has two
        different functions: connection point AND starting point)
    
    Returns
    -------
    edge : (u, v, key, geometry)
        The new adjusted edge (that now includes the [point])
    points_adjusted : List
        Adjusted names of the list of [points_to_be_adjusted], e.g, temp_last_point and 
    edge_old_NEW : List
        adjusting the edge might lead to connection points being changed it might happen that our edge_old does not 
        exist anymore after this process. This is only the case if it has been split in the process. Therefore, we 
        also ask for the split edges [edge_old_NEW], as these are the potential new edge_olds.
    temp_new : Boolean
        adjusted temp_new
    
    """
    # Find geometry of current edge to be changed
    geom = G.edges[edge[0], edge[1], edge[2]]['geometry']
    edge_points_list = [y for y in geom.coords]
    #print(edge_points_list)

    # Check to which point the connection point should be close  
    ctps = G.edges[edge[0], edge[1], edge[2]]['close_to_point_start']
    ctpe = G.edges[edge[0], edge[1], edge[2]]['close_to_point_end']

    # Determine distance to both endpoints of the be edge
    p1, p2 = G.nodes[edge[0]], G.nodes[edge[1]]  
    dist_1, dist_2 = sqrt((p1['x']-point[1])**2+(p1['y']-point[0])**2), sqrt((p2['x']-point[1])**2+(p2['y']-point[0])**2)
    
    # Initialize list of points that may be adjusted due to the adjustment
    points_adjusted = points_to_be_adjusted
    edge_old_NEW = None # Initialize object that will save possible edge_olds when these are changed during the adjustment procedure. (The edges that are split due to a new connection point being made)
    
    if do_print: print("--------------------------")
    # Point added on the first line piece -> relocate the connection point (remove, if possible, the old connection point) 
    # If the connection point is 'Start', then we do not relocate, as this was the start point of a given edge. 
    if (be == 0) & (ctps != 'Start'): 
        if do_print: print("Add point to a new edge before the first GPS point..")
        
        # Check if previous point was projected on the to be adjusted edge. It could be that the previous point was projected on the to be adjusted edge. This means that we can not use the prvious point information (as this point lies on the edge that will be adjusted). Therefore, we still need the old information of the connection point. Note that this may occur when the previous point did not change the geometry, because it was close to a corner point (or merged with an endpoint). Either way, 
        prev_point_projected_on_edge = False
        try:
            if old_point[1][0:3] == edge[0:3]: prev_point_projected_on_edge = True
        except: prev_point_projected_on_edge = False
        
        # If we can use the previous point (old_point[0] is not None), we do not need to know the close_to_point start, as we are defining a new (more accurate (close)) close_to_point_start. Namely the one corresponding to this current new point. If we canNOT use the previous point (old_point[0] is None), for instance when the edge is opposite,  we must first find the old point to which the connection point should be close.
        if (old_point[0] is None) | prev_point_projected_on_edge: 
            # First, find the edge onto which the close point is projected. We do this by finding a projection point close to the start point of this edge. We therefore use [float('inf'), (0,float('inf'))] as max_actual_dist. This means that everything close to zero is possible
            if do_print: print("--------------------------")
            if do_print: print("Find the edge onto which the close point is projected...")
            if do_print: print("Close to the point: {0}, which is projected onto edge (backward): {1}".format((p1["x"], p1["y"]), edge))
            projectedPoint_cp, closestDistance_cp, edge_cp, be_cp = findProjectionPoint(G, (ctps[0][0], ctps[0][1], -1), close_to_edge = edge, connecting = True, forward = False, temp_point = (p1["x"], p1["y"]), settings = settings, max_actual_dist = [float('inf'), (0,float('inf'))], crs= crs)
            if do_print: print("Projected point: {0}".format(projectedPoint_cp))
            if do_print: print("Closest distance: " + str(closestDistance_cp) + " meters.")
            if do_print: print("Projected onto edge: {0}".format(edge_cp))

            # Suppose the close_to_point_start is projected on the edge we are investigating (probably on a corner point). We cannot project our node on an edge. We can just add the GPS point to the geometry without relocating the connecting point. The current connecting point will still be close to the close_to_point_start.
            if (edge[0:3] == edge_cp[0:3]):
                if do_print: print("Close to point start is onto the newly created edge... No need for relocation of connection point.")
                edge_newgeom = LineString(edge_points_list[0:(be+1)] +  [(point[1], point[0])] + edge_points_list[(be+1):])
                # Adjust the geometry of the new edge (include point on the right spot)
                G.edges[edge[0], edge[1], edge[2]]['geometry'] = edge_newgeom
                G.edges[edge[0], edge[1], edge[2]]['length'] = computeLengthLinestring(edge_newgeom, method = 'euclidean')
                edge = (edge[0], edge[1], edge[2], edge_newgeom)
                return edge, points_adjusted, None, temp_new
        
        # So, at this point we know the close_to point. Either we use the previous point (only used when available) or we use projectedPoint_cp just found using ctps. Alternatively, we could always use this second option, but this new point provides more information about where the connection point should be.

        # Remove the old edge, but keep its osmid and oneway
        id_keep = G.edges[edge[0], edge[1], edge[2]]['osmid']
        oneway_keep = G.edges[edge[0], edge[1], edge[2]]['oneway']
        datvel_keep = G.edges[edge[0], edge[1], edge[2]]['DatesVelocities']
        G.remove_edge(edge[0], edge[1], key = edge[2])

        # Update the ctps for next rounds. 
        max_actual_dist_new = [ctps[1][0], (0, ctps[1][1][1])] 
        
        # Find the new connection point
        if do_print: print("--------------------------")
        if do_print: print("Find the new connecting point....")
        # If we cannot use the previous point, either because previous point was not absorbed, or if we are dealing with an opposite edge that we want to adjust, we use the ctps_new
        if (old_point[0] is None) | prev_point_projected_on_edge: 
            if do_print: print("Close to the point: {0}, which is projected onto edge (forward): {1}".format(projectedPoint_cp, edge_cp))
            projectedPoint_NEW, closestDistance_NEW, edge_NEW, be_NEW = findProjectionPoint(G, point, close_to_edge = edge_cp, connecting = True, forward = True, temp_point = projectedPoint_cp, settings = settings, max_actual_dist = max_actual_dist_new, crs = crs)
            ctps = [ctps[0], max_actual_dist_new] 
        else: # If we can find a new connecting using the previous point that was absorbed (old_point is not none), we can find this in the "normal" way. Note old_point is the previous point that was projected.
            if do_print: print("Close to the point: {0}, which is projected onto edge (forward): {1}".format(old_point[0], old_point[1]))
            projectedPoint_NEW, closestDistance_NEW, edge_NEW, be_NEW = findProjectionPoint(G, point, close_to_edge = old_point[1], connecting = True, forward = True, temp_point = old_point[0], settings = settings, max_actual_dist = max_actual_dist, crs = crs)
            ctps = [(old_point[0][1],old_point[0][0] ,-1), max_actual_dist]
        if do_print: print("Projected point: {0}".format(projectedPoint_NEW))
        if do_print: print("Closest distance: " + str(closestDistance_NEW) + " meters.")
        if do_print: print("Projected onto edge: {0}".format(edge_NEW))
        if do_print: print("--------------------------")

        # Now that we have a new connection point, check if this connection point can be merged with an existing node. 
        node_to_be_merged = mergeNode(G, (projectedPoint_NEW[1], projectedPoint_NEW[0]), max_dist_merge = settings[2], indmax = settings[4])
        if node_to_be_merged is not None: 
            if do_print: print("This projection point is merged with: " + str(node_to_be_merged))
            currentPoint = node_to_be_merged 
            # In the case that we merged with the startpoint of a currently creating edge, we assume that this starting point is no longer explicitly added (as it has now two functions). Note that the first point of [points_to_be_adjusted] is the temp_last_point, the starting point of a currently newly created edge. 
            if len(points_to_be_adjusted) > 0:
                if currentPoint == points_to_be_adjusted[0]: temp_new = False
        else: # Node can not be merged
            if do_print: print("This projection point is not merged, it is added explicitly.")
            # Explicitly add the new projection of the new node (on the first line piece) 
            currentPoint = str(edge[0]) # Use the same name as the old connection point. Note that this old node will be removed. If it could not be removed, because it was relevant, we can still distinguish them due to the | added to this node. 
            # Add the point explicitly. Note that edge_old_NEW is required to define possible edge_olds, as we are splitting edges that may have been the edge_old. 
            currentPoint, edge_old_NEW = add_point_expl(G, projectedPoint_NEW, edge_NEW, be_NEW, node_name = currentPoint, settings = settings, merge = False, two_way = settings[3], do_print = do_print)

        # Add the new edge (note that it is a new edge, so highway is in any case not defined (we use driven))
        keyNew = max([item[2] for item in G.edges(currentPoint, edge[1], keys = True) if ((item[0] == currentPoint) & (item[1] == edge[1]))], default=-1) + 1
        constructing_edge_newgeom = LineString([(G.nodes[currentPoint]['x'], G.nodes[currentPoint]['y'])] + [(point[1], point[0])] + edge_points_list[1:])
        G.add_edge(currentPoint, edge[1], osmid = id_keep, new = True, driven = True, DatesVelocities = datvel_keep, ref = None, highway=None, oneway= oneway_keep, length = computeLengthLinestring(constructing_edge_newgeom, method = 'euclidean'), geometry = constructing_edge_newgeom, close_to_point_start = ctps, close_to_point_end = ctpe, maxspeed = None, service = None, bridge= None, lanes = None, u = currentPoint, v = edge[1], key = keyNew)
 
        # Also remove the corresponding old connection point when: 
        # - the new projection point is not merged with the old projection point AND
        # - the old end point is not still the starting point of the edge.
        if (edge[0] != currentPoint) & (edge[0] != edge[1]): 
            if do_print: print("Remove " + str(edge[0]) + " from the graph.")
            
            # Remove the point
            remove_point(G, old_point = edge[0], do_print = do_print) 

            # If any of the relevant points was equal to the removed point, adjust it to the newly added node (e.g., temp_last_point).  
            points_adjusted = []
            for i in points_to_be_adjusted:
                if (i == edge[0]): points_adjusted += [currentPoint]
                else: points_adjusted += [i]
        
        # Define possible (if needed) new old edges
        incoming_edges = list(G.in_edges(currentPoint, keys = True, data = 'geometry'))
        edge = (currentPoint, edge[1], keyNew, constructing_edge_newgeom) # Edge onto which the original point was projected. 
        edge_old_NEW = incoming_edges + [edge]      
        
    elif (be == (len(edge_points_list)-2)) & (ctpe != 'End'): 
        if do_print: print("Add point to a new edge after the last GPS point..")

        # We cannot use the previous point, because the close point comes next. Therefore, we must first find the old point to which the connection point should be close.
        # First, find the edge onto which the close point is projected. We do this by finding a projection point close to the end point of this edge. We therefore use [float('inf'), (0,float('inf'))] as max_actual_dist. This means that everything close to zero is possible
        if do_print: print("--------------------------")
        if do_print: print("Find the edge onto which the close point is projected...")
        if do_print: print("Close to the point: {0}, which is projected onto edge (forward): {1}".format((p2["x"], p2["y"]), edge))
        projectedPoint_cp, closestDistance_cp, edge_cp, be_cp = findProjectionPoint(G, (ctpe[0][0], ctpe[0][1], -1), close_to_edge = edge, connecting = True, forward = True, temp_point = (p2["x"], p2["y"]), settings = settings, max_actual_dist = [float('inf'), (0,float('inf'))], crs= crs)
        if do_print: print("Projected point: {0}".format(projectedPoint_cp))
        if do_print: print("Closest distance: " + str(closestDistance_cp) + " meters.")
        if do_print: print("Projected onto edge: {0}".format(edge_cp))
        
        # Suppose the close_to_point_end is projected on the edge we are investigating (probably on a corner point). We cannot project our node on an edge. We can just add the GPS point to the geometry without relocating the connecting point. The current connecting point will still be close to the close_to_point_start. 
        if edge[0:3] == edge_cp[0:3]:
            if do_print: print("Close to point end is onto the newly created edge... No need for relocation of connection point.  ")
            edge_newgeom = LineString(edge_points_list[0:(be+1)] +  [(point[1], point[0])] + edge_points_list[(be+1):])
            # Adjust the geometry of the new edge (include point on the right spot)
            G.edges[edge[0], edge[1], edge[2]]['geometry'] = edge_newgeom  
            G.edges[edge[0], edge[1], edge[2]]['length'] = computeLengthLinestring(edge_newgeom, method = 'euclidean')
            edge = (edge[0], edge[1], edge[2], edge_newgeom)
            return edge, points_to_be_adjusted, None, temp_new
        
        # So, at this point we know that the close_to point (projectedPoint_cp) is projected onto edge (edge_cp).
        
        # Remove the old edge, but keep its osmid/oneway
        id_keep = G.edges[edge[0], edge[1], edge[2]]['osmid']
        oneway_keep = G.edges[edge[0], edge[1], edge[2]]['oneway']
        datvel_keep = G.edges[edge[0], edge[1], edge[2]]['DatesVelocities']
        G.remove_edge(edge[0], edge[1], key = edge[2])

        # Update the ctpe for next rounds. Since we add another point on the last line piece, we change the lower bound of MLDC to the close_to point to zero (because we are getting closer to the close_to_point...)
        max_actual_dist_new = [ctpe[1][0], (0, ctpe[1][1][1])]
        
        # Find the new connection point 
        # Next, find the new connecting point (close to projectedPoint_cp on edge_cp) in a backward manner
        if do_print: print("--------------------------")
        if do_print: print("Find the new connecting point....")
        if do_print: print("Close to the point: {0}, which is projected onto edge (backward): {1}".format(projectedPoint_cp, edge_cp))
        projectedPoint_NEW, closestDistance_NEW, edge_NEW, be_NEW = findProjectionPoint(G, point, close_to_edge = edge_cp, connecting = True, forward = False, temp_point = projectedPoint_cp, settings = settings, max_actual_dist = max_actual_dist_new, crs= crs)
        ctpe = [ctpe[0], max_actual_dist_new] 
        if do_print: print("Projected point: {0}".format(projectedPoint_NEW))
        if do_print: print("Closest distance: " + str(closestDistance_NEW) + " meters.")
        if do_print: print("Projected onto edge: {0}".format(edge_NEW))
        if do_print: print("--------------------------")

        # Now that we have a new connection point, check if this connection point can be merged with an existing node. 
        node_to_be_merged = mergeNode(G, (projectedPoint_NEW[1], projectedPoint_NEW[0]), max_dist_merge = settings[2], indmax = settings[4])
        if node_to_be_merged is not None:
            if do_print: print("This projection point is merged with: " + str(node_to_be_merged))
            currentPoint = node_to_be_merged
            # In the case that we merged with the startpoint of a currently creating edge, we assume that this starting point is no longer explicitly added (as it has now two functions). NOte that the first point of [points_to_be_adjusted] is the temp_last_point, the starting point of a currently newly created edge. 
            if len(points_to_be_adjusted) > 0:
                if currentPoint == points_to_be_adjusted[0]: temp_new = False
        else: # Node can not be merged
            if do_print: print("This projection point is not merged, it is added explicitly.")
            # Explicitly add the new projection of the new node (on the last line piece) 
            currentPoint = str(edge[1]) # Use the same name as the old connection point. Note that this old node will be removed. If it could not be removed, because it was relevant, we can still distinguish them due to the | added to this node. 
            # Add the point explicitly. Note that edge_old_NEW is required to define possible edge_olds, as we are splitting edges that may have been the edge_old. 
            currentPoint, edge_old_NEW = add_point_expl(G, projectedPoint_NEW, edge_NEW, be_NEW, node_name = currentPoint, settings = settings, merge = False, two_way = settings[3], do_print = do_print)

        # Add the new edge (note that it is a new edge, so highway is in any case not defined (we use driven))
        keyNew = max([item[2] for item in G.edges(edge[0], currentPoint, keys = True) if ((item[0] == edge[0]) & (item[1] == currentPoint))], default=-1) + 1
        constructing_edge_newgeom = LineString(edge_points_list[:-1] +  [(point[1], point[0])] + [(G.nodes[currentPoint]['x'], G.nodes[currentPoint]['y'])])
        G.add_edge(edge[0], currentPoint, osmid = id_keep, new = True, driven = True, DatesVelocities = datvel_keep, ref = None, highway=None, oneway= oneway_keep, length = computeLengthLinestring(constructing_edge_newgeom, method = 'euclidean'), geometry = constructing_edge_newgeom, close_to_point_start = ctps, close_to_point_end = ctpe, maxspeed = None, service = None, bridge= None, lanes = None, u = edge[0], v = currentPoint, key = keyNew)
        
        # Remove the corresponding old connection point when: 
        # - the end node of the edge is not equal to the old connection point AND 
        # - the old end point is not still the starting point of the edge. 
        if (edge[1] != currentPoint) & (edge[1] != edge[0]):
            if do_print: print("Remove " + str(edge[1]) + " from the graph.")
            
            # Remove the point. 
            remove_point(G, old_point = edge[1], do_print = do_print)
            
            # If any of the relevant points was equal to the removed point, adjust it to the newly added node (e.g., temp_last_point).  
            points_adjusted = []
            for i in points_to_be_adjusted:
                if (i == edge[1]): points_adjusted += [currentPoint]
                else: points_adjusted += [i]
        
        # Define possible (if needed) new old edges
        incoming_edges = list(G.in_edges(edge[0], keys = True, data = 'geometry'))
        edge = (edge[0], currentPoint, keyNew, constructing_edge_newgeom) # Edge onto which the original point was projected.
        edge_old_NEW = incoming_edges + [edge]         
        
    else: # Add point to the interior of the geometry
        if do_print: print("Add point to a new edge (no relocation needed)..")
        # Find new geometry
        edge_newgeom = LineString(edge_points_list[0:(be+1)] +  [(point[1], point[0])] + edge_points_list[(be+1):])
        # Adjust the geometry of the new edge (include point on the right spot)
        G.edges[edge[0], edge[1], edge[2]]['geometry'] = edge_newgeom
        G.edges[edge[0], edge[1], edge[2]]['length'] = computeLengthLinestring(edge_newgeom, method = 'euclidean')
        edge = (edge[0], edge[1], edge[2], edge_newgeom)
        points_adjusted = points_to_be_adjusted

    return edge, points_adjusted, edge_old_NEW, temp_new

#TODO DONE
def add_point_expl(G, point, edge, be, node_name, settings, merge = False, two_way = False, do_print = False):
    """
    Add the [point] to the graph [G]. This point is added explicitly to the [edge] on the 
    [be] piece of this edge. In other words, [edge] is split in two pieces and [point] is 
    now the node inbetween. The name of this new node is: [node_name] when given. 

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    point : tuple (x,y)
        the point that will be inserted into the edge.
    edge : list (or tuple)
        (u, v, key, geometry)
        e.g., [6518847503, 7611774266, 0, <shapely.geometry.linestring.LineString object at 0x7ff2cd34fdc0>]
    be : number
        edge piece id at which the point will be inserted.
    node_name: String
        The name of the new node. 
    merge : Boolean
        If True, this means that we already have the intermediate node. We split the 
        edge in two pieces and reconnect them via the existing node
    two_way : Boolean
        If True, this means that we try to add the point in two ways. 
         
    Returns
    -------
    None, this function adjusts the input graph G
    
    """
    # If node is not merged with an existing node, add the node to the network
    if not merge: 
        # Determine the name (check whether the name already exists (should not happen))
        while node_name in G.nodes:
            node_name = node_name + "|"
        G.add_node(node_name, y = point[1], x = point[0], osmid = node_name, geometry = Point(point[0], point[1]))
    
    # Split the connecting edge in two edges from -> node_name -> to (projected point is not merged) #TODO Could improve using new_nodes, but will not have a mega-effect. not used that much
    if do_print: print("Split edge: {0}".format(edge))
    geom_1, geom_2 = split_edge(edge, be, point)
    
    # Keep relevant information from the to be splitted edge. These will be used for both the two new edges (that form this old edge)
    ctps = G[edge[0]][edge[1]][edge[2]]['close_to_point_start']
    ctpe = G[edge[0]][edge[1]][edge[2]]['close_to_point_end']
    new = G[edge[0]][edge[1]][edge[2]]['new']
    oneway = G[edge[0]][edge[1]][edge[2]]['oneway']
    highway = G[edge[0]][edge[1]][edge[2]]['highway']
    datvel = G[edge[0]][edge[1]][edge[2]]['DatesVelocities']
    
    # If two_way is true, we add the node inbetween at least two opposite edges. Note that we do not change the geometries of the second_way edge as this will have a minimal effect. If the lines do not align exactly, this may cause a small (negligible) error in the length of the geometry.  
    if two_way: 
        # First, determine all edges that might be opposite (all close edges). If there is no, we only add one way
        a = get_nearest_edge_FULL(G, Point(point), settings[4], return_geom=True, return_dist=True)
        if len(a) > 1: # If there is at least one edge close, for each opposite edge that is within 1m, we add the point. 
            ind = 0 
            while a[ind][1] < 1:
                current_edge = a[ind][0]
                # The edge itself is never the opposite edge
                if current_edge[0:3] == edge[0:3]: 
                    ind += 1
                    if (ind >= len(a)): break
                    continue
                if do_print: print("Edge also merged in another edge.")
                # Add the point to this current (possible) opposite edge
                _, _, be = projectPointOnEdge(current_edge, (Point(point),-1), alpha_bar = None, max_dist = float('inf'))
                add_point_expl(G, point = point, edge = current_edge, be = be, node_name = node_name, settings = settings, merge = True, two_way = False, do_print = do_print)    
                # Check the next possible opposite point (if possible)
                ind += 1
                if (ind >= len(a)): break       
    
    # Determine the new keys of the two split edges
    keyNew_1 = max([item[2] for item in G.edges(edge[0], node_name, keys = True) if ((item[0] == edge[0]) & (item[1] == node_name))], default=-1) + 1
    keyNew_2 = max([item[2] for item in G.edges(node_name, edge[1], keys = True) if ((item[0] == node_name) & (item[1] == edge[1]))], default=-1) + 1
    
    # Create the two edges and remove the old non-splitted edge (note that both are now driven.)
    G.add_edge(edge[0], node_name, osmid = 'Edge_split_2: ' + str(edge[0]) + "_" + str(node_name), new = new, driven = True, DatesVelocities = datvel, ref = None, highway=highway, oneway= oneway, length = computeLengthLinestring(geom_1, method= "euclidean"), geometry = geom_1, close_to_point_start = ctps, close_to_point_end = 'End', maxspeed = None, service = None, bridge= None, lanes = None, u = edge[0], v = node_name, key = keyNew_1)
    G.add_edge(node_name, edge[1], osmid = 'Edge_split_2: ' + str(node_name) + "_" + str(edge[1]), new = new, driven = True, DatesVelocities = datvel, ref = None, highway=highway, oneway= oneway, length = computeLengthLinestring(geom_2, method= "euclidean"), geometry = geom_2, close_to_point_start = 'Start', close_to_point_end = ctpe, maxspeed = None, service = None, bridge= None, lanes = None, u = node_name, v = edge[1], key = keyNew_2)
    G.remove_edge(edge[0], edge[1], edge[2])

    return node_name, [(edge[0], node_name, keyNew_1, geom_1), (node_name, edge[1], keyNew_2, geom_2)]

#TODO DONE
def remove_point(G, old_point, do_print = False):
    """
    When [old_point] is a node "without function", the [old_point] node is 
    removed from graph [G]. This is the case, when it is a node with 1 incoming 
    and 1 outgoing edge, Node --> [old_point] --> Node, or when 2 outgoing edges 
    go to the start points of 2 incoming edges, Node <==> [old_point] <==> Node.
    
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    old_point : string
        string that is related to a node of G. 

    Returns
    -------
    None, this function adjusts the input graph G
    """
    # Check if point is relevant for other edges
    incoming_edges = list(G.in_edges(old_point, keys = True, data = True))
    outgoing_edges = list(G.out_edges(old_point, keys = True, data = True))
    edges = incoming_edges+outgoing_edges
    edges = [e[0:3] for e in edges]
    
    # First check whether the node we would like to remove is an existing node (in OSM). If so, we do not remove it. 
    try: 
        int(old_point.split("/")[0])
        check_if_existing_node = True
    except: check_if_existing_node = False
    if check_if_existing_node: 
        if do_print: print("Point not removed, existed already in base map..")
    
    # If the point contains a self-edge, we do not remove the point. 
    elif len(edges) != len(set(edges)):
        if do_print: print("Point not removed, there is at least one self edge going from/to this node.")
            
    # If old_point has one incoming edge and one outgoing edge, we can remove this explicit node from the edge
    elif ((len(incoming_edges) == 1) & (len(outgoing_edges) == 1)): 
        # Remove subedges and node 
        G.remove_node(old_point)
        
        # Determine start and end edge of the new merged edge
        start_edge = incoming_edges[0]
        end_edge = outgoing_edges[0]
        
        # Keep relevant information from the to be merged edge. These will be used for the new merged edge
        ctps_1, ctpe_2 = start_edge[3]['close_to_point_start'], end_edge[3]['close_to_point_end']
        oneway = start_edge[3]['oneway']
        if start_edge[3]['oneway'] is not end_edge[3]['oneway']:
            print("Very strange, a splitted edge should have on both sides the same value for oneway..") 
            stop
        new = start_edge[3]['new'] & end_edge[3]['new'] # Both should be the same
        if start_edge[3]['new'] is not end_edge[3]['new']:
            print("Very strange, a splitted edge should have on both sides the same value for new..") 
            stop
        highway = start_edge[3]['highway'] # Both should be the same
        if start_edge[3]['highway'] is not end_edge[3]['highway']:
            print("Very strange, a splitted edge should have on both sides the same value for highway..")
            stop 
        driven = start_edge[3]['driven'] # Both should be the same and should be True
        if start_edge[3]['driven'] is not end_edge[3]['driven']:
            print("Very strange, a splitted edge should have on both sides the same value for highway..") 
            stop
        
        new_datvel = start_edge[3]['DatesVelocities']
        datvel = add_datvels(current_string = new_datvel, new_string = end_edge[3]['DatesVelocities'])

        # Determine full edge geometry. The location of old_point is preserved, but without an explicit node. 
        geom_1 = start_edge[3]['geometry']
        geom_2 = end_edge[3]['geometry']
        edge_points_list_1 = [y for y in geom_1.coords]
        edge_points_list_2 = [y for y in geom_2.coords]
        
        # If we are removing a point, we only want to keep the information of the point when applicable. If the point was a corner point of the edge (brng _diff >= 1), we keep the point in the geometry. If not, we know that it can be removed (it has no function, it was only included to create a new connection point). Note that if a new connection point was created on a corner point, this corner point was establsihed before, and we still want to include this corner point in the new geometry. 
        brng_before = computeBearing(Point(edge_points_list_1[-2]), Point(edge_points_list_1[-1]))
        brng_after = computeBearing(Point(edge_points_list_2[0]), Point(edge_points_list_2[1]))
        brng_diff = computeAngularDifference(brng_before,brng_after)        
        if (brng_diff < 1): edge_newgeom = LineString(edge_points_list_1[:-1] + edge_points_list_2[1:])
        else: edge_newgeom = LineString(edge_points_list_1[:-1] + edge_points_list_2)

        # Restore splitted edge
        keyNew = max([item[2] for item in G.edges(start_edge[0], end_edge[1], keys = True) if ((item[0] == start_edge[0]) & (item[1] == end_edge[1]))], default=-1) + 1
        G.add_edge(start_edge[0], end_edge[1], osmid = 'Edge_resplit: '+ str(start_edge[0]) + "_" + str(end_edge[1]), new = new, driven = driven, DatesVelocities = datvel, ref = None, highway=highway, oneway= oneway, length = computeLengthLinestring(edge_newgeom, method = 'euclidean'), geometry = edge_newgeom, close_to_point_start = ctps_1, close_to_point_end = ctpe_2, maxspeed = None, service = None, bridge= None, lanes = None, u = start_edge[0], v = end_edge[1], key = keyNew)
        if do_print: print("Edge {0} is added.".format((start_edge[0], end_edge[1], keyNew)))
        return [(start_edge[0], end_edge[1], keyNew, edge_newgeom)]
       
    # If old_point has two incoming and two outgoing edges, we can only remove old_point if the outgoing edges go to the start points of the incoming edges. 
    elif ((len(incoming_edges) == 2) & (len(outgoing_edges) == 2)):
        if do_print: print("Point has two incoming/outgoing edges. Examine whether double edge could be re-established.")
    
        # Define lists of start and end points of the incoming and outgoing edges
        start_points_incoming_edges = set([incoming_edges[0][0], incoming_edges[1][0]])
        end_points_outgoing_edges = set([outgoing_edges[0][1], outgoing_edges[1][1]])
        # Check if the point can be removed. The points to the "left" and "right" are the same. If so, the old_point can be removed. 
        if start_points_incoming_edges == end_points_outgoing_edges:
            # Remove subedges and node 
            G.remove_node(old_point)
            # Initialize return_edges, the list of the two recovered edges. 
            return_edges  = []
            
            # Add the two edges: from start_point of the first and second incoming edge. 
            for start_edge in (incoming_edges[0], incoming_edges[1]):
                # Determine which edges to connect. Note that for this, we compare the lengths of the subedges. Note that we cannot look at the end point and start point of the edge, as they might be the same, e.g., with a double point on a self-edge. 
                if round(outgoing_edges[0][3]['length'],2) == round(start_edge[3]['length'],2): end_edge = outgoing_edges[1]
                else: end_edge = outgoing_edges[0]
                
                # Keep relevant information from the to be merged edge. These will be used for the new merged edge
                ctps_1, ctpe_2 = start_edge[3]['close_to_point_start'], end_edge[3]['close_to_point_end']
                oneway = start_edge[3]['oneway']
                if start_edge[3]['oneway'] is not end_edge[3]['oneway']:
                    print("Very strange, a splitted edge should have on both sides the same value for oneway..") 
                    stop
                new = start_edge[3]['new']# & end_edge[3]['new']) # Both should be the same
                if start_edge[3]['new'] is not end_edge[3]['new']:
                    print("Very strange, a splitted edge should have on both sides the same value for new..") 
                    stop
                highway = start_edge[3]['highway'] # Both should be the same
                if start_edge[3]['highway'] is not end_edge[3]['highway']:
                    print("Very strange, a splitted edge should have on both sides the same value for highway..") 
                    stop
                driven = start_edge[3]['driven'] # Both should be the same and should be True
                if start_edge[3]['driven'] is not end_edge[3]['driven']:
                    print("Very strange, a splitted edge should have on both sides the same value for highway..")
                    stop
                
                new_datvel = start_edge[3]['DatesVelocities'] 
                datvel = add_datvels(current_string = new_datvel, new_string = end_edge[3]['DatesVelocities'])
                
                # Determine geometry of this edge. The location of old_point is preserved, but without an explicit node. 
                geom_1 = start_edge[3]['geometry']     
                geom_2 = end_edge[3]['geometry']
                edge_points_list_1 = [y for y in geom_1.coords]
                edge_points_list_2 = [y for y in geom_2.coords]
                
                # If we are removing a point, we only want to keep the information of the point when applicable. If the point was a corner point of the edge (brng _diff >= 1), we keep the point in the geometry. If not, we know that it can be removed (it has no function, it was only included to create a new connection point). Note that if a new connection point was created on a corner point, this corner point was establsihed before, and we still want to include this corner point in the new geometry. 
                brng_before = computeBearing(Point(edge_points_list_1[-2]), Point(edge_points_list_1[-1]))
                brng_after = computeBearing(Point(edge_points_list_2[0]), Point(edge_points_list_2[1]))
                brng_diff = computeAngularDifference(brng_before,brng_after)
                if (brng_diff < 1): edge_newgeom = LineString(edge_points_list_1[:-1] + edge_points_list_2[1:])
                else: edge_newgeom = LineString(edge_points_list_1[:-1] + edge_points_list_2)
                
                # Restore splitted edge
                keyNew = max([item[2] for item in G.edges(start_edge[0], end_edge[1], keys = True) if ((item[0] == start_edge[0]) & (item[1] == end_edge[1]))], default=-1) + 1
                G.add_edge(start_edge[0], end_edge[1], osmid = 'Edge_resplit: '+ str(start_edge[0]) + "_" + str(end_edge[1]), new = new, driven = driven, DatesVelocities = datvel, ref = None, highway=highway, oneway= oneway, length = computeLengthLinestring(edge_newgeom, method = 'euclidean'), geometry = edge_newgeom, close_to_point_start = ctps_1, close_to_point_end = ctpe_2, maxspeed = None, service = None, bridge= None, lanes = None, u = start_edge[0], v = end_edge[1], key = keyNew)
                if do_print: print("Edge {0} is added.".format((start_edge[0], end_edge[1], keyNew)))         
    
                # Add the restored edges to the list of return edges.
                return_edges = return_edges + [(start_edge[0], end_edge[1], keyNew, edge_newgeom)]
            
            return return_edges
    
    # In any other situation, the old_point cannot be removed
    if do_print: print("Point cannot be removed, it is relevant for other edges.. ")
 
#TODO DONE
def relabel(edge, temp_last_point, node_to_be_merged, newNode):
    """
    This function adjusts the points that occur in [edge] and [temp_last_point], 
    that may change because the node [node_to_be_merged] is merged with another 
    node. The new name of this node is [newNode].

    Parameters
    ----------
    edge : (u, v, key, geometry)
        The edge (endpoints may be adjusted).
    temp_last_point: String
        temp_last_point that may needs to be adjusted.
    node_to_be_merged : String
        Old node name that is now merged. 
    newNode : String
        New name of the merged node.
    
    Returns
    -------
    edge : (u, v, key, geometry)
        The new adjusted edge (endpoints may be adjusted).
    temp_last_point: String
        The possibly changed temp_last_point.
        
    """
    # Adjust edge (if necessary)
    if edge is not None: 
        if (edge[0] == node_to_be_merged) & (edge[1] == node_to_be_merged): 
            edge = list(edge)
            edge[0], edge[1] = newNode, newNode
            edge = tuple(edge)
        elif edge[0] == node_to_be_merged: 
            edge = list(edge)
            edge[0] = newNode
            edge = tuple(edge)
        elif edge[1] == node_to_be_merged: 
            edge = list(edge)
            edge[1] = newNode
            edge = tuple(edge)
    
    # Adjust the temp_last_point (if necessary)
    if temp_last_point == node_to_be_merged: temp_last_point = newNode
    
    return edge, temp_last_point

"""------------------------------------------- PROJECTION FUNCTIONS ----------------------------------------"""
"""---------------------------------------------------------------------------------------------------------"""
#TODO DONE
def ExtractSettings(settings):
    """ Extract the settings: max_dist_projection, max_bearingdiff_projection, max_dist_merge, _, indmax """
    return settings[0:len(settings)]

#TODO DONE
def findProjectionPoint(G, point, close_to_edge, connecting, forward, temp_point, settings, max_actual_dist, crs, target_dist = None):
    """
    Project [point] onto an edge of graph [G]. This edge is chosen based on the 
    maximum distance covered (MDC) and Most Likely Distance Covered (MLDC), which are 
    stored in max_actual_dist, from the the point [temp_point], which lies on the [close_to_edge]. 
    Moreover, if a connection must be made with the network, [connecting] is set to True. We refer
    to the paper (or the code below) for what exactly is meant with this attribute.
         
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    point : tuple
        (y, x, course)
    temp_point : tuple
        (x, y)
        Point to which the projection must be close. This point lies on the close_to_edge. It is 
        needed to check onto which side of close_to_edge we must project. 
    close_to_edge : tuple
        (u, v, key, geometry)
    connecting : boolean
        indicating whether the point is projected in order to connect (True) or to absorb (False)
    forward : boolean
        indicating whether we look forward (True) or backward (False).
    settings : -
        rest of the algorithm settings. 
    max_actual_dist : list
        two numbers representing the maximum distance covered (based on MAX_SPEED) and the average distance that could be covered (based on AVERAGE_SPEED)
    crs: string
        Coordinate Reference System
    target_dist : float
        If we are not connecting, we migth want to project onto an edge that has a specific projection distance. 
   
    Returns
    -------
    projection : tuple 
        (projectedPoint (lat, long), closestDistance (number), edge (u, v, key, geometry, distance)  best_edge)
    
        Tuple that includes the distance to the projected point (closestDistance)
        the projected point itself (projectedPoint), the edge onto which the point is 
        projected and the index of the piece of the edge onto which the projected point lies (best_edge).
        If the point could not have been projected, (None, None, None, None) is returned. 
    """
    # Extract settings
    max_dist_projection, max_bearingdiff_projection, max_dist_merge, _, indmax, _ = ExtractSettings(settings)
    try: curdatvel = point[3]
    except: curdatvel = "NN"
    # Initialize parameters (used everywhere)
    p = Point(reversed(point[0:2]))
    point = (p, point[2])
    if close_to_edge is not None: 
        p_temp = Point(temp_point) 
        close_to_edge = (close_to_edge[0], close_to_edge[1], close_to_edge[2], close_to_edge[3])
    
    # Create dataframe with options. If there is a previous point to which we want to be close, also incorporate the distance from this previous point to the current point
    if close_to_edge is not None: df = pd.DataFrame(columns = ["Edge", "ClosestDistance", "ProjectedPoint", "BestEdge", "DistanceFromOld", "DiffDistanceFromOld", "EdgesDriven"])
    else: df = pd.DataFrame(columns = ["Edge", "ClosestDistance", "ProjectedPoint", "BestEdge", "EdgesDriven"])
    
    # The point does not establish a connecting edge between a new edge and the current graph
    if not connecting:
        # Find the possible close edges and remove impossible edges
        closestEdges = get_nearest_edge_FULL(G, p, indmax = indmax, return_geom = True, return_dist = True)
        closestEdges = [a for a in closestEdges if a[1] < max_dist_projection[1]]
        closestEdges = closestEdges[0:50] # To save computation time
        
        # Initialize parameters
        dist, first_dist, ind = 0, 0, 0

        # Examine all edges that are close enough for point p to be absorbed (note that this also includes existing edges not satisfying their threshold )
        while (first_dist < max_dist_projection[1]) & (ind < len(closestEdges)):
            # Get closest edge
            (u, v, key, geom), first_dist = closestEdges[ind]
            closestEdge = (u, v, key, geom, first_dist)

            # We can immediately disregard existing edges that are not within distance 
            if (G.edges[u, v, key]['new'] is False) & (first_dist > max_dist_projection[0]): 
                ind += 1
                continue 
            # Set the maximum difference in bearing and the maximum projection distance (depends on whether edge is new or not)
            if (G.edges[u, v, key]['new'] is False):  # Absorbing by an existing edge
                bearing_diff = max_bearingdiff_projection[0]
                max_dist = max_dist_projection[0]
            else: 
                bearing_diff = max_bearingdiff_projection[1]
                max_dist = max_dist_projection[1]  
            
            # Determine all candidate projection points. Points that satisfy the maximum projection distance (max_dist) and the maximum difference in bearing (bearing_diff)
            candidatePoints = projectPointOnEdge(closestEdge, point, bearing_diff, max_dist)
            
            # For each option (note that an edge could have multiple options (there is, however, only one for each line piece)), we check if it is a possible edge for projection.
            for option in candidatePoints:
                # Extract information of this option
                dist, projected_point, be = option
            
                # If we must be close to a previous point, we compute the distance from the previous (forward).
                if close_to_edge is not None: 
                    l, edges_driven, _, _ = get_SP_distance(G, from_point = (temp_point, close_to_edge), to_point = (projected_point, closestEdge)) 
                    # If we can reach this point with maximum speed, consider it as a possible edge 
                    if (l <= max_actual_dist[0]): 
                        # Check the difference from the most likely distance covered from the old point (ddfo). If it is in the range, set it equal to zero, otherwise set it equal to the smallest difference to this MLDC.
                        if (max_actual_dist[1][0]) <= l <= (max_actual_dist[1][1]): ddfo = 0
                        else: ddfo = min(abs(l - max_actual_dist[1][0]), abs(l - max_actual_dist[1][1]))
                        df = df.append({'Edge': closestEdge, 'ClosestDistance': dist, 'ProjectedPoint': projected_point, 'BestEdge': be, 'DistanceFromOld': l, 'DiffDistanceFromOld': ddfo, 'EdgesDriven': edges_driven}, ignore_index = True)
                # If we do not need to be close to an edge, always consider the current projection as an option. The only edge that will be driven is the projection edge. 
                else: df = df.append({'Edge': closestEdge, 'ClosestDistance': dist, 'ProjectedPoint': projected_point, 'BestEdge': be, 'EdgesDriven': [closestEdge[0:3]]}, ignore_index = True)   

            # Examine the next closest edge, increase the ind
            ind += 1   

        # If there is a target distance, choose the one with the smallest deviation from this target distance
        if target_dist is not None:
            df['TargetDiff'] = abs(df.ClosestDistance - target_dist)
            if len(df) > 0: df = df[df.TargetDiff == min(df.TargetDiff)].reset_index(drop=True)
        

        # If we do not need to be close to a previous point, this means that we are currently constructing a new edge. We only look for edges 
        # that are close to any edge (satisfying the max_dist_projection)
        if close_to_edge is None: 
            # Return the nearest edge (incorporating direction) if there is at least one edge left 
            if len(df) > 0:
                df = df[df.ClosestDistance == min(df.ClosestDistance)].reset_index(drop=True)
                #df['Edge_string'] = df['Edge'].astype(str)
                #df.sort_values(by='Edge_string', inplace=True, ignore_index=True)
                # Define the edge to be returned
                edge_new = (df.Edge[0][0], df.Edge[0][1], df.Edge[0][2], df.Edge[0][3], df.Edge[0][4])
                # Ensure that the edge onto which the point is projected now gets indicated as "driven" (True)
                drivenEdge = df.EdgesDriven[0][0]
                if drivenEdge[0] != drivenEdge[1]:
                    G.edges[drivenEdge[0], drivenEdge[1], drivenEdge[2]]['driven'] = True
                    # Speed registration for the projection edge
                    #if curdatvel != "NN": G.edges[drivenEdge[0], drivenEdge[1], drivenEdge[2]]['DatesVelocities'] += "|" + curdatvel
                    if curdatvel != "NN": G.edges[drivenEdge[0], drivenEdge[1], drivenEdge[2]]['DatesVelocities'] = add_datvels(G.edges[drivenEdge[0], drivenEdge[1], drivenEdge[2]]['DatesVelocities'], curdatvel)
                    
                return (df.ProjectedPoint[0], df.ClosestDistance[0], edge_new , df.BestEdge[0])
            else: return (None, None, None, None) # Point cannot be projected
        
        # If we do need to be close to a previous point, we select based on the two anomalies. The projection distance to the road and the 
        # ddfo, both weighed equally. (note that the closestDistance already could be possible)
        else:
            if len(df) > 0:
                df['combinedDistance'] = df.ClosestDistance + df.DiffDistanceFromOld
                df = df[df.combinedDistance == min(df.combinedDistance)].reset_index(drop = True)
                #df['Edge_string'] = df['Edge'].astype(str)
                #df.sort_values(by='Edge_string', inplace=True, ignore_index=True)
                # If mutliple options have the same sum of anomalies, we prefer the one that lies on the same edge as the close_to_point lies. 
                if any(df.Edge.apply(lambda x: x[0:2]) == close_to_edge[0:2]): df = df[df.Edge.apply(lambda x: x[0:2]) == close_to_edge[0:2]].reset_index(drop=True)
                # If multiple options remain, choose the one with the smallest distance from the previous point. Another option would be to use ClosestDistance. 
                df = df[df.DistanceFromOld == min(df.DistanceFromOld)].reset_index(drop = True)
                # Define the edge to be returned
                edge_new = (df.Edge[0][0], df.Edge[0][1], df.Edge[0][2], df.Edge[0][3], df.Edge[0][4])
                # Ensure that the edges that have been traversed to get to this new point are classified as driven (True)
                for drivenEdge in df.EdgesDriven[0]: 
                    if drivenEdge[0] != drivenEdge[1]:
                        G.edges[drivenEdge[0], drivenEdge[1], drivenEdge[2]]['driven'] = True
                        
                # Only speed registration for the last driven edge (current projection)
                #if curdatvel != "NN": G.edges[edge_new[0], edge_new[1], edge_new[2]]['DatesVelocities'] += "|" + curdatvel
                if curdatvel != "NN": G.edges[edge_new[0], edge_new[1], edge_new[2]]['DatesVelocities'] = add_datvels(G.edges[edge_new[0], edge_new[1], edge_new[2]]['DatesVelocities'], curdatvel)
                return (df.ProjectedPoint[0], df.ClosestDistance[0], edge_new , df.BestEdge[0])   
            else: return (None, None, None, None) # Point cannot be projected (no point within distance)
    
    # If the point establishes a connecting edge between a new edge and the current graph, 
    # we incorporate the closeness to the close_to_edge. This means that close_to_edge MUST be defined. Note that we do not incorporate the bearing here (always -1)
    else:
        # Find the possible close edges
        closestEdges = get_nearest_edge_FULL(G, p, indmax = indmax, return_geom = True, return_dist = True)
        closestEdges = [a for a in closestEdges if a[1] < max_actual_dist[0]]
        closestEdges = closestEdges[0:50] # To save computation time
     
        # If no edges remain, we connect with the last point that was projected, the temp_point projected onto close_to_edge. Note that we must get an answer, there must be a road.
        if len(closestEdges) == 0: # Hopefully never/rarely occurs 
            # print("There is no edge to which the point could be connected that lies within max distance, just add a straight line...")
            closestDistance = sqrt((temp_point[0]-p.x)**2 + (temp_point[1] - p.y)**2)
            _, _, be = projectPointOnEdge(close_to_edge, (Point(temp_point), -1), alpha_bar = None, max_dist = float('inf'))
            return (temp_point, closestDistance, close_to_edge, be)
        
        # If at least one edge remains, we check for each edge and compute its projected distance
        for i in range(0, len(closestEdges)):
            # Get closest edge
            (u, v, key, geom), _ = closestEdges[i]
            closestEdge = (u, v, key, geom)
            
            # Determine all candidate projection points. Points that satisfy the maximum projection distance (max_dist). Note that there is no restriction on the maximum bearing difference as we ignore bearing (course = -1). If we are considering the close_to_edge (when this is the closestEdge), we incorporate p_temp into this geometry. In other words, for the projection on this edge, at the line piece where p_temp was projected, we now get two candidtate projecftion points.(e.g., 2395) 
            if (closestEdge[0:3] == close_to_edge[0:3]):
                # Find projection of p_temp
                p_temp_projection = projectPointOnEdge(closestEdge, (p_temp, -1), max_bearingdiff_projection[0], max_dist = float('inf'))       
                be_old = p_temp_projection[2]
                                
                # Create temporary (artifical) edge geometry
                edge_points_list = [y for y in closestEdge[3].coords]
                edge_newgeom = LineString(edge_points_list[0:(p_temp_projection[2]+1)]  + [(p_temp_projection[1][0], p_temp_projection[1][1])] + edge_points_list[(p_temp_projection[2]+1):])
                closestEdge_artgeom = (closestEdge[0], closestEdge[1], closestEdge[2], edge_newgeom)

                # Find candidate projection points based on the artificial edge
                candidatePoints_temp = projectPointOnEdge(closestEdge_artgeom, point, max_bearingdiff_projection[0], max_dist = max_actual_dist[0])
                if type(candidatePoints_temp) is tuple: candidatePoints_temp = [candidatePoints_temp] 
                
                # Change the be for some of the found candidate points (this means that we get now 2 candidate points for one edge piece)
                candidatePoints = []
                for candidate in candidatePoints_temp:
                     if (candidate[2] > be_old): candidatePoints += [(candidate[0],candidate[1], candidate[2]-1)]
                     else: candidatePoints += [candidate]
            else: candidatePoints = projectPointOnEdge(closestEdge, point, max_bearingdiff_projection[0], max_dist = max_actual_dist[0])
            if type(candidatePoints) is tuple: candidatePoints = [candidatePoints] 

            # For each option (note that an edge could have multiple options (there is, however, only one for each line piece)), we check if it is a possible edge for projection.
            for option in candidatePoints: 
                # Extract information of this option
                dist, projected_point, be = option
                
                # For each edge, we compute the distance from the previous (forward) or distance to the next (backwards) point. 
                if forward: 
                    l, edges_driven, _, _ = get_SP_distance(G, from_point = (temp_point, close_to_edge), to_point = (projected_point, closestEdge))
                else: # Backwards
                    l, edges_driven, _, _ = get_SP_distance(G, from_point = (projected_point, closestEdge), to_point = (temp_point, close_to_edge))
                
                # Add the projection distance
                l += dist
         
                # If we can reach this point with maximum speed, consider it as a possible edge 
                if ((l <= max_actual_dist[0])):
                    # Check the difference from the most likely distance covered from the old point (ddfo). If it is in the range, set it equal to zero, otherwise set it equal to the smallest difference to this MLDC.
                    if (max_actual_dist[1][0]) <= l <= (max_actual_dist[1][1]): ddfo = 0
                    else: ddfo = min(abs(l - max_actual_dist[1][0]), abs(l - max_actual_dist[1][1]))
                    df = df.append({'Edge': closestEdge, 'ClosestDistance': dist, 'ProjectedPoint': projected_point, 'BestEdge': be, 'DistanceFromOld': l, 'DiffDistanceFromOld': ddfo, 'EdgesDriven': edges_driven}, ignore_index = True)
        
        # We select based on the two anomalies. The projection distance to the road and the (ddfo), both weighed equally. 
        if len(df) > 0:
            df['combinedDistance'] = df.ClosestDistance + df.DiffDistanceFromOld
            df = df[df.combinedDistance == min(df.combinedDistance)].reset_index(drop = True)  
            #df['Edge_string'] = df['Edge'].astype(str)
            #df.sort_values(by='Edge_string', inplace=True, ignore_index=True)
            # If mutliple options have the same sum of anomalies, we prefer the one that lies on the same edge as the close_to_point lies. 
            if any(df.Edge.apply(lambda x: x[0:2]) == close_to_edge[0:2]): df = df[df.Edge.apply(lambda x: x[0:2]) == close_to_edge[0:2]].reset_index(drop=True)
            # If multiple options remain, choose the one with the smallest distance. Another option would be to use ClosestDistance. 
            df = df[df.DistanceFromOld == min(df.DistanceFromOld)].reset_index(drop = True)  
            # Define the edge to be returned
            edge_new = (df.Edge[0][0], df.Edge[0][1], df.Edge[0][2], df.Edge[0][3])
            # Ensure that the edges that have been traversed to get to this new point are classified as driven (True)
            for drivenEdge in df.EdgesDriven[0]: 
                if drivenEdge[0] != drivenEdge[1]:
                    G.edges[drivenEdge[0], drivenEdge[1], drivenEdge[2]]['driven'] = True
            return (df.ProjectedPoint[0], df.ClosestDistance[0], edge_new , df.BestEdge[0])  
        else: # Point could not be projected close to the temp_point onto edge_old. We just return the temp_point as our projection point. Note that close_to_edge is already driven..
            _, _, be = projectPointOnEdge(close_to_edge, (Point(temp_point), -1), alpha_bar = None)
            return (temp_point, point[0].distance(Point(temp_point)), close_to_edge, be)

#TODO DONE
def projectPointOnEdge(edge, point, alpha_bar = None, max_dist = float('inf')): 
    """
    Given an [edge] of a graph, this function determines all (one per line piece) candidate 
    points for the projection of [point] onto this edge. A candidate point must satisfy the 
    maximum "bearing difference" and "projection distance" constraint. 

    Parameters
    ----------
    edge : tuple
        (u, v, key, geometry)
    point : tuple
        (Point(x,y), Course), if the course = -1, no maximum angle difference is included. 
    alpha_bar : float
        The maximum difference in bearing (see SETTINGS).
    max_dist 
        The maximum projection distance

    Returns
    -------
    candidatePoints tuple (if no max_distance_constraint) / list of tuples
        (closestDistance, projectedPoint (x, y), counter) (or a list of these)
    
        Tuple(s) that include the distance to the projected point (closestDistance)
        the projected point itself (projectedPoint) and the index of the piece of 
        the edge onto which the projected point lies (counter).
    """
    # Extract the location from point
    projPoint = point[0]
    
    # Create list of all points in the geometry of the edge
    edge_points_list = edge[3].coords

    # Initialize parameters
    closestDistance, prev_node, counter = float('inf'), edge_points_list[0], 0

    candidatePoints = []
    for e in edge_points_list[1:]:
        # Find the closest (closest) projection distance onto this edge piece. (Note that the bearing is the same on this complete piece)
        if prev_node == e: # The edge consists of just one point. This point is the closest point. 
            distance = sqrt((prev_node[0]-projPoint.x)**2+(prev_node[1]-projPoint.y)**2)
            x, y = prev_node[0], prev_node[1]
        else: x, y, distance = dist(prev_node[0], prev_node[1], e[0], e[1], projPoint.x, projPoint.y)
            
        # If there is no restriction on bearing (bearing = -1)
        if (point[1] == -1):
            # Check for every line piece (no bearing restriction) whether it satisfies the maximum projection distance. 
            if max_dist == float('inf'): # No restriction on the projection distance (return only the point with the smallest projection distance)
                if (distance < closestDistance):
                    candidatePoints = (distance, (x,y), counter)
                    closestDistance = distance
            else: # There is a restriction on the projection distance, return all points that satisfy this restriction (add to the list). 
                if distance <= max_dist: 
                    candidatePoints += [(distance, (x,y), counter)]
        else: # There is a restriction on the bearing 
            # Determine the bearing of the line piece
            start_point, end_point = Point(prev_node[0],prev_node[1]), Point(e[0], e[1])
            brng = computeBearing(start_point, end_point)
            if max_dist == float('inf'): # No restriction on projection distance (return only the point with the smallest projection distance that satisfies the maximum bearing difference)
                if (distance < closestDistance) & (computeAngularDifference(brng, point[1]) <= alpha_bar):
                    candidatePoints = (distance, (x,y), counter)
                    closestDistance = distance     
            else:# There is a restriction on the projection distance, return all points that satisfy this restriction and the maximum bearing difference restriction (add to the list). 
                if (distance <= max_dist) & (computeAngularDifference(brng, point[1]) <= alpha_bar):
                    candidatePoints += [(distance, (x,y), counter)]
                    
        # Go to the next line piece, increase the edge counters
        prev_node = e
        counter += 1
    
    return candidatePoints

        

#TODO DONE
def get_SP_distance_TIME(G, from_point, to_point, SPV):
    """
    This function computes the shortest path between [from_point] and [to_point]
    using Graph [G]. 
    
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    from_point : tuple 
        (point (x,y), edge (u, v, key, geometry))
        The [point] must be projected (or is projected) onto this [edge]. 
    to_point : tuple
        (point (x,y), edge (u, v, key, geometry))
        The [point] must be projected (or is projected) onto this [edge].
    
    Returns
    -------
    length : float
        The shortest distance from the [from_point] to the [to_point], when both 
        are projected onto their current edges. 
    driven_paths : list
        list of edges (u, v, key, geometry) that are used in the shortest path 
        (geometries are cut to the piece that is used in the SP)
    projected_point_from : (x, y)
        The projection point of the [from_point] used in the SP
    projected_point_to : (x, y)
        The projection point of the [to_point] used in the SP
    """
    # Initialize the projection points
    projected_point_from, projected_point_to = None, None
    dist_to_end_point = 1 # if distance to one of the end point is smaller than or equal to this number, it is considered as this node. 
    
    # Check if from_point is an endpoint of the edge
    merge_from = None
    dist_from_1 = sqrt((from_point[0][0]-G.nodes[from_point[1][0]]['x'])**2+(from_point[0][1]-G.nodes[from_point[1][0]]['y'])**2) # euclidean distance to the start point
    dist_from_2 = sqrt((from_point[0][0]-G.nodes[from_point[1][1]]['x'])**2+(from_point[0][1]-G.nodes[from_point[1][1]]['y'])**2) # euclidean distance to the end point
    if (dist_from_1 < dist_from_2) & (dist_from_1 <= dist_to_end_point): 
        merge_from = from_point[1][0] # from point is equal to the start point of the corresponding edge
        projected_point_from = (G.nodes[merge_from]['x'], G.nodes[merge_from]['y'])
    elif (dist_from_2 <= dist_from_1) & (dist_from_2 <= dist_to_end_point): 
        merge_from = from_point[1][1] # from point is equal to the end point of the corresponding edge
        projected_point_from = (G.nodes[merge_from]['x'], G.nodes[merge_from]['y'])
         
    # Check if to_point is an endpoint of the edge
    merge_to = None
    dist_to_1 = sqrt((to_point[0][0]-G.nodes[to_point[1][0]]['x'])**2+(to_point[0][1]-G.nodes[to_point[1][0]]['y'])**2) # euclidean distance to the start point
    dist_to_2 = sqrt((to_point[0][0]-G.nodes[to_point[1][1]]['x'])**2+(to_point[0][1]-G.nodes[to_point[1][1]]['y'])**2) # euclidean distance to the end point
    if (dist_to_1 < dist_to_2) & (dist_to_1 <= dist_to_end_point): 
        merge_to = to_point[1][0] # to point is equal to the start point of the corresponding edge
        projected_point_to = (G.nodes[merge_to]['x'], G.nodes[merge_to]['y'])
    elif (dist_to_2 <= dist_to_1) & (dist_to_2 <= dist_to_end_point): 
        merge_to = to_point[1][1] # to point is equal to the end point of the corresponding edge
        projected_point_to = (G.nodes[merge_to]['x'], G.nodes[merge_to]['y'])

    # If [from_point] is not merged, we find the projection point and split the projection edge
    if merge_from is None: 
        from_p = (Point(from_point), -1)
        dist, projectedPoint, best_edge = projectPointOnEdge(from_point[1], from_p, alpha_bar = None, max_dist = float('inf'))
        a_from, b_from = split_edge(from_point[1], best_edge, projectedPoint)
        projected_point_from = projectedPoint
        
    # If [to_point] is not merged, we find the projection point and split the projection edge
    if merge_to is None:
        to_p = (Point(to_point), -1)
        dist, projectedPoint, best_edge = projectPointOnEdge(to_point[1], to_p, alpha_bar = None, max_dist = float('inf'))
        a_to, b_to = split_edge(to_point[1], best_edge, projectedPoint)
        projected_point_to = projectedPoint

    # Initialize parameters
    skip_path_inbetween, traveltime, length,  driven_paths = False, 0,0, []
    # Examine the part of the edges onto which the endpoints are projected (a_from, b_from) and (a_to, b_to)
    # Both from_point and to_point were not be merged with an existing node. 
    if (merge_from is None) & (merge_to is None):
        # If both points are projected onto the same edge, we can take "the middle part" as the shortest distance
        if (from_point[1][0:3] == to_point[1][0:3]):
            # Compute the length of the edge inbetween the two points
            length += computeLengthLinestring(b_from, method = "euclidean") - computeLengthLinestring(b_to, method = "euclidean")
            # If the length >= 0, we can drive over this middle part. Determine the geometry and we can skip the "part inbetween"  
            if length >= 0:
                b_from_coords = list(b_from.coords)
                b_to_coords = list(b_to.coords)
                right_elements_to_be_removed = len(b_to_coords)-1
                # print("1")
                
                #TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO
                speed = from_point[1][4+SPV]/3.6 # is equal to to_point[1][4+SPV]/3.6 
                traveltime += ((computeLengthLinestring(b_from, method = "euclidean") - computeLengthLinestring(b_to, method = "euclidean"))/speed)/60
                #TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO
                
                driven_paths += [from_point[1][0:3] + tuple([LineString(b_from_coords[0:(len(b_from_coords)-right_elements_to_be_removed)]+[b_to_coords[0]])])] 
                skip_path_inbetween = True # Do not add edges between the from_point and to_point
            
            else: # If the length < 0, it means that it is the wrong direction. Then, we check the length when going in the opposite direction and maybe ending up at the end point again. 
                length = computeLengthLinestring(b_from, method = "euclidean") + computeLengthLinestring(a_to, method = "euclidean")
                # print("2")
                #TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO
                speed = from_point[1][4+SPV]/3.6 # is equal to to_point[1][4+SPV]/3.6 
                traveltime += ((computeLengthLinestring(b_from, method = "euclidean")+computeLengthLinestring(a_to, method = "euclidean"))/speed)/60
                #TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO

                driven_paths += [from_point[1][0:3] + tuple([b_from]), to_point[1][0:3] + tuple([a_to])]
                
        else: # If both are not projected onto the same edge, only consider the small pieces of the projection edges. 
            length += computeLengthLinestring(b_from, method = "euclidean") + computeLengthLinestring(a_to, method = "euclidean")
            # print(3)
            #TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO
            speed_from = from_point[1][4+SPV]/3.6
            traveltime += (computeLengthLinestring(b_from, method = "euclidean")/speed_from)/60
            speed_to = to_point[1][4+SPV]/3.6
            traveltime += (computeLengthLinestring(a_to, method = "euclidean")/speed_to)/60
            #TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO
            
            driven_paths += [from_point[1][0:3] + tuple([b_from]), to_point[1][0:3] + tuple([a_to])]
            
        # Define the origin and target node for the "part inbetween"
        orig_node = from_point[1][1] # End point of the edge onto which [old_point] was projected
        target_node = to_point[1][0] # Starting point of the edge onto which [current_point] was projected
    # Only the from_point could not be merged with an existing node. 
    elif (merge_from is None) & (merge_to is not None):
        length += computeLengthLinestring(b_from, method = "euclidean")
        # print(4)
        #TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO
        speed = from_point[1][4+SPV]/3.6
        traveltime += (computeLengthLinestring(b_from, method = "euclidean")/speed)/60
        #TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO
        
        driven_paths += [from_point[1][0:3] + tuple([b_from])]
        orig_node = from_point[1][1]
        target_node = merge_to
    # Only the to_point could not be merged with an existing node. 
    elif (merge_from is not None) & (merge_to is None):
        length += computeLengthLinestring(a_to, method = "euclidean")
        # print(5)
        #TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO
        speed = to_point[1][4+SPV]/3.6
        traveltime += (computeLengthLinestring(a_to, method = "euclidean")/speed)/60
        #TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO#TODO
        
        driven_paths += [to_point[1][0:3] + tuple([a_to])]
        orig_node = merge_from
        target_node = to_point[1][0]
    # If both points were merged with nodes in the graph.
    elif (merge_from is not None) & (merge_to is not None):
        # print(6)
        orig_node = merge_from
        target_node = merge_to
    
    
    
    speeds_options = ['TravelTime_00', 'TravelTime_01', 'TravelTime_10', 'TravelTime_11', 'TravelTime_20', 'TravelTime_21', 'TravelTime_AVG']
    # Examine the "part inbetween" the end node of the from_edge and the from node of the to_edge (only when both points were not projected on the same edge with the right direction). 
    if not skip_path_inbetween:
        try: 
            #sp = nx.shortest_path(G, source=orig_node, target=target_node, weight='length')
            #length += nx.path_weight(G, sp, weight="length")
            sp = nx.shortest_path(G, source=orig_node, target=target_node, weight=speeds_options[SPV])
            traveltime += nx.path_weight(G, sp, weight=speeds_options[SPV]) #minutes
            
            for u, v in zip(sp[:-1], sp[1:]):
                a = G.get_edge_data(u, v)
                #key = list(a.keys())[list(a.values()).index(min(a.values(), key = lambda d: d["length"]))]
                key = list(a.keys())[list(a.values()).index(min(a.values(), key = lambda d: d[speeds_options[SPV]]))]
                driven_paths += [(u,v,key,G.edges[u,v,key]['geometry'])]
                
        except: # No (shortest) path exists 
            traveltime = float('inf')

    return traveltime, driven_paths, projected_point_from, projected_point_to







#TODO DONE
def get_SP_distance(G, from_point, to_point):
    """
    This function computes the shortest path between [from_point] and [to_point]
    using Graph [G]. 
    
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    from_point : tuple 
        (point (x,y), edge (u, v, key, geometry))
        The [point] must be projected (or is projected) onto this [edge]. 
    to_point : tuple
        (point (x,y), edge (u, v, key, geometry))
        The [point] must be projected (or is projected) onto this [edge].
    
    Returns
    -------
    length : float
        The shortest distance from the [from_point] to the [to_point], when both 
        are projected onto their current edges. 
    driven_paths : list
        list of edges (u, v, key, geometry) that are used in the shortest path 
        (geometries are cut to the piece that is used in the SP)
    projected_point_from : (x, y)
        The projection point of the [from_point] used in the SP
    projected_point_to : (x, y)
        The projection point of the [to_point] used in the SP
    """
    # Initialize the projection points
    projected_point_from, projected_point_to = None, None
    dist_to_end_point = 1 # if distance to one of the end point is smaller than or equal to this number, it is considered as this node. 
    
    # Check if from_point is an endpoint of the edge
    merge_from = None
    dist_from_1 = sqrt((from_point[0][0]-G.nodes[from_point[1][0]]['x'])**2+(from_point[0][1]-G.nodes[from_point[1][0]]['y'])**2) # euclidean distance to the start point
    dist_from_2 = sqrt((from_point[0][0]-G.nodes[from_point[1][1]]['x'])**2+(from_point[0][1]-G.nodes[from_point[1][1]]['y'])**2) # euclidean distance to the end point
    if (dist_from_1 < dist_from_2) & (dist_from_1 <= dist_to_end_point): 
        merge_from = from_point[1][0] # from point is equal to the start point of the corresponding edge
        projected_point_from = (G.nodes[merge_from]['x'], G.nodes[merge_from]['y'])
    elif (dist_from_2 <= dist_from_1) & (dist_from_2 <= dist_to_end_point): 
        merge_from = from_point[1][1] # from point is equal to the end point of the corresponding edge
        projected_point_from = (G.nodes[merge_from]['x'], G.nodes[merge_from]['y'])
         
    # Check if to_point is an endpoint of the edge
    merge_to = None
    dist_to_1 = sqrt((to_point[0][0]-G.nodes[to_point[1][0]]['x'])**2+(to_point[0][1]-G.nodes[to_point[1][0]]['y'])**2) # euclidean distance to the start point
    dist_to_2 = sqrt((to_point[0][0]-G.nodes[to_point[1][1]]['x'])**2+(to_point[0][1]-G.nodes[to_point[1][1]]['y'])**2) # euclidean distance to the end point
    if (dist_to_1 < dist_to_2) & (dist_to_1 <= dist_to_end_point): 
        merge_to = to_point[1][0] # to point is equal to the start point of the corresponding edge
        projected_point_to = (G.nodes[merge_to]['x'], G.nodes[merge_to]['y'])
    elif (dist_to_2 <= dist_to_1) & (dist_to_2 <= dist_to_end_point): 
        merge_to = to_point[1][1] # to point is equal to the end point of the corresponding edge
        projected_point_to = (G.nodes[merge_to]['x'], G.nodes[merge_to]['y'])

    # If [from_point] is not merged, we find the projection point and split the projection edge
    if merge_from is None: 
        from_p = (Point(from_point), -1)
        dist, projectedPoint, best_edge = projectPointOnEdge(from_point[1], from_p, alpha_bar = None, max_dist = float('inf'))
        a_from, b_from = split_edge(from_point[1], best_edge, projectedPoint)
        projected_point_from = projectedPoint
        
    # If [to_point] is not merged, we find the projection point and split the projection edge
    if merge_to is None:
        to_p = (Point(to_point), -1)
        dist, projectedPoint, best_edge = projectPointOnEdge(to_point[1], to_p, alpha_bar = None, max_dist = float('inf'))
        a_to, b_to = split_edge(to_point[1], best_edge, projectedPoint)
        projected_point_to = projectedPoint

    # Initialize parameters
    skip_path_inbetween, length, driven_paths = False, 0, []
    # Examine the part of the edges onto which the endpoints are projected (a_from, b_from) and (a_to, b_to)
    # Both from_point and to_point were not be merged with an existing node. 
    if (merge_from is None) & (merge_to is None):
        # If both points are projected onto the same edge, we can take "the middle part" as the shortest distance
        if (from_point[1][0:3] == to_point[1][0:3]):
            # Compute the length of the edge inbetween the two points
            length += computeLengthLinestring(b_from, method = "euclidean") - computeLengthLinestring(b_to, method = "euclidean")
            # If the length >= 0, we can drive over this middle part. Determine the geometry and we can skip the "part inbetween"  
            if length >= 0:
                # print(1)
                b_from_coords = list(b_from.coords)
                b_to_coords = list(b_to.coords)
                right_elements_to_be_removed = len(b_to_coords)-1
                driven_paths += [from_point[1][0:3] + tuple([LineString(b_from_coords[0:(len(b_from_coords)-right_elements_to_be_removed)]+[b_to_coords[0]])])] 
                skip_path_inbetween = True # Do not add edges between the from_point and to_point
            else: # If the length < 0, it means that it is the wrong direction. Then, we check the length when going in the opposite direction and maybe ending up at the end point again. 
                # print(2)
                length = computeLengthLinestring(b_from, method = "euclidean") + computeLengthLinestring(a_to, method = "euclidean")
                driven_paths += [from_point[1][0:3] + tuple([b_from]), to_point[1][0:3] + tuple([a_to])]
        else: # If both are not projected onto the same edge, only consider the small pieces of the projection edges. 
            # print(3)
            length += computeLengthLinestring(b_from, method = "euclidean") + computeLengthLinestring(a_to, method = "euclidean")
            driven_paths += [from_point[1][0:3] + tuple([b_from]), to_point[1][0:3] + tuple([a_to])]            
            
        # Define the origin and target node for the "part inbetween"
        orig_node = from_point[1][1] # End point of the edge onto which [old_point] was projected
        target_node = to_point[1][0] # Starting point of the edge onto which [current_point] was projected
    # Only the from_point could not be merged with an existing node. 
    elif (merge_from is None) & (merge_to is not None):
        # print(4)
        length += computeLengthLinestring(b_from, method = "euclidean")
        driven_paths += [from_point[1][0:3] + tuple([b_from])]
        orig_node = from_point[1][1]
        target_node = merge_to
    # Only the to_point could not be merged with an existing node. 
    elif (merge_from is not None) & (merge_to is None):
        # print(5)
        length += computeLengthLinestring(a_to, method = "euclidean")
        driven_paths += [to_point[1][0:3] + tuple([a_to])]
        orig_node = merge_from
        target_node = to_point[1][0]
      
    # If both points were merged with nodes in the graph.
    elif (merge_from is not None) & (merge_to is not None):
        # print(6)
        orig_node = merge_from
        target_node = merge_to
    
    # Examine the "part inbetween" the end node of the from_edge and the from node of the to_edge (only when both points were not projected on the same edge with the right direction). 
    if not skip_path_inbetween:
        try: 
            sp = nx.shortest_path(G, source=orig_node, target=target_node, weight='length')
            length += nx.path_weight(G, sp, weight="length")

            for u, v in zip(sp[:-1], sp[1:]):
                a = G.get_edge_data(u, v)
                key = list(a.keys())[list(a.values()).index(min(a.values(), key = lambda d: d["length"]))]
                driven_paths += [(u,v,key,G.edges[u,v,key]['geometry'], G.edges[u,v,key]['length'])]

        except: # No (shortest) path exists
            length = float('inf')

    return length, driven_paths, projected_point_from, projected_point_to
    
    
    
    
    
    
#TODO Done
def split_edge(edge, be, projectedPoint):
    """
     Splits an edge at a given point
    
     Parameters
     ----------
     edge : tuple  
         (u, v, key, geometry) 
     be : float
         edge piece id at which the projected point lies. 
     projectedPoint: tuple (x,y)
        point that splits the edge
    
     Returns
     -------
     Two geometries that represent the two (splitted) edges 
    """
    edge_points_list = [y for y in edge[3].coords]
    edge_1_geom = edge_points_list[0:(be+1)] + [projectedPoint]
    edge_2_geom = [projectedPoint] + edge_points_list[(be+1):]
        
    return LineString(edge_1_geom), LineString(edge_2_geom)

#TODO DONE
def get_nearest_edge_FULL(G, point, indmax, return_geom=False, return_dist=False):
    """
    Find the nearest edges to a point by minimum Euclidean distance.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    point : Point(x,y)
        the point for which we will find the nearest edge in the graph
    indmax : list
        [nodes_used_extended, initial_G_extended.nodes]
        determines which nodes must be used in the subgraph. All nodes that are originally 
        considered (based on nodes_used_extended in the used polygon) and the new nodes 
    return_geom : bool
        Optionally return the geometry of the nearest edge
    return_dist : bool
        Optionally return the distance in meters between
        the point and the nearest edge

    Returns
    -------
    tuple
        Graph edge unique identifier as a tuple of (u, v, key).
        Or a tuple of (u, v, key, geom) if return_geom is True.
        Or a tuple of (u, v, key, dist) if return_dist is True.
        Or a tuple of (u, v, key, geom, dist) if return_geom and return_dist are True.
    """
    # Get the nodes that are within our specific region for this ID. Nodes that still exist (within the original polygon) + nodes that are added
    #new_nodes = list(indmax[0] & G.nodes) + list(set(G.nodes) - set(indmax[1]))
    l1, l2 = set(G.nodes), set(indmax[1])
    new_nodes = list(indmax[0] & G.nodes) + [item for item in l1 if item not in l2]
    
    # Create a subgraph
    edges = G.subgraph(new_nodes).edges(data='geometry', keys = True)
    
    # If there are no edges in the subgraph, no close edges will be returned 
    if len(edges) == 0: return [] 
    
    # Compute perpendicular distances and sort the dataframe that will be returned
    edge_distances = [(edge, point.distance(edge[3])) for edge in edges]
    edge_distances_sorted = sorted(edge_distances, key = lambda x: (x[1], str(x[0])))
    return edge_distances_sorted

"""---------------------------------------------- CHECKING FUNCTIONS ---------------------------------------"""
"""---------------------------------------------------------------------------------------------------------"""
#TODO DONE
def mergeNode(G, point, max_dist_merge, indmax):
    """
    This function checks whether a [point] can be merged
    with another node in graph [G]. 

    Parameters
    ----------
    G : Graph (networkx)
        the graph onto which the [point] should be merged.
    node : tuple (y, x)
        The x and y of the point that may be merged.
    max_dist_merge : number
        The maximum distance between two points which are considered as one node.
    indmax : list 
        Contains the nodes that were existing in the current specific region, and the nodes that 
        are in the whole graph G before processing this trajectory. 
    
    Returns
    -------
    n : string/number/None
        if mergeable: the osmid of the node with which [point] should be merged
        if not mergeable: None
       
    """
    # If there are no nodes in G, [node] can never be merged
    if len(G.nodes)==0: return None
     
    # Get the nodes that are within our specific region for this ID. [Nodes that still exist (within the original polygon)] + [nodes that are added by this trajectory]
    new_nodes = list(indmax[0] & G.nodes) + list(set(G.nodes) - set(indmax[1]))
    if len(new_nodes) == 0: return None
    
    # Find the nearest node of the corresponding subgraph
    n, dist = ox.get_nearest_node(G.subgraph(new_nodes), point, method='euclidean', return_dist=True)
    if dist < max_dist_merge: return n
    else: return None

#TODO DONE
def get_max_distance_bound(G, edge, settings):
    """
    This function checks whether edge is a new edge or not. The 
    function returns the maximum distance that should be used, based 
    on the settings assumed

    Parameters
    ----------
    G : MultiDigraph
        graph that includes [edge] (needed because of the "new" attribute)
    edge : tuple
        (u, v, key, geometry)
    settings : tuple
        Algorithm settings used

    Returns
    -------
    max_dist_projection : float
        Max dist projection based on the assumptions in settings. 
    
    """
    max_dist_projection, _, _, _, _, _ = ExtractSettings(settings)
    if G.edges[edge[0], edge[1], edge[2]]['new']: return max_dist_projection[1]
    else: return max_dist_projection[0]

#TODO DONE
def closeToCorner(point, edge, max_dist):
    """
    This function checks whether [point] is within [max_dist] from 
    corner point of [edge]
    
    Parameters
    ----------
    point : tuple 
        (x,y) 
    edge : tuple
        (u, v, key, geometry)     
    max_dist : float 
        closeness indication (e.g., 15m).

    Returns
    -------
    True/False 
        indication if [point] is close to a corner point in [edge]
    """
    # Determine the list of cornerpoints of [edge]
    cornerList = [y for y in edge[3].coords]
    
    for cornerPoint in cornerList:
        distance = sqrt((point[0]-cornerPoint[0])**2+(point[1]-cornerPoint[1])**2)
        if distance <= max_dist: return True
    return False

#TODO DONE
def checkIfNoCornerPointsFromStart(edge, point_right):
    """
    This function checks whether there are no corner points 
    on [edge] between the start of this edge and [point_right].

    Parameters
    ----------
    edge : (u, v, key, geometry)
    point_right : (x, y)

    Returns
    -------
    Boolean : True / False
        
    """
    # Extract the line pieces from the geometry
    edge_points_list = [y for y in edge[3].coords]
    edge_points_list = sorted(set(edge_points_list), key = edge_points_list.index) # Do not incorporate double points if existing
    
    # When an edge has zero length (going from and to a single node) (will not occur much), there is no corner point inbetween
    if len(edge_points_list) == 1: return True 
    
    # If the first point after the starting point (or left point) is equal to the right point, we know that there is no corner point inbetween. 
    if (edge_points_list[1] == point_right): return True
    
    return False
"""---------------------------------------------------------------------------------------------------------"""
"""---------------------------------------------------------------------------------------------------------"""    
