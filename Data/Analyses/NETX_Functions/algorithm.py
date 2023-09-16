import pandas as pd
import geopandas
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString, MultiPoint

from NETX_Functions.GraphOperations import mergeNode, findProjectionPoint, add_point_expl, include_point_in_newedge, projectPointOnEdge, get_SP_distance, remove_point, get_nearest_edge_FULL, get_max_distance_bound, closeToCorner, checkIfNoCornerPointsFromStart, ensure_point_in_network, add_datvels
from NETX_Functions.MathOperations import computeLengthLinestring

def ExtendGraph(initial_G, points, settings, MAX_STEP_INPUT = None, do_print = False):
    # If only a portion should be examined per trip
    if MAX_STEP_INPUT is None : MAX_STEP_INPUT = len(points)
    trips = points.ID.unique()
    
    # Create dataset with the start and end points of trips (which will be marked yellow) 
    trip_points = pd.DataFrame(columns = ['Trip ID', 'lat_start', 'lon_start', 'lat_end', 'lon_end', 'start_id', 'end_id'])
    trip_points["Trip ID"] = points.ID.unique()
    
    # Initialize parameter
    step_counter, edge_plot, additional_points, bearing_notes, crs_old = 0, None, [], [], 'start'
    
    for trip in trips:
        # Define tripdf and fraction examined (step)
        tripdf = points.loc[points.ID == trip,:].reset_index(drop = True)
        step = max(min(len(tripdf), MAX_STEP_INPUT-step_counter),0)
        if do_print: print("Number of points examined for this trip: " + str(step) + "/" + str(len(tripdf)))
        
        # Update the df with start/end points
        trip_points.loc[trip_points['Trip ID'] == trip, "lat_start"] = tripdf.loc[0,"Latitude"]
        trip_points.loc[trip_points['Trip ID'] == trip, "lon_start"] = tripdf.loc[0,"Longitude"]
        trip_points.loc[trip_points['Trip ID'] == trip, "start_id"] = str(initial_G.number_of_nodes()) # Needed when plotting a route
        
        # Project graph
        tripdf = geopandas.GeoDataFrame(tripdf, geometry = geopandas.points_from_xy(tripdf.Longitude, tripdf.Latitude), crs="EPSG:4326")
        tripdf = ox.project_gdf(tripdf, to_crs = initial_G.graph['crs'])
        
        # Print the current tripdf
        if do_print: print(tripdf)
        
        """------------------"""
        ExtendGraph_trajectory(initial_G, tripdf = tripdf, step = step, settings = settings, do_print = do_print)
        """------------------"""
        
        crs_old = tripdf.crs
        
        # Update the df with start/end points
        if (step == len(tripdf)):
            trip_points.loc[trip_points['Trip ID'] == trip, "lat_end"] = tripdf.loc[step-1,"Latitude"]
            trip_points.loc[trip_points['Trip ID'] == trip, "lon_end"] = tripdf.loc[step-1,"Longitude"]
            trip_points.loc[trip_points['Trip ID'] == trip, "end_id"] = str(initial_G.number_of_nodes()-1) # Needed when plotting a route
        
        # Termination criteria. Stop the procedure when having reached the max_step_input limit
        step_counter = step_counter + step
        if (step_counter + 1) > MAX_STEP_INPUT: break
    
    
def ExtendGraph_trajectory(G_proj, tripdf, step, settings, do_print = False):
    """
    A few key parameters: 
    
    temp_last_point : String
        Represents the name of the node at which a newly created edge started. Only defined when a new edge is created. Otherwise None.
    
    close_to_point_start : tuple 
        A new edge is created when a point could not be absorbed. This point is then connected to the network, close to the point that was previously absorbed. This point is stored, so that
        later on, when this edge might be reconnected, the new connection point could again be close to this point. If two_way = True, this is not very useful (makes no sense). Also, only 
        used when we could adjust new edges. 
        
    first_point_of_edge : Boolean
        Indicates whether the current point is the first point of a potential (!) new edge. As an example, if first_point_of_edge and we can not absorb the point into the edge, we start 
        with creating a new edge. Similarly, if we can absorb a point but it is not the first_point_of_edge, we have been creating a new edge. Therefore, we must finish this edge. 
    
    (projectedPoint_OLD, edge_old) : (tuple, tuple)
        This is information for the next iteration. If the current point is absorbed, the next point must be absorbed into an edge that is "reachable" from this current point. Therefore, 
        we store the current edge onto which it has been projected, along with its projected point. 
    
    started_new_edge : Boolean
        Indicates whether the current point that is absorbed was also the point that could not be absorbed, In this case, we project this point to two sides of the network. In the other 
        case, we project the previous point to the network. 
    
    temp_new/newNode_new : Boolean
        Indicates whether the temp_last_point (temp) or the last added node (newNode) is added explicitly, or if it is merged with an existing node.
    """
    
    # Define coordinate reference system
    crs = G_proj.graph['crs']

    # Initialize parameters
    first_point_of_edge, currentLine, projectedPoint, projectedPoint_OLD, edge_old, p, skipped, start_point, keep_start, started_new_edge  = True, [], None, None, None, 0, False, False, False, False
    temp_last_point = None
    temp_new = None
    trip = tripdf.ID[0]
    close_to_corner = 15
    
    while p < step:
        
        """"------------------------------------------------------------------------------------------"""
        if do_print: print("----- Currently examining point " + str(p) + " ------")
        y_point = tripdf.geometry[p].y
        x_point = tripdf.geometry[p].x
        curDateVel = str(tripdf.DateTime[p]) + "|" + str(tripdf.Speed[p])
        
        
        
        # Determine the maximum (actual) distance from the last GPS point (note that the previous point could have been merged, there subtract/add this distance (widen the interval))
        max_actual_dist = [tripdf.MaxDistance_TIME[p]+settings[2], (tripdf.MinDistance_LOC[p]-settings[2], tripdf.MaxDistance_VELTIME[p]+settings[2])]
        if do_print: print(max_actual_dist)

        if p == 0: # This means that it is a start point of a trip. Explicitly add this point as a node (or merge with an existing node) and start a new edge.
            """"------------------------------------- TRY TO ABSORB --------------------------------------"""
            if do_print: print("||||| Try to project the point.")
            if do_print: print("Close to the point: {0}, which is projected onto edge (forward): {1}".format(None, None))
            # Find the nearest edge (with the right direction). Note that if the point will not be merged, the direction does not make sense.
            projectedPoint, closestDistance, edge, be_edgenew = findProjectionPoint(G_proj, (y_point,x_point,tripdf.Course[p],curDateVel), close_to_edge = None, connecting = False, forward = None, temp_point = None, settings = settings, max_actual_dist = None, crs= crs)
            if do_print: print("Projected point: {0}".format(projectedPoint))
            if do_print: print("Closest distance: " + str(closestDistance) + " meters.")
            if do_print: print("Projected onto edge: {0}".format(edge))
            
            # If point cannot be projected (within the threshold distance) and when not incorporating the distance covered 
            merged_with_point = False
            if closestDistance is None: # Check if start node is already another point in the graph. Note projectedPoint_OLD is always None here. 
                node_to_be_merged = mergeNode(G_proj, (y_point, x_point), max_dist_merge = settings[2], indmax = settings[4])
                if node_to_be_merged is not None:
                    if do_print: print("Start node is merged with: " + str(node_to_be_merged))
                    projectedPoint = (G_proj.nodes[node_to_be_merged]['x'], G_proj.nodes[node_to_be_merged]['y'])
                    closestDistance = 0
                    edge = (node_to_be_merged, node_to_be_merged, 0, LineString([(G_proj.nodes[node_to_be_merged]['x'], G_proj.nodes[node_to_be_merged]['y']),(G_proj.nodes[node_to_be_merged]['x'], G_proj.nodes[node_to_be_merged]['y'])]))
                    if do_print: print("Projected point: {0}".format(projectedPoint))
                    if do_print: print("Projected onto edge: {0}".format(edge)) 
                    merged_with_point = True
            """"------------------------------------------------------------------------------------------"""
            """"------------------------------------------------------------------------------------------"""
            if closestDistance is not None: # Start point is absorbed
                if do_print: print("||||| Absorb point into the network.")
                
                # Check if point is absorbed in a new edge (added by the algorithm). Note that when mergin networks, we do not adjust edge geometries (settings[5] must be False). Also, when a point is merged with an existing node, we also do not adjust an edge (makes no sense, because this point will already be in this edge)
                if (not merged_with_point) & (not settings[5]):
                    if G_proj.edges[edge[0], edge[1], edge[2]]['new']:  
                        # If so, we adjust the geometry of this new edge (if not merged onto a corner point)
                        if do_print: print("Absorption into newly created edge. Adjust geometry of this edge.")
    
                        """ ________________________________ ADJUST THE OPPOSITE EDGE (WHEN ABSORBING) _______________________________"""
                        # Find the opposite edge (if there is one)
                        ce = get_nearest_edge_FULL(G_proj, Point(projectedPoint), settings[4], return_geom=True, return_dist=True)
                        ce = [a for a in ce if a[1] < settings[0][1]]
                        if len(ce) > 1: # if there is only 1, we do not have an opposite edge.
                            # when the closest edge is the edge itself, we take the second closest edge as our opposite edge.
                            closestEdge = tuple(list(ce[0][0]) + [ce[0][1]])
                            if closestEdge[0:3] == edge[0:3]: closestEdge = tuple(list(ce[1][0]) + [ce[1][1]])
                            
                            COND_1 = G_proj.edges[closestEdge[0],closestEdge[1],closestEdge[2]]['new']
                            #COND_2 = (not closeToCorner(G_proj, (x_point, y_point), closestEdge, max_dist = 10))
                            # Find the line piece onto which the point is projected. This line piece will be adjusted
                            _, projectedPoint_r, best_edge = projectPointOnEdge(closestEdge, (Point(x_point,y_point), -1), alpha_bar = None, max_dist = float('inf'))
                            COND_3 = (not closeToCorner(projectedPoint_r, closestEdge, max_dist = close_to_corner))
                            
                            # If closestEdge is a newly created edge AND if the point is not close to an interior point of the edge geometry: adjust it!
                            if COND_1 & COND_3:
                                if do_print: print("Opposite edge {0} also adjusted!".format(closestEdge))

                                # Adjust closestEdge. Note that there is no old_point, so we cannot use this information   
                                include_point_in_newedge(G_proj, closestEdge, best_edge, (y_point, x_point, -1), old_point = (None, None), points_to_be_adjusted = [], settings = settings, crs = crs, max_actual_dist = None, temp_new = None, do_print = do_print)  
                                    
                                # By including this point in the opposite edge, we might need to adjust our edge_old and projectionPoint_OLD. However, since both are None, we just reproject the point to find the updated projection parameters. Note that changing an opposite edge might change node names, geometries etc. We reproject the point, but we have the additional restriction that we would like to obtain a similar closestDistance as before (not projecting on a different edge as before) .
                                """ _____________________________________ UPDATE PROJECTION PARAMETERS _______________________________________"""
                                # Update the projection parameters
                                projectedPoint, _, edge, be_edgenew = findProjectionPoint(G_proj, (y_point, x_point, -1), close_to_edge = None, connecting = False, forward = True, temp_point = None, settings = settings, max_actual_dist = None, crs= crs, target_dist = closestDistance)                   
                                # If something is wrong, throw an error
                                if projectedPoint is None:
                                    print("OEI, FOUND THE WRONG EDGE_OLD AND PROJECTEDPOINT_OLD. Maybe we need to adjust the original location instead of reprojecting. Should not happen because this is the opposite edge, but in extreme cases...,")
                                    stop
                                """ __________________________________________________________________________________________________________"""
                        """ _________________________________________________________________________________________________________"""
                
                        """ ________________________________ ADJUST THE ABSORPTION EDGE (WHEN ABSORBING) _____________________________"""
                        # Now, secondly, adjust the edge onto which the start point was originally projected and update the projection parameters
                        if (not closeToCorner(projectedPoint, edge, max_dist = close_to_corner)):# & (not closeToCorner(G_proj, (x_point, y_point), edge, max_dist = 10)):   
                            #TODO OPPOSITE IS NOT NEEDED HERE               
                            edge, _, _, _ = include_point_in_newedge(G_proj, edge, be_edgenew, (y_point, x_point,-1), old_point = (None, None), points_to_be_adjusted = [], settings = settings, crs = crs, max_actual_dist = None, temp_new = None, do_print = do_print)
                            projectedPoint = (x_point, y_point)
                        """ __________________________________________________________________________________________________________"""         
                        
                # Update close to parameters for the next round
                edge_old = edge
                projectedPoint_OLD = projectedPoint
            else: # Start point could not be absorbed
                if do_print: print("Start node added explicitly.")
                # First check if the point can be merged in an existing edge. Therefore, we first find the closest edges to this start node
                ce = get_nearest_edge_FULL(G_proj, Point(x_point, y_point), settings[4], return_geom=True, return_dist=True) # Returns ordered list 
                ce = [a for a in ce if a[1] < settings[0][1]] # Remove non-options beforehand
                # If edges are left, determine projected points, etc.
                if (len(ce)>0): 
                    # Check (and find) closestEdge that satisfies the maximum projection distance for that edge (only useful when bar{d} and barbar{d} are different (then we pick the top one). Note that an edge might be further away, but this may be new edge for which it does satisfy the maximum projection distance. 
                    ind = 0
                    closestEdge = tuple(list(ce[ind][0]) + [ce[ind][1]])
                    while (closestEdge[4] > get_max_distance_bound(G_proj, closestEdge, settings)) & (ind < len(ce)-1):
                        ind += 1
                        closestEdge = tuple(list(ce[ind][0]) + [ce[ind][1]])
                        
                    # Still check whether we did not run out of options in the while loop above
                    if (closestEdge[4] < get_max_distance_bound(G_proj, closestEdge, settings)): # Start point added to an existing edge
                        # Find projected point (we already have the distance)
                        if do_print: print("Start point projected onto existing edge: {0}".format((closestEdge[0],closestEdge[1],closestEdge[2])))
                        _, projectedPoint, be = projectPointOnEdge(closestEdge, (Point(x_point, y_point), -1), alpha_bar = None, max_dist = float('inf'))
                        if do_print: print("Projected point: {0}".format(projectedPoint))
                        if do_print: print("Closest distance: " + str(closestEdge[4]) + " meters.")
                        
                        # Add this point to the network (either explicitly or by merging). Note that we add it in two ways, because we do not have info about the direction of this point. newNode will be the point that is finally added (note that [name] could have been merged with an existing node)
                        """ ___________________________ ADD (MAKE SURE) THE CONNECTION POINT TO THE NETWORK _________________________"""
                        newNode, _, _, _, _, _= ensure_point_in_network(G_proj, projectedPoint, closestEdge, be, name = 'Start node: ' + str(trip) + "_" + str(p), settings = settings, two_way = True, temp_last_point = None, start_point = None, do_print = do_print)
                        """ _________________________________________________________________________________________________________"""
                        
                        # The next point must be close to this point. Since we are using edges, we can set the close_to_edge equal to a nonexisting self-edge
                        projectedPoint_OLD = (G_proj.nodes[newNode]['x'], G_proj.nodes[newNode]['y'])
                        edge_old = (newNode, newNode, 0, LineString([projectedPoint_OLD, projectedPoint_OLD]))
                        
                        # Go to the next point
                        p += 1
                        if do_print: print()
                        if do_print: print()
                        continue  
                          
                # If not, or no feasible point were found in the inner if statement, add this start node explicitly to the network (not connected to the current network) and start a new edge.
                newNode = 'Start node: ' + str(trip) + "_" + str(p)
                G_proj.add_node(newNode, y = y_point, x = x_point, osmid = newNode, geometry = Point(x_point, y_point))
                # Start a new edge that connects this point to the current network
                if do_print: print("Start a new edge!")
                currentLine = currentLine + [(x_point, y_point)]
                curDateVels = "|"+curDateVel      
                # Update parameters
                temp_last_point = newNode # Starting node of the newly created edge
                first_point_of_edge = False # The next point is not the first point of an edge
                close_to_point_start = "Start" # There is no point to which the starting point is close (first point of trace)
                started_new_edge = False # If we would run this round again, trying to project the starting point, we would not be able to (as we did not use any restriction on the distance covered before) In other words, the current point will never be absorbed in future rounds. 
                temp_new = True  # Start node (temp_last_point) is added explicitly
            
            # Go to the next point    
            p += 1
            if do_print: print()
            if do_print: print()
            continue
            """"------------------------------------------------------------------------------------------"""
            
        """"--------------------------------------- TRY TO ABSORB ------------------------------------""" 
        # First, check if the current point can be absorbed
        if do_print: print("||||| Try to project the point.")
        if do_print: print("Close to the point: {0}, which is projected onto edge (forward): {1}".format(projectedPoint_OLD, edge_old))
        if edge_old is not None: # Throw error when the close_to_edge does not exist (anymore)
            if ((edge_old[0], edge_old[1], edge_old[2]) not in G_proj.edges) and (edge_old[0] != edge_old[1]):
                print("YOU FOOL!! Edge: {0} does not exist (anymore)!".format(edge_old))
                stop
        # Find the nearest edge (with the right direction). Note that if the point will not be merged, the direction does not make sense. 
        projectedPoint, closestDistance, edge, be_edgenew = findProjectionPoint(G_proj, (y_point,x_point,tripdf.Course[p],curDateVel), close_to_edge = edge_old, connecting = False, forward = True, temp_point = projectedPoint_OLD, settings = settings, max_actual_dist = max_actual_dist, crs= crs)
        if do_print: print("Projected point: {0}".format(projectedPoint))
        if do_print: print("Closest distance: " + str(closestDistance) + " meters.")
        if do_print: print("Projected onto edge: {0}".format(edge))
            
        # If the current point could not be absorbed, check whether turning (at the previous node) may resolve the issue (if a previous point exists). 
        if (closestDistance is None) & (projectedPoint_OLD is not None):
            # Check if there are potential opposite edges (within 1m). 
            ce = get_nearest_edge_FULL(G_proj, Point(projectedPoint_OLD), settings[4], return_geom=True, return_dist=True)
            ce = [a for a in ce if a[1] < 1]
            if (len(ce) > 1): # If there is at least one additional edge close (besided the edge itself), we have an opposite edge and we can turn. 
                # when the closest edge is the edge itself, we take the second closest edge as our opposite edge.
                closestEdge = tuple(list(ce[0][0]) + [ce[0][1]])
                if (closestEdge[0:3] == edge_old[0:3]): closestEdge = tuple(list(ce[1][0]) + [ce[1][1]])
            
                # Find the projectedPoint of the previous point on the opposite edge
                _, projectedPoint_OLD_opposite, be_old_opposite = projectPointOnEdge(closestEdge, (Point(projectedPoint_OLD), -1), alpha_bar = None)
                
                if do_print: print("||||| Try to project the point after turning at the previous point.")
                if do_print: print("Close to the point: {0}, which is projected onto edge (forward): {1}".format(projectedPoint_OLD, closestEdge))
                # Find the nearest edge (with the right direction). Note that if the point will not be merged, the direction does not make sense. 
                projectedPoint_2, closestDistance_2, edge_2, be_2 = findProjectionPoint(G_proj, (y_point,x_point,tripdf.Course[p],curDateVel), close_to_edge = closestEdge, connecting = False, forward = True, temp_point = projectedPoint_OLD_opposite, settings = settings, max_actual_dist = max_actual_dist, crs= crs)
                if do_print: print("Projected point: {0}".format(projectedPoint_2))
                if do_print: print("Closest distance: " + str(closestDistance_2) + " meters.")
                if do_print: print("Projected onto edge: {0}".format(edge_2))
            
                # If the point can now be projected, we add this turning point. Note that due to this addition, the temp_last_point may be adjusted (when the point that we added merged with this temp_last_point). Moreover, we keep the edges that were formed due to the ensuring the point. Either the two splitted egdes or a self edge from a merged node. 
                if closestDistance_2 is not None:
                    _, _, new_edges, temp_last_point, _, _ = ensure_point_in_network(G_proj, projectedPoint = projectedPoint_OLD, edge = closestEdge, be = be_old_opposite, name = "Turning Point " + str(trip) + "_" + str(p), settings = settings, two_way = True, temp_last_point = temp_last_point, start_point = None, do_print = do_print)
                    # First define a new edge_old, it is one of the edges that was formed when adding the point. It does not matter which,  projectedPoint_OLD is either at the start or at the end of the splitted edge. 
                    edge_old = new_edges[0]
                    # Find the new projection information
                    projectedPoint, closestDistance, edge, be_edgenew = findProjectionPoint(G_proj, (y_point,x_point,tripdf.Course[p]), close_to_edge = edge_old, connecting = False, forward = True, temp_point = projectedPoint_OLD, settings = settings, max_actual_dist = max_actual_dist, crs = crs)
            
        # If point still cannot be projected (within the threshold distance) and when creating a new edge, check if point is merged with an existing node. 
        merged_with_point = False
        if (closestDistance is None) & (projectedPoint_OLD is None): # Only when creating a new edge. When not creating a new edge, we cannot just merge this point
            node_to_be_merged = mergeNode(G_proj, (y_point, x_point), max_dist_merge = settings[2], indmax = settings[4])
            if (node_to_be_merged is not None): 
                if do_print: print("Node is merged with: " + str(node_to_be_merged))
                projectedPoint = (G_proj.nodes[node_to_be_merged]['x'], G_proj.nodes[node_to_be_merged]['y'])
                closestDistance = 0
                edge = (node_to_be_merged, node_to_be_merged, 0, LineString([(G_proj.nodes[node_to_be_merged]['x'], G_proj.nodes[node_to_be_merged]['y']),(G_proj.nodes[node_to_be_merged]['x'], G_proj.nodes[node_to_be_merged]['y'])]))
                if do_print: print("Projected point: {0}".format(projectedPoint))
                if do_print: print("Projected onto edge: {0}".format(edge)) 
                merged_with_point = True
        """"------------------------------------------------------------------------------------------"""
            
        """"------------------------------------------------------------------------------------------"""
        if (closestDistance is not None): # interior points can only be merged when distance to one of the edges is close enough..
            if do_print: print("||||| Absorb point into the network.")
            
            # Check if point is absorbed in a new edge (added by the algorithm). Note that when merging networks, we do not adjust edge geometries (settings[5] must be False). Also, when a point is merged with an existing node, we also do not adjust an edge (makes no sense, because this point will already be in this edge)
            if (not merged_with_point) & (not settings[5]):
                newEdge = G_proj[edge[0]][edge[1]][edge[2]]
                if newEdge['new']: 
                    # If so, we adjust the geometry of this new edge (if not merged onto a corner point)
                    if do_print: print("Absorption into newly created edge. Adjust geometry of this edge.")
                    
                    """ ________________________________ ADJUST THE OPPOSITE EDGE (WHEN ABSORBING) _______________________________"""
                    # Find the opposite edge (if there is one)
                    ce = get_nearest_edge_FULL(G_proj, Point(projectedPoint), settings[4], return_geom=True, return_dist=True)
                    ce = [a for a in ce if a[1] < settings[0][1]]
                    if len(ce) > 1: # if there is only 1, we do not have an opposite edge.
                        # when the closest edge is the edge itself, we take the second closest edge as our opposite edge.
                        closestEdge = tuple(list(ce[0][0]) + [ce[0][1]])
                        if closestEdge[0:3] == edge[0:3]: closestEdge = tuple(list(ce[1][0]) + [ce[1][1]])
                        
                        COND_1 = G_proj.edges[closestEdge[0],closestEdge[1],closestEdge[2]]['new']
                        #COND_2 = (not closeToCorner(G_proj, (x_point, y_point), closestEdge, max_dist = 10))
                        # Find the line piece onto which the point is projected. This line piece will be adjusted
                        _, projectedPoint_r, best_edge = projectPointOnEdge(closestEdge, (Point(x_point,y_point), -1), alpha_bar = None, max_dist = float('inf'))
                        COND_3 = (not closeToCorner(projectedPoint_r, closestEdge, max_dist = close_to_corner))
                        
                        # If closestEdge is a newly created edge AND if the point is not close to an interior point of the edge geometry: adjust it!
                        if  COND_1 & COND_3:
                            if do_print: print("Opposite edge {0} also adjusted!".format(closestEdge))
                              
                            # If the edge_old is an artificial edge, we want to keep this edge for updating the edge_old after the inclusion process. As the nodes may change, we keep the information whether the edge was artificial and we feed its node to the inclusion method (for returning an updated node).
                            artificial_edge_old, artificial_edge_point = False, None
                            if edge_old is not None: 
                                if edge_old[0:3] not in G_proj.edges: # We are dealing with an artificial edge. 
                                    artificial_edge_old = True
                                    artificial_edge_point = edge_old[0] #or edge_old[0], both are the same
                                    
                            # Adjust closestEdge. Note that there is no old_point, so we cannot use this information. Note that when creating a new edge and, due to this inclusion, and if a new connection point is merged with our starting point of our newly created edge, we assume that this point is no longer new (as it now has two different functions: connection point AND starting point). Moreover, due to the adjustment procedure, temp_last_point and artificial_edge_point may have changed. Therefore, we also want to receive updates about the changes in these two nodes (names). Finally, because adjusting the edge might lead to connection points being changed it might happen that our edge_old does not exist anymore after this process. This is only the case if it has been split in the process. Therefore, we also ask for the split edges [edge_old_NEW], as these are the potential new edge_olds. 
                            edge_TEMP, points_adjusted, edge_old_NEW, temp_new = include_point_in_newedge(G_proj, closestEdge, best_edge, (y_point, x_point, -1), old_point = (None, None), points_to_be_adjusted = [temp_last_point, artificial_edge_point],  settings = settings, crs = crs, max_actual_dist = max_actual_dist, temp_new = temp_new, do_print = do_print)  
   
                            # Update points that may have been adjusted due to the inclusion processs.
                            temp_last_point = points_adjusted[0]
                            artificial_edge_point = points_adjusted[1]      
                                                               
                            # Including a point in a new edge, we compromise the projection parameters found before. To obtain the right parameters again, we reproject the point on the graph. However, to do this, we first need to adjust the edge_old and projectedPoint_OLD, as these may also been compromised. (edge_old may not exist anymore, or geometries could have been changed)

                            """ ___________________________________ UPDATE edge_old / projectedPoint_OLD _________________________________"""
                            # Update edge_old and projectedPoint_OLD
                            if edge_old is not None: # If both are None, we do not have to change anything
                                # If the edge_old was an articifial edge (node - node), we keep this artificial edge. However, the nodes may have changed, but this is incorporated above.
                                if artificial_edge_old:
                                    projectedPoint_OLD = (G_proj.nodes[artificial_edge_point]['x'], G_proj.nodes[artificial_edge_point]['y'])
                                    edge_old = (artificial_edge_point, artificial_edge_point, 0, LineString([projectedPoint_OLD, projectedPoint_OLD]))
                                # If edge_old is the opposite edge that is just adjusted, we know that the new edge_old is edge_TEMP. We can find the projectionPoint_OLD by projecting the old projectionpoint_OLD on this edge, and then we can recompute the projection parameters. 
                                elif edge_old[0:3] == closestEdge[0:3]:
                                    edge_old = edge_TEMP
                                    _, projectedPoint_OLD, _ = projectPointOnEdge(edge_old, (Point(projectedPoint_OLD), -1), alpha_bar = None, max_dist = float('inf'))
                                # If edge_old is not the opposite edge, but if this edge still exists in the network, we only need to adjust the geometry (may have been changed) and then we can use this adjusted edge_old to recompute the projectedPoint_OLD, after which we can recompute the projection parameters. 
                                elif edge_old[0:3] in G_proj.edges: 
                                    # ProjectedPoint_OLD may not be on edge_old anymore, due to changes in the geometry of edge_old (most likely no changes occurred)
                                    edge_old = (edge_old[0],edge_old[1],edge_old[2],G_proj.edges[edge_old[0],edge_old[1],edge_old[2]]['geometry'])
                                    _, projectedPoint_OLD, _ = projectPointOnEdge(edge_old, (Point(projectedPoint_OLD), -1), alpha_bar = None, max_dist = float('inf')) 
                                # If edge_old does not exist anymore, we know that it is splitted by adding a new connection point. We know that edge_old is therefore one of the two sides of this splitted edge. We check which has the closest projection distance and use this edge (part of the splitted edge) as our new edge_old. 
                                else: 
                                    if edge_old_NEW is None: 
                                            print("CANNOT HAPPEN! THEN, ANY OF THE ABOBE CASES SHOULD BE RELEVANT!!") 
                                            stop
                                    # find splitted edges
                                    d_min = float('inf')
                                    for e in edge_old_NEW:
                                        d1, projectedPoint_OLD_1, _ = projectPointOnEdge(e, (Point(projectedPoint_OLD), -1), alpha_bar = None, max_dist = float('inf'))
                                        if d1 < d_min: 
                                            projectedPoint_OLD, edge_old = projectedPoint_OLD_1, e
                                            d_min = d1
                                    
                                # If still something is wrong, throw an error
                                if projectedPoint_OLD is None:
                                    print("YOU FOOL!!! THE EDGE_OLD DOES NOT EXIST ANYMORE, ADJUST IT PROPERLY!")
                                    stop            
                            """ __________________________________________________________________________________________________________"""
 
                            """ _____________________________________ UPDATE PROJECTION PARAMETERS _______________________________________"""
                            # Update the projection parameters. We reproject the point, but we have the additional restriction that we would like to obtain a similar closestDistance as before (not projecting on a different edge as before).
                            projectedPoint, _, edge, be_edgenew = findProjectionPoint(G_proj, (y_point, x_point, -1), close_to_edge = edge_old, connecting = False, forward = True, temp_point = projectedPoint_OLD, settings = settings, max_actual_dist = max_actual_dist, crs= crs, target_dist = closestDistance)        
                            # If still something is wrong, throw an error
                            if projectedPoint is None:
                                print("OEI, FOUND THE WRONG EDGE_OLD AND PROJECTEDPOINT_OLD. Maybe we need to adjust the original location instead of reprojecting. Should not happen because this is the opposite edge, but in extreme cases...,")
                                stop
                            """ __________________________________________________________________________________________________________"""
                    """ __________________________________________________________________________________________________________"""          

                    """ ________________________________ ADJUST THE ABSORPTION EDGE (WHEN ABSORBING) _____________________________"""
                    # Now, secondly, adjust the edge onto which the current point was originally projected and update the projection parameters
                    if (not closeToCorner(projectedPoint, edge, max_dist = close_to_corner)):# & (not closeToCorner(G_proj, (x_point, y_point), edge, max_dist = 10)):       
                        # Adjust closestEdge. Note that there is an old_point, so we can use this information when we are adding a point to the first line piece of an edge. In this case, we can update the close_to_point_start information with the information of projectedPoint_OLD. As before, when creating a new edge and, due to this inclusion, and if a new connection point is merged with our starting point of our newly created edge, we assume that this point is no longer new (as it now has two different functions: connection point AND starting point). Moreover, due to the adjustment procedure, temp_last_point may have changed. Therefore, we also want to receive updates about the changes in this node (name). 
                        edge, points_adjusted, _, temp_new = include_point_in_newedge(G_proj, edge, be_edgenew, (y_point, x_point, -1), old_point = (projectedPoint_OLD, edge_old), points_to_be_adjusted = [temp_last_point], settings = settings, crs = crs, max_actual_dist = max_actual_dist, temp_new = temp_new, do_print = do_print)
                        temp_last_point = points_adjusted[0]
                        projectedPoint = (x_point, y_point)
                    """ __________________________________________________________________________________________________________"""          
             
            # Update close to parameters for the next round
            edge_old = edge
            projectedPoint_OLD = projectedPoint 
              
            # If it is not the first point of an edge, we must finish a newly created edge. 
            if not first_point_of_edge:
                if do_print: print("Point connects the new edge with the existing network!")
              
                # Project the previous point onto the graph (this is the connection point)
                if do_print: print("--------------------------")
                if started_new_edge: # Point that could not be absorbed is also the point that connects the new edge to the network
                    # Reproject the last point without incorporating the course, but it must be close to the edge onto which the current point was added. Note that the most likely distance covered is not accurate anymore, as this was the MLDC from the last point. We are now looking at the MLDC from the starting point of the edge (last point was absorbed). Therefore, we use as MLDC (0,0). This means that, we want to project such that we minimize the distance to the temp_last_point. 
                    if do_print: print("Reproject the current point onto the graph.")
                    if do_print: print("Close to the point: {0}, which is projected onto edge (backwards): {1}".format(projectedPoint, edge))
                    projectedPoint, closestDistance, edge, be = findProjectionPoint(G_proj, (G_proj.nodes[temp_last_point]['y'], G_proj.nodes[temp_last_point]['x'], -1), close_to_edge = edge, connecting = True, forward = False, temp_point = projectedPoint, settings = settings, max_actual_dist = [max_actual_dist[0], (0,0)], crs = crs) 
                else: 
                    # Reproject the previous point without incorporating the course, but it must be close to the edge onto which the current point was added
                    if do_print: print("Reproject the previous point onto the graph.")
                    if do_print: print("Close to the point: {0}, which is projected onto edge (backwards): {1}".format(projectedPoint, edge))
                    projectedPoint, closestDistance, edge, be = findProjectionPoint(G_proj, (tripdf.geometry[p-1].y, tripdf.geometry[p-1].x, -1), close_to_edge = edge, connecting = True, forward = False, temp_point = projectedPoint, settings = settings, max_actual_dist = max_actual_dist, crs = crs)
                if do_print: print("Projected point: {0}".format(projectedPoint))
                if do_print: print("Closest distance: " + str(closestDistance) + " meters.")
                if do_print: print("Projected onto edge: {0}".format(edge))
                
                # Store information whether the point is absorbed in a newly created edge. Note that this edge may be splitted in the process of ensuring. Therefore, we store it as this ealry moment. Note that the edge may not exist (artificial). This edge has no geometry, so the point is not absorbed in a new edge. 
                try: absorbedInNewEdge = G_proj[edge[0]][edge[1]][edge[2]]['new']
                except: absorbedInNewEdge = False
                
                # We store the start point of the edge to check condidion (2) for adjusting. Note that this may be adjusted when ensuring the connection point in the network. 
                start_point = edge[0]
                # Ensure that the connection point is in the network. Note that we also want to know whether this connection point is added explicitly (newNode_new = True) or (newNode_new = False). Moreover, the temp_last_point may have changed, the edge itself and the node name (of newNode) itself. We return the updated values. Moreover, we want to when the edge is split, the splitted parts. When checking whether we want to adjust an existing (newly created) edge, we check only the left hand side of the splitted edge (this is the part that is possibly adjusted). Note that the start_point of the edge may be adjusted. We need the updated value to check condidion (2) for adjusting. 
                """ ___________________________ ADD (MAKE SURE) THE CONNECTION POINT TO THE NETWORK _________________________"""
                newNode, newNode_new, splitted_edges, temp_last_point, start_point, edge = ensure_point_in_network(G_proj, projectedPoint, edge, be, name = 'Node_p: '+ str(trip) + "_" + str(p-1), settings= settings, two_way = settings[3], temp_last_point = temp_last_point, start_point = start_point, do_print = do_print)
                """ _________________________________________________________________________________________________________"""
              
                # Add last piece of the edge
                if do_print: print("Possible new edge from: " + str(temp_last_point) + " to " + str(newNode))
                # Note that 
                #if (temp_last_point == newNode) & started_new_edge: 
                #    print(stop)
                #    currentLine = [(G_proj.nodes[newNode]['x'], G_proj.nodes[newNode]['y'])]
                # Add the point of the end node to the current geometry
                currentLine += [(G_proj.nodes[newNode]['x'], G_proj.nodes[newNode]['y'])]
                       
                # Check the length of the new edge (from temp_last_point to newNode (the previously projected point))
                length_new_edge = computeLengthLinestring(LineString(currentLine), method = 'euclidean')
                try: length_existing = nx.shortest_path_length(G_proj, source = temp_last_point, target = newNode, weight='length')
                except: length_existing = float('inf')
                if do_print: print("Length of the possible new edge: " + str(length_new_edge))
                if do_print: print("Length of the path between "+ str(temp_last_point) + " and " + str(newNode) + " (without the edge added): " + str(length_existing))
                length_deviation = abs(length_existing - length_new_edge)
                              
                """ __________________________________ MERGING NODES MAY REDUCE THE LENGTH __________________________________"""
                # It may be the case that we can also travel via an opposite edge to the new node. In other words, a shortest path already existed, but we were not able to traverse this, due to the projection point(s) of the start and end point of the new edge (projected onto different edges). Therefore, we check if an SP exists with a smaller deviation, when using opposite edge(s). Note that we only do this when we are not considering the two-way situations. If two-ways is activated, does not make sense (useless work), because each point is already added in two ways, and such situations will not occur. 
                if not settings[3]: 
                    # Check if temp_last_point could be merged. 
                    temp_last_point_mergeable = False
                    ce = get_nearest_edge_FULL(G_proj, Point((G_proj.nodes[temp_last_point]['x'], G_proj.nodes[temp_last_point]['y'])), settings[4], return_geom=True, return_dist=True)
                    ce = [a for a in ce if a[1] < 1]
                    # If at least 2 edges are left, determine projected points, etc. (note that 1 of the two is the edge itself)
                    if (len(ce) > 1): 
                        # Check (and find) closestEdge that satisfies the maximum projection distance for that edge (only useful when bar{d} and barbar{d} are different (then we pick the top one). Note that an edge might be further away, but this may be new edge for which it does satisfy the maximum projection distance. 
                        ind = 0
                        closestEdge_temp = tuple(list(ce[ind][0]) + [ce[ind][1]])
                        # If temp_last_point is part of one of the end points of the opposite edge, we are not getting a reduced SP. We are looking for an edge into which the temp_last_point may be merged. 
                        while (temp_last_point in [closestEdge_temp[0], closestEdge_temp[1]]) & (ind < len(ce)-1):
                            ind += 1
                            closestEdge_temp = tuple(list(ce[ind][0]) + [ce[ind][1]])
            
                        # Still check whether we did not run out of options in the while loop above
                        if (temp_last_point not in [closestEdge_temp[0], closestEdge_temp[1]]):
                            temp_last_point_mergeable = True
            
                    # Check if newNode could be merged
                    newNode_mergeable = False
                    ce = get_nearest_edge_FULL(G_proj, Point((G_proj.nodes[newNode]['x'], G_proj.nodes[newNode]['y'])), settings[4], return_geom=True, return_dist=True)
                    ce = [a for a in ce if a[1] < 1]
                    if (len(ce) > 1): 
                        # Check (and find) closestEdge that satisfies the maximum projection distance for that edge (only useful when bar{d} and barbar{d} are different (then we pick the top one). Note that an edge might be further away, but this may be new edge for which it does satisfy the maximum projection distance. 
                        ind = 0
                        closestEdge_newNode = tuple(list(ce[ind][0]) + [ce[ind][1]])
                        # If newNode is part of one of the end points of the opposite edge, we are not getting a reduced SP. We are looking for an edge into which the newNode may be merged. 
                        while (newNode in [closestEdge_newNode[0], closestEdge_newNode[1]]) & (ind < len(ce)-1):
                            ind += 1
                            closestEdge_newNode = tuple(list(ce[ind][0]) + [ce[ind][1]])
                        
                        # Still check whether we did not run out of options in the while loop above
                        if (newNode not in [closestEdge_newNode[0], closestEdge_newNode[1]]):
                            newNode_mergeable = True
            
                    # Distinguish three different situations. In each scenario, we use the actual points corresponding to temp_last_point (from) and newNode (to). Therefore, we first define the start and end point of the edge (using x and y coordinates)
                    temp_last_point_POINT = (G_proj.nodes[temp_last_point]['x'], G_proj.nodes[temp_last_point]['y'])
                    newNode_POINT = (G_proj.nodes[newNode]['x'], G_proj.nodes[newNode]['y'])
                    
                    # 1) Merge temp_last_point, NOT merge newNode
                    length_existing_new_11 = float('inf')
                    if temp_last_point_mergeable:
                        from_point = (temp_last_point_POINT, closestEdge_temp)
                        to_point = (newNode_POINT, (newNode, newNode)) # Note that the self edge is not used in the SP_distance (newNode is a node in the graph)
                        length_existing_new_1, _, _, _ = get_SP_distance(G_proj, from_point = from_point, to_point = to_point)
                        length_existing_new_11 = abs(length_existing_new_1 - length_new_edge)
                    # 2) Merge newNode, NOT merge temp_last_point
                    length_existing_new_22 = float('inf')
                    if newNode_mergeable: 
                        from_point = (temp_last_point_POINT, (temp_last_point, temp_last_point)) # Note that the self edge is not used in the SP_distance (tmep_last_point is a node in the graph)
                        to_point = (newNode_POINT, closestEdge_newNode)
                        length_existing_new_2, _, _, _ = get_SP_distance(G_proj, from_point = from_point , to_point = to_point)
                        length_existing_new_22 = abs(length_existing_new_2 - length_new_edge)
                    # 3) Merge temp_last_point and merge newNode
                    length_existing_new_33 = float('inf')
                    if temp_last_point_mergeable & newNode_mergeable:
                        from_point = (temp_last_point_POINT, closestEdge_temp)
                        to_point = (newNode_POINT, closestEdge_newNode)
                        length_existing_new_3, _, _, _ = get_SP_distance(G_proj, from_point = from_point , to_point = to_point)
                        length_existing_new_33 = abs(length_existing_new_3 - length_new_edge)
            
                    # Create list of all deviations from the real and determine what is the best (smallest) deviation
                    length_list_dev = [length_existing_new_11, length_existing_new_22, length_existing_new_33]
                    best_deviation = min(length_list_dev)
                    minind = length_list_dev.index(best_deviation)
            
                    # If there is better deviation, then apply the corresponding merging procedures.
                    if best_deviation < length_deviation:
                        length_deviation = best_deviation
                        # Merge the right points in the right edges
                        if minind in [1,3]: # Merge temp_last_point
                            _,_, be_temp = projectPointOnEdge(closestEdge_temp, (Point(temp_last_point_POINT), -1), alpha_bar = None, max_dist = float('inf'))
                            # Add the point. Note that we also want to return the splitted edges. We need these two when, the newNode will also be merged in this same edge. In this case, the edge is changed to the two splitted edges.
                            _, split_edges = add_point_expl(G_proj, point = temp_last_point_POINT, edge = closestEdge_temp, be = be_temp, node_name = temp_last_point, settings = settings, merge = True, two_way = False, do_print = do_print)
                        if minind in [2,3]: # Merge newNode 
                            # In the case that the temp_last_point was already merged in an edge. If both points were supposed to be merged in the same edge, we now need to adjust this edge, as temp_last_point was now merged in the edge (and therefore this edge is now splitted)
                            if (minind == 3) & (closestEdge_temp[0:3] == closestEdge_newNode[0:3]): # Already merged temp_last_point AND we must adjust the edge. 
                                closestDistance, _, _ = projectPointOnEdge(split_edges[0], (Point(newNode_POINT), -1), alpha_bar = None, max_dist = float('inf'))
                                # We check if the distance is smaller than 1. Note that this was true before, so it has to be true now. If the projection distance is larger than 1, it means that we have to project on the other split of the egde. 
                                if closestDistance < 1: closestEdge_newNode = split_edges[0] 
                                else: closestEdge_newNode = split_edges[1]
                            # Merge the newNode
                            _, _, be_temp = projectPointOnEdge(closestEdge_newNode, (Point(newNode_POINT), -1), alpha_bar = None, max_dist = float('inf'))
                            add_point_expl(G_proj, point = newNode_POINT, edge = closestEdge_newNode, be = be_temp, node_name = newNode, settings = settings, merge = True, two_way = False, do_print = do_print)    
                """ _________________________________________________________________________________________________________"""
                
                """ __________________________________ CONSIDER ADJUSTING/ADDING THE EDGE ___________________________________"""
                # Adjust an existing newly created edge INSTEAD of adding the new edge when:
                # 1) the last point was absorbed in a new edge (We only adjust newly created edges)
                # 2) the start point of the edge to be adjusted must be the temp_last_point (the start point of the newly created edge). In other words, we are only adjusting edge if the newly created edge starts and ends on this same edge. If there is a switch between edges, we do not adjust either edge. 
                # 3) there are no intermediate interior points inbetween the start and end point of the newly created edge. Note that if the point was added to the edge, we are looking at the left part of this edge (splitted_edges[0]). Otherwise, if the point is for instance merged, we are just looking at the edge onto which the last point was projected. 
                if splitted_edges is not None: noInteriorPointsInbetweenStartEnd = checkIfNoCornerPointsFromStart(splitted_edges[0], currentLine[-1])
                else: noInteriorPointsInbetweenStartEnd = checkIfNoCornerPointsFromStart(edge, currentLine[-1])
                # 4) the new edge does not contain too many new points, otherwise just add this edge instead of adjusting (currently set to 5 intermediate points ( = 7 used))
                # 5) the edge is not a new self-edge from an existing node, we will NOT adjust any edges in this case
                check_self_existing = False if ((temp_last_point == newNode) & (not temp_new)) else True
                # 6) we are not merging networks. In this case, we do not adjust existing newly created edges. 
                # 7) There must be at least one point that would adjust the edge
                ADJUST_SITUATION_1 = absorbedInNewEdge & (temp_last_point == start_point) & noInteriorPointsInbetweenStartEnd & (len(currentLine) < 7) & check_self_existing & (not settings[5]) & (len(currentLine)>2)

                if ADJUST_SITUATION_1:
                    """ _____________________________ ADJUST AN EDGE OF THE NETWORK BASED ON THIS EDGE __________________________"""
                    if do_print: print("Edge adjusts the geometry of existing (newly created) edge...")

                    # Store the temp_last_point, as it will be used for determining the edge_old and projectedPoint_OLD when we did NOT absorb and absorbed the edge (started_new_edge). In this case, the temp_last_point is the last point we had and we are going to remove it now. 
                    temp_last_point_OLD = Point(G_proj.nodes[temp_last_point]['x'], G_proj.nodes[temp_last_point]['y'])
                    edge_to_be_adjusted_add = None
                    # We start with removing the start and end point of the newly created edge. We do not need these as, we are adjusting the edge. Note that we only remove when this start (or end) point was not merged with an existing point. 
                    # 1) If both points were not merged. 
                    if temp_new & newNode_new: # Both are separate new points. Note that temp_last_point could be added double. Therefore, remove as second. If two-way, it doesnt matter.
                        re_old = remove_point(G_proj, newNode, do_print = do_print)
                        re = remove_point(G_proj, temp_last_point, do_print = do_print)
                        edge_to_be_adjusted = re[0] # Must exist
                        # If both the point were added in two ways, adjust both the (opposite) edges
                        if (len(re_old) > 1) & (len(re) > 1):
                            edge_to_be_adjusted_add = re[1]
                        # Note that the newNode is added explicitly, so not merged with an existing node. This means that the end of edge[1] has to be the end of the edge_to_be adjusted. In case we added the node(s) two ways, this might be the opposite edge. 
                        elif (len(re) > 1) & (edge_to_be_adjusted[1] != edge[1]): 
                            edge_to_be_adjusted = re[1]

                    # 2) If the temp_last_point has been added explicitly and the new node is merged with this temp_last_point
                    elif temp_new & (temp_last_point == newNode): # One node added to the network, self edge shapes the geometry
                        re = remove_point(G_proj, newNode, do_print = do_print)
                        edge_to_be_adjusted = re[0]
                        # If the new node (temp_last_point) was added in two ways (len(re) > 1), for instance a start point, we want to also adjust the opposite edge. 
                        if (len(re) > 1):
                            edge_to_be_adjusted_add = re[1]
                    # 3) If temp_last_point has been added explicitly, newNode not
                    elif temp_new & (not newNode_new): 
                        re = remove_point(G_proj, temp_last_point, do_print = do_print)
                        edge_to_be_adjusted = re[0]
                        # If we removed a point with a double edge, and the right end point of the new edge is NOT the newNode, we must have the other (opposite) recovered edge
                        if (len(re) > 1) & (edge_to_be_adjusted[1] != newNode): edge_to_be_adjusted = re[1]
                    # 4) If newNode has been added explicitly, temp_last_point not
                    elif (not temp_new) & newNode_new: 
                        re = remove_point(G_proj, newNode, do_print = do_print)
                        edge_to_be_adjusted = re[0] 
                        # If we removed a point with a double edge, and the left end point of the new edge is NOT the temp_last_point, we must have the other (opposite) recovered edge
                        if (len(re) > 1) & (edge_to_be_adjusted[0] != temp_last_point): edge_to_be_adjusted = re[1]
                    # 5) If both points already existed, we do not remove any points
                    else: edge_to_be_adjusted = edge 
                    
                    # Add the points to the geometry of edge_old
                    for np in currentLine[1:-1]:
                        _, _, best_edge = projectPointOnEdge(edge_to_be_adjusted, (Point(np), -1), alpha_bar = None, max_dist = float('inf'))
                        edge_to_be_adjusted, _, _, _ = include_point_in_newedge(G_proj, edge_to_be_adjusted, best_edge, (np[1], np[0], -1), old_point = (None, None), points_to_be_adjusted = [], settings = settings, crs = crs, max_actual_dist = None, temp_new = None, do_print = do_print)      
                        
                        if edge_to_be_adjusted_add is not None:
                            _, _, best_edge = projectPointOnEdge(edge_to_be_adjusted_add, (Point(np), -1), alpha_bar = None, max_dist = float('inf'))
                            edge_to_be_adjusted_add, _, _, _ = include_point_in_newedge(G_proj, edge_to_be_adjusted_add, best_edge, (np[1], np[0], -1), old_point = (None, None), points_to_be_adjusted = [], settings = settings, crs = crs, max_actual_dist = None, temp_new = None, do_print = do_print)      
                                               
                    
                    """ ___________________________________ UPDATE edge_old / projectedPoint_OLD _________________________________"""
                    if not merged_with_point: # Note that this means that the current point was absorbed by driving
                        # We find the projectionPoint of the current point (that was absorbed) close to this old point. Note that it must be close to the projected last point. We would like to satisfy the distance covered. Note that in the case that the current point was not absorbed AND absorbed, we want to project the current point as close as possible to temp_last_point, which is closer to the current_point than the previous point was. Therefore, we set the MMLD equal to 0. Moreover, because the maxdistance covered could have been violated, because a connection had to be established, we now also ignore this max-distance and only compare the anomalies. 

                        # Note that the connection point has been removed. Determine the last point that was used to create the edge (lp: last_point). Note that the new edge is not added. Therefore, the last point would be the 
                        if started_new_edge: 
                            lp = (temp_last_point_OLD, -1)
                            max_actual_dist = [float('inf'), (0, max_actual_dist[1][1])]
                        else: 
                            lp = (Point(tripdf.geometry[p-1].x, tripdf.geometry[p-1].y), -1)
                            max_actual_dist = [float('inf'), max_actual_dist[1]]
                        
                        # Now, we find the projectionPoint of the current point (that was absorbed) close to this old point. We do not care about the course, as this may be inaccurate after adjusting the edge. But, we do look at the (most likely) distance covered from the last point. (in other words, we just look at the anomalies) First, we determine the projected point of previous point on the adjusted edge.
                        _, projectedPoint, _ = projectPointOnEdge(edge_to_be_adjusted, lp, alpha_bar = None)
                        projectedPoint_OLD, _, edge_old, _ = findProjectionPoint(G_proj, (y_point,x_point,-1), close_to_edge = edge_to_be_adjusted, connecting = False, forward = True, temp_point = projectedPoint, settings = settings, max_actual_dist = max_actual_dist, crs= crs)
                        
                        # If no projection is found, it means that we have to project on a the opposite edge (note that two different edges might be adjusted)
                        if projectedPoint_OLD is None:
                            _, projectedPoint, _ = projectPointOnEdge(edge_to_be_adjusted_add, lp, alpha_bar = None)
                            projectedPoint_OLD, _, edge_old, _ = findProjectionPoint(G_proj, (y_point,x_point,-1), close_to_edge = edge_to_be_adjusted_add, connecting = False, forward = True, temp_point = projectedPoint, settings = settings, max_actual_dist = max_actual_dist, crs= crs)
                        
                        # Strange if no projection point could have been found up to this point. 
                        if projectedPoint_OLD is None:
                            print("Kind of weird. ")
                            stop
                    else: # If the last point was absorbed by merging, we find the point again (name may have changed) We could also adjust the previous....
                        node_to_be_merged = mergeNode(G_proj, (y_point, x_point), max_dist_merge = settings[2], indmax = settings[4])
                        if (node_to_be_merged is not None): 
                            projectedPoint_OLD = (G_proj.nodes[node_to_be_merged]['x'], G_proj.nodes[node_to_be_merged]['y'])
                            edge_old = (node_to_be_merged, node_to_be_merged, 0, LineString([projectedPoint_OLD, projectedPoint_OLD]))
                        else: # node_to_be_merged was removed. This means that node_to_be_merged was one of the end points of the edge that is used to adjust the other edge. Therefore, we re-project the point on the adjusted edge to find the projection parameters for the next round. 
                            _, projectedPoint_OLD,_ = projectPointOnEdge(edge_to_be_adjusted, (Point(x_point, y_point), -1), alpha_bar = None, max_dist = float('inf')) 
                            edge_old = edge_to_be_adjusted
                    """ __________________________________________________________________________________________________________"""
                
                else: # We may need to add the edge to the network
                    # Do NOT add the edge when: 
                    # (1) If the existing length is "close" to the length of the new edge (taking into account the projection distances)
                    SITUATION_1 = (length_deviation < 2 * settings[0][1]) 
                    
                    if SITUATION_1:
                        """ _____________________________________________ DO NOTHING ________________________________________________"""
                        if do_print: print("Edge already exists.. Remove the start node created for this edge (keep the end point, as it may be used for the next iteration..).")
                        
                        # If the start and end point of the edge are the same, we do not remove the start point. 
                        if temp_last_point != newNode:
                            # Check if the start node was merged with another node. In this case, do not remove the whole node (only adjust the name)
                            node_parts = str(temp_last_point).split("/")
                            if (len(node_parts) == 1): # Start node was not merged, remove point
                                if do_print: print("Start node of this edge: " + str(temp_last_point) + " removed.")
                                remove_point(G_proj, temp_last_point, do_print = do_print)
                            else: # Only adjust name
                                try: temp_last_point_new = int("/".join(node_parts[:-1]))
                                except: temp_last_point_new = "/".join(node_parts[:-1]) # We may set all to string from the beginning, but this is less efficient (string vs int)
                                nx.relabel_nodes(G_proj, {temp_last_point: temp_last_point_new}, copy = False)
                                if do_print: print("Start node of this edge: " + str(temp_last_point) + " adjusted to: " + str(temp_last_point_new))
                        """ __________________________________________________________________________________________________________"""
                        
                        """ ___________________________________ UPDATE edge_old / projectedPoint_OLD _________________________________"""
                        if not merged_with_point: # Note that this means that the last point was absorbed by driving
                            # If we did not add an edge, we use the end point of the edge that would have been added (we did not remove it). 
                            projectedPoint = (G_proj.nodes[newNode]['x'], G_proj.nodes[newNode]['y'])
                            edge_old = (newNode, newNode, 0, LineString([projectedPoint, projectedPoint]))
                           
                            # We again try to find the projection point close to this old point. However, now, we set minimal most likely distance covered equal to zero, as this may not be accurate anymore. We could project the current point closer to the end point of the edge, but NOT further away. Therefore, we only adjust the minimal most likely distance covered. 
                            max_actual_dist = [float('inf'), (0, max_actual_dist[1][1])]
                            projectedPoint_OLD, _, edge_old, _ = findProjectionPoint(G_proj, (y_point,x_point,tripdf.Course[p]), close_to_edge = edge_old, connecting = False, forward = True, temp_point = projectedPoint, settings = settings, max_actual_dist = max_actual_dist, crs = crs)
                            
                            # If no projection is found, we just use this last node added as our close-to point for next round. May happen for instance when self edge was removed. 
                            if projectedPoint_OLD is None:
                                print("THIS IS STRANGE, AS THE POINT USED TO BE ABSORBED...")
                                stop
                        else: # If the last point was absorbed by merging, we find the point again (name may have changed) We could also adjust the preivous....
                            node_to_be_merged = mergeNode(G_proj, (y_point, x_point), max_dist_merge = settings[2], indmax = settings[4])
                            if (node_to_be_merged is not None): 
                                projectedPoint_OLD = (G_proj.nodes[node_to_be_merged]['x'], G_proj.nodes[node_to_be_merged]['y'])
                                edge_old = (node_to_be_merged, node_to_be_merged, 0, LineString([projectedPoint_OLD, projectedPoint_OLD]))
                            else: # Node to be merged was removed. This cannot happen, as we only removed temp_last_point and this could not be the node_to_be_merged (unless it is newNode, but then we did not remove it..)
                                print("THAT IS VERY STRANGE....")
                                stop 
                        """ __________________________________________________________________________________________________________"""
                    else: # Add the new edge 
                        """ ___________________________ ADD NEW EDGE TO THE NETWORK _________________________"""
                        if do_print: print("Add the edge between " + str(temp_last_point) + " and " + str(newNode) + ".")
                        keyNew = max([item[2] for item in G_proj.edges(temp_last_point, newNode, keys = True) if ((item[0] == temp_last_point) & (item[1] == newNode))], default=-1) + 1
                        G_proj.add_edge(temp_last_point, newNode, osmid = 'Edge: ' + str(trip) + "_" + str(p), new = True, driven = True, DatesVelocities = curDateVels, ref = None, highway = None, oneway = not settings[3], length = computeLengthLinestring(LineString(currentLine), method = 'euclidean'), geometry = LineString(currentLine), close_to_point_start = close_to_point_start, close_to_point_end = [(projectedPoint_OLD[1],projectedPoint_OLD[0],tripdf.Course[p]), max_actual_dist], maxspeed = None, service = None, bridge= None, lanes = None, u = temp_last_point, v = newNode, key = keyNew)
                        # Add the opposite new edge (if two_way considered). Also, check if there is already another edge between the two nodes (to use the right key)
                        if settings[3]:
                            keyNew = max([item[2] for item in G_proj.edges(newNode, temp_last_point, keys = True) if ((item[0] == newNode) & (item[1] == temp_last_point))], default=-1) + 1
                            G_proj.add_edge(newNode, temp_last_point, osmid = 'Edge: ' + str(trip) + "_" + str(p) + "_r", new = True, driven = True, DatesVelocities = curDateVels, ref = None, highway = None, oneway = not settings[3], length = computeLengthLinestring(LineString(currentLine[::-1]), method = 'euclidean'), geometry = LineString(currentLine[::-1]), close_to_point_start = [(projectedPoint_OLD[1],projectedPoint_OLD[0],tripdf.Course[p]), max_actual_dist], close_to_point_end = close_to_point_start, maxspeed = None, service = None, bridge= None, lanes = None, u = newNode, v = temp_last_point, key = keyNew)
                        """ _________________________________________________________________________________"""
                        
                        """ ___________________________________ UPDATE edge_old / projectedPoint_OLD _________________________________"""
                        if not merged_with_point: # Note that this means that the current point was absorbed by driving
                            # We find the projectionPoint of the current point (that was absorbed) close to this old point. Note that it must be close to the projected last point. We would like to satisfy the distance covered. Note that in the case that the current point was not absorbed AND absorbed, we want to project the current point as close as possible to the connection of the new edge, which is closer to the current_point than the previous point was. Therefore, we set the MMLD equal to 0. Moreover, because the maxdistance covered could have been violated, because a connection had to be established, we now also ignore this max-distance and only compare the anomalies. 

                            # Determine the last point that was used to create the edge (lp: last_point). If the current point was not absorbed AND absorbed, we use the temp_last_point as this point is the last point that was used to create the edge.
                            if started_new_edge: 
                                lp = (Point(G_proj.nodes[temp_last_point]['x'], G_proj.nodes[temp_last_point]['y']), -1)
                                max_actual_dist = [float('inf'), (0, max_actual_dist[1][1])]
                            else: 
                                lp = (Point(tripdf.geometry[p-1].x, tripdf.geometry[p-1].y), -1)
                                max_actual_dist = [float('inf'), max_actual_dist[1]]
                                
                            # Define the newly added edge  
                            edge_added = (temp_last_point, newNode, keyNew, LineString(currentLine))
                            # Determine the projected point of the last point on the new/adjusted edge.
                            _, projectedPoint, _ = projectPointOnEdge(edge_added, lp, alpha_bar = None, max_dist = float('inf'))     
                            projectedPoint_OLD, _, edge_old, _ = findProjectionPoint(G_proj, (y_point,x_point,tripdf.Course[p]), close_to_edge = edge_added, connecting = False, forward = True, temp_point = projectedPoint, settings = settings, max_actual_dist = max_actual_dist, crs= crs)
                            # If no projection is found, 
                            if projectedPoint_OLD is None:
                                print("THIS IS STRANGE, AS THE POINT USED TO BE ABSORBED...")
                                stop
                        else: # If the current point was absorbed by merging, we find this merging node again (name may have changed). We could also adjust the previous....
                            node_to_be_merged = mergeNode(G_proj, (y_point, x_point), max_dist_merge = settings[2], indmax = settings[4])
                            if (node_to_be_merged is not None): 
                                projectedPoint_OLD = (G_proj.nodes[node_to_be_merged]['x'], G_proj.nodes[node_to_be_merged]['y'])
                                edge_old = (node_to_be_merged, node_to_be_merged, 0, LineString([projectedPoint_OLD, projectedPoint_OLD]))
                            else: # Node to be merged was removed.
                                print("VERY STRANGE BECAUSE NO POINTS ARE REMOVED.")
                                stop
                        """ __________________________________________________________________________________________________________"""
                
                # Update parameters for next round
                first_point_of_edge = True
                temp_last_point = None
                """ _________________________________________________________________________________________________________"""

        else: # Point p cannot be merged and it needs to be added to the graph (as a corner point).  
            if first_point_of_edge: 
                if do_print: print("||||| Point cannot be merged with the current network. Start a new edge!")
                # Reproject point without incorporating bearing (because this did not make sense). This will result in the connection point of the new edge with the existing network
                if do_print: print("--------------------------")
                if do_print: print("Reproject the point onto the graph without incorporating bearing.")
                if do_print: print("Close to the edge (forward): {0}".format(edge_old))
                projectedPoint, closestDistance, edge, be = findProjectionPoint(G_proj, (y_point,x_point,-1), close_to_edge = edge_old, connecting = True, forward = True, temp_point = projectedPoint_OLD, settings = settings, max_actual_dist = max_actual_dist, crs = crs)
                if do_print: print("Projected point: {0}".format(projectedPoint))
                if do_print: print("Closest distance: " + str(closestDistance) + " meters.")
                if do_print: print("Projected onto edge: {0}".format(edge))
                
                # Add this point to the network (either explicitly or by merging). currentPoint will be the point that is finally added (note that [name] could have been merged with an existing node). In this situation, we also want to know whether the point is added explicitly (temp_new = True) or merged with an existing point (temp_new = False). 
                """ ___________________________ ADD (MAKE SURE) THE CONNECTION POINT TO THE NETWORK _________________________"""
                currentPoint, temp_new, _, _, _, _= ensure_point_in_network(G_proj, projectedPoint, edge, be, name = 'Node_p: '+ str(trip) + "_" + str(p), settings = settings, two_way = settings[3], temp_last_point = None, start_point = None, do_print = do_print)
                """ _________________________________________________________________________________________________________"""
           
                # Start a new edge that connects this point to the current network, add first piece of the edge
                currentLine = [(G_proj.nodes[currentPoint]['x'], G_proj.nodes[currentPoint]['y'])]
                curDateVels = "|"+curDateVel
                
                # Update parameters for the new edge (to be created). Note that the starting connection point of this new edge is based on projectedPoint_OLD and its corresponding course. Moreover, we can use the distance covered from this previous point to the new point in future iterations. 
                temp_last_point = currentPoint
                close_to_point_start = [(projectedPoint_OLD[1], projectedPoint_OLD[0], tripdf.Course[p-1]), max_actual_dist]
                
                # Update parameters 
                first_point_of_edge = False # The next point is not the first point of an edge
                started_new_edge = True 

                # The next point (try to absorb the same point again) does not need to be close to an existing point (previous point(s) not absorbed in the network). (Note that therefore the same point might now be absorbed)
                edge_old = None 
                projectedPoint_OLD = None 

                if do_print: print()
                if do_print: print("||||||||||||||||||||||||||||||||||||||||||||")
                if do_print: print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                if do_print: print()
                continue
                
            """ _________________________________ ADJUST THE OPPOSITE EDGE (WHEN CREATING) _______________________________"""
            # Check if there is an opposite (new) edge that may be adjusted. Note that when merging networks, we do not adjust edge geometries (settings[5] must be False)
            if (not settings[5]): 
                ce = get_nearest_edge_FULL(G_proj, Point(x_point, y_point), settings[4], return_geom=True, return_dist=True)
                ce = [a for a in ce if a[1] < settings[0][1]] # Check if there is at least one possible edge
                if (len(ce)>0): # if there is at least 1, we have an opposite edge.
                    # Note that closestEdge is never the current edge, as this edge does not exist yet.
                    closestEdge = tuple(list(ce[0][0]) + [ce[0][1]])
                    
                    # First, we check whether we can use the geometry of the opposite edge for the geometry of our currently creating edge. This only happens when the opposite edge is also a newly created edge. We assume that we cannot be an opposite edge to an existing edge, as these edges would be registered two-way in this case. Secondly, when the opposite edge is newly created, we also adjust it with the information of this new point. 
                    if G_proj.edges[closestEdge[0],closestEdge[1],closestEdge[2]]['new']:
                        """ ___________________________ ADJUST THE CURRENTLY CREATING EDGE (BASED ON THIS OPPOSITE EDGE) ___________________________"""
                        # Project the current and the previous GPS point onto this closestEdge. We need the projected point of the current point later on, when we want to check whether this projected point is far away from a current corner point. Therefore, we already save it in projectedPoint_r. Note that we only incorporate additinol corner points of an opposite edge when both points (previous and current) can be projected on the same edge. If there is a node inbetween, we do not incorporate points. 
                        proj_dist_previous, _, best_edge2 = projectPointOnEdge(closestEdge, (Point(currentLine[-1]), -1), alpha_bar = None, max_dist = float('inf'))# Previous (corner) point
                        _, projectedPoint_r, best_edge = projectPointOnEdge(closestEdge, (Point(x_point,y_point), -1), alpha_bar = None, max_dist = float('inf')) # Current (corner) point
                        
                        # If both points are close enough to the edge (note that the current point is already proven to be close..), we check if there are points that are missed by this edge, but that are in the opposite edge.
                        if (proj_dist_previous <= settings[0][1]):
                            # Extract geometry of the opposite edge. 
                            geom = G_proj.edges[closestEdge[0],closestEdge[1],closestEdge[2]]['geometry']
                            edge_points_list = [y for y in geom.coords]
                            
                            #  Add the missing points to the geometry of the currently creating edge
                            a, b = best_edge, best_edge2
                            if (b - a) > 0: currentLine += reversed(edge_points_list[a+1:(b+1)]) # Note that the (b+1)th element is not included in the list
                        """ ________________________________________________________________________________________________________________________"""

                        """ ____________________________________________ ADJUST THE OPPOSITE EXISTING EDGE _________________________________________"""
                        # Check if closestEdge is a new edge (that may incorporate adjustments) and check whether the projected point is not close to an existing interior point (if so, we do not adjust the edge geometry)
                        if (not closeToCorner(projectedPoint_r, closestEdge, max_dist = close_to_corner)):#&(not closeToCorner(G_proj,(x_point,y_point),closestEdge, max_dist=10)): 
                            # Adjust closestEdge. Note that there is no old_point, so we can not use this information when we are adding a point to the first line piece of an edge. As before, when creating a new edge and, due to this inclusion, and if a new connection point is merged with our starting point of our newly created edge, we assume that this point is no longer new (as it now has two different functions: connection point AND starting point). Moreover, due to the adjustment procedure, temp_last_point may have changed. Therefore, we also want to receive updates about the changes in this node (name).
                            if do_print: print("Opposite edge {0} also adjusted!".format(closestEdge))
                            closestEdge, points_adjusted, _, temp_new = include_point_in_newedge(G_proj, closestEdge, best_edge, (y_point, x_point, -1), old_point = (None, None), points_to_be_adjusted = [temp_last_point], settings = settings, crs = crs, max_actual_dist = max_actual_dist, temp_new = temp_new, do_print = do_print)  
                            temp_last_point = points_adjusted[0]
                        """ ________________________________________________________________________________________________________________________"""
            """ __________________________________________________________________________________________________________"""
            
            if p == (len(tripdf)-1): # This means that it is an end point of a trip. Explicitly add this point as a node and finish the edge.
                if do_print: print("End node added explicitly.")
                # First check if the point can be merged in an existing edge. Therefore, we first find the closest edges to this start node
                ce = get_nearest_edge_FULL(G_proj, Point(x_point, y_point), settings[4], return_geom=True, return_dist=True)
                ce = [a for a in ce if a[1] < settings[0][1]]# Remove non-options beforehand
                end_node_added_explicitly = False
                # If edges are left, determine projected points, etc.
                if (len(ce)>0): 
                    # Check (and find) closestEdge that satisfies the maximum projection distance for that edge (only useful when bar{d} and barbar{d} are different (then we pick the top one). Note that an edge might be further away, but this may be new edge for which it does satisfy the maximum projection distance. 
                    ind = 0
                    closestEdge = tuple(list(ce[ind][0]) + [ce[ind][1]])
                    while (closestEdge[4] > get_max_distance_bound(G_proj, closestEdge, settings)) & (ind < len(ce)-1):
                        ind += 1
                        closestEdge = tuple(list(ce[ind][0]) + [ce[ind][1]])
                        
                    # Still check whether we did not run out of options in the while loop above
                    if (closestEdge[4] < get_max_distance_bound(G_proj, closestEdge, settings)): # End point added to an existing edge
                        # Find projected point (we already have the distance)
                        if do_print: print("End point projected onto existing edge: {0}".format((closestEdge[0],closestEdge[1],closestEdge[2])))
                        _, projectedPoint, be = projectPointOnEdge(closestEdge, (Point(x_point, y_point), -1), alpha_bar = None, max_dist = float('inf'))
                        if do_print: print("Projected point: {0}".format(projectedPoint))
                        if do_print: print("Closest distance: " + str(closestEdge[4]) + " meters.")
                
                        # Add this point to the network (either explicitly or by merging). Note that we add it in two ways, because we do not have info about the direction of this point. last_point will be the point that is finally added (note that [name] could have been merged with an existing node). Moreover, notice that, in constrast with the starting point, we now also want to knwo whether the starting node of the new edge (temp_last_point) is changed. Therefore, we add this as one of the outputs.
                        """ ______________________________ ADD (MAKE SURE) THE END POINT TO THE NETWORK _____________________________"""
                        last_point, _, _, temp_last_point, _, _= ensure_point_in_network(G_proj, projectedPoint, closestEdge, be, name = 'End node: ' + str(trip) + "_" + str(p), settings = settings, two_way = True, temp_last_point = temp_last_point, start_point = None, do_print = do_print)
                        """ _________________________________________________________________________________________________________"""
                    # If we were not able to project the end point onto an edge, we add the end node explicitly. 
                    else: 
                        last_point = 'End node: ' + str(trip) + "_" + str(p)
                        G_proj.add_node(last_point, y = y_point, x = x_point, osmid = last_point, geometry = Point(x_point, y_point))
                        end_node_added_explicitly = True
                else: # If we were not able to project the end point onto an edge, we add the end node explicitly. 
                    last_point = 'End node: ' + str(trip) + "_" + str(p)
                    G_proj.add_node(last_point, y = y_point, x = x_point, osmid = last_point, geometry = Point(x_point, y_point))  
                    end_node_added_explicitly = True
                
                # At this point, we have added the end point to the network. Either by adding it to an existing edge, or by adding it expliclty as a new node. 
                # Either way, the name of the last node (the end point) is referred to as [last_point].
                
                # Add last piece of the edge. If the end node is not added explicitly, we do also add the last corner point (point itself) besides the location of the end node.
                if do_print: print("Possible new edge from: " + str(temp_last_point) + " to " + str(last_point))
                if not end_node_added_explicitly: currentLine += [(x_point, y_point)]
                currentLine += [(G_proj.nodes[last_point]['x'], G_proj.nodes[last_point]['y'])]
                
                # Check the length of the new edge (from temp_last_point to newNode (the previously projected point))
                length_new_edge = computeLengthLinestring(LineString(currentLine), method = 'euclidean')
                try: length_existing = nx.shortest_path_length(G_proj, source = temp_last_point, target = last_point, weight='length')
                except: length_existing = float('inf')
                if do_print: print("Length of the possible new edge: " + str(length_new_edge))
                if do_print: print("Length of the path between "+ str(temp_last_point) + " and " + str(last_point) + " (without the edge added): " + str(length_existing))
                length_deviation = abs(length_existing - length_new_edge)
                
                """ __________________________________ MERGING NODES MAY REDUCE THE LENGTH __________________________________"""
                # It may be the case that we can also travel via an opposite edge to the new node. In other words, an shortest path already existed, but we were not able to traverse this, due to the projection point(s) of the start and end point of the new edge (projected onto different edges). Therefore, we check if an SP with a smaller deviation exists, when using opposite edge(s). Note that we only do this when we are not considering the two-way situations. If two-ways is activated, does not make sense (useless work), becuase each point is already added in two ways, and such situations will not occur. 
                if not settings[3]: 
                    # Check if temp_last_point could be merged. Note that the last point can never be merged. It is either explicitly added, or added (in two ways) to an existing edge. In ohter words, there is no way for reducing the deviation. We only check what happens when re-projecting the start point of the edge (temp_last_point).
                    ce = get_nearest_edge_FULL(G_proj, Point((G_proj.nodes[temp_last_point]['x'], G_proj.nodes[temp_last_point]['y'])), settings[4], return_geom=True, return_dist=True)
                    ce = [a for a in ce if a[1] < 1]
                    # If at least 2 edges are left, determine projected points, etc. (note that 1 of the two is the edge itself)
                    if (len(ce) > 1): 
                        # Check (and find) closestEdge that satisfies the maximum projection distance for that edge (only useful when bar{d} and barbar{d} are different (then we pick the top one). Note that an edge might be further away, but this may be new edge for which it does satisfy the maximum projection distance. 
                        ind = 0
                        closestEdge_temp = tuple(list(ce[ind][0]) + [ce[ind][1]])
                        # If temp_last_point is part of one of the end points of the opposite edge, we are not getting a reduced deviation. We are looking for an edge into which the temp_last_point may be merged. 
                        while (temp_last_point in [closestEdge_temp[0], closestEdge_temp[1]]) & (ind < len(ce)-1):
                            ind += 1
                            closestEdge_temp = tuple(list(ce[ind][0]) + [ce[ind][1]])
                        
                        # Still check whether we did not run out of options in the while loop above
                        if (temp_last_point not in [closestEdge_temp[0], closestEdge_temp[1]]):
                            from_point = ((G_proj.nodes[temp_last_point]['x'], G_proj.nodes[temp_last_point]['y']), closestEdge_temp)
                            to_point = ((G_proj.nodes[last_point]['x'], G_proj.nodes[last_point]['y']), (last_point, last_point))
                            length_existing_new_1, _, _, _ = get_SP_distance(G_proj, from_point = from_point , to_point = to_point)
                            length_existing_new_11 = abs(length_existing - length_existing_new_1)
                            
                            # If there is better deviation, then apply the corresponding merging procedures.
                            if length_existing_new_11 < length_deviation:
                                length_deviation = length_existing_new_11
                                # Merge temp_last_point in the corresponding edge
                                _, _, be_temp = projectPointOnEdge(closestEdge_temp, (Point((G_proj.nodes[temp_last_point]['x'], G_proj.nodes[temp_last_point]['y'])), -1), alpha_bar = None, max_dist = float('inf'))
                                add_point_expl(G_proj, point = (G_proj.nodes[temp_last_point]['x'], G_proj.nodes[temp_last_point]['y']), edge = closestEdge_temp, be = be_temp, node_name = temp_last_point, settings = settings, merge = True, two_way = False, do_print = do_print)      
                """ _________________________________________________________________________________________________________"""
                                
                # Do NOT add the edge when: 
                # (1) If the existing length is "close" to the length of the new edge (taking into account the projection distances)
                SITUATION_1 = (length_deviation < 2 * settings[0][1]) 
                if SITUATION_1:
                    if do_print: print("Edge already exists.. Remove the start- and end node created for this edge.") 

                    # If the start and end point of the edge are different, we remove both the points. Note that, compared to the non-last-point situation, we do not need to keep the end point for the next iteration. Therefore, we can also remove the end point.
                    if temp_last_point != last_point:
                        # Check if the start node was merged with another node. In this case, do not remove the whole node (only adjust the name)
                        node_parts = str(temp_last_point).split("/")
                        if (len(node_parts) == 1): # Start node was not merged, remove point
                            if do_print: print("Start node of this edge: " + str(temp_last_point) + " removed.")
                            remove_point(G_proj, temp_last_point, do_print = do_print)                                
                        else: # Only adjust name
                            try: temp_last_point_new = int("/".join(node_parts[:-1]))
                            except: temp_last_point_new = "/".join(node_parts[:-1]) # We may set all to string from the beginning, but this is less efficient (string vs int)
                            nx.relabel_nodes(G_proj, {temp_last_point: temp_last_point_new}, copy = False) # Note that we do not want a new graph, we want to adjust the current graph (copy = False)
                            if do_print: print("Start node of this edge: " + str(temp_last_point) + " adjusted to: " + str(temp_last_point_new))
                        # Check if the end node [last_point] was merged with another node. In this case, do not remove the whole node (only adjust the name)
                        node_parts = str(last_point).split("/")
                        if (len(node_parts) == 1): # End node was not merged, remove point
                            if do_print: print("End node of this edge: " + str(last_point) + " removed.")
                            remove_point(G_proj, last_point, do_print = do_print)                                
                        else: # Only adjust name
                            try: last_point_new = int("/".join(node_parts[:-1]))
                            except: last_point_new = "/".join(node_parts[:-1]) # We may set all to string from the beginning, but this is less efficient (string vs int)
                            nx.relabel_nodes(G_proj, {last_point: last_point_new}, copy = False) # Note that we do not want a new graph, we want to adjust the current graph (copy = False)
                            if do_print: print("End node of this edge: " + str(last_point) + " adjusted to: " + str(last_point_new))
                    # If the start and end node of the new edge are the same, we only have to remove this (single) point 
                    else:
                        # Note that the last point was merged with the start point. Check whether the start point was merged with another point. In this case, do not remove the whole node (only adjust the name). Note that, now, having 2 parts in the node's name, means that the node was NOT merged (name of the start and name of the end node combined).
                        node_parts = str(temp_last_point).split("/")
                        if len(node_parts) == 2:
                            if do_print: print("Start/End node of this self-edge: " + str(temp_last_point) + " removed.")
                            remove_point(G_proj, temp_last_point, do_print = do_print) 
                        else:
                            try: temp_last_point_new = int("/".join(node_parts[:-2]))
                            except: temp_last_point_new = "/".join(node_parts[:-2]) # We may set all to string from the beginning, but this is less efficient (string vs int)
                            nx.relabel_nodes(G_proj, {temp_last_point: temp_last_point_new}, copy = False) # Note that we do not want a new graph, we want to adjust the current graph (copy = False)
                            if do_print: print("Start/End node of this self-edge: " + str(temp_last_point) + " adjusted to: " + str(temp_last_point_new))         
                else: 
                    """ ___________________________ ADD NEW EDGE TO THE NETWORK _________________________"""
                    if do_print: print("Add the edge between " + str(temp_last_point) + " and " + str(last_point) + ".")
                    keyNew = max([item[2] for item in G_proj.edges(temp_last_point, last_point, keys = True) if ((item[0] == temp_last_point) & (item[1] == last_point))], default=-1) + 1
                    G_proj.add_edge(temp_last_point, last_point, osmid = 'Edge: ' + str(trip) + "_" + str(p), new = True, driven = True, DatesVelocities = "|"+curDateVel, ref = None, highway=None, oneway= not settings[3], length = computeLengthLinestring(LineString(currentLine), method= "euclidean"), geometry = LineString(currentLine), close_to_point_start = close_to_point_start, close_to_point_end = 'End', maxspeed = None, service = None, bridge= None, lanes = None, u = temp_last_point, v = last_point, key = keyNew)
                    # Add the opposite new edge (if two_way considered). Also, check if there is already another edge between the two nodes (to use the right key)
                    if settings[3]:
                        keyNew = max([item[2] for item in G_proj.edges(last_point, temp_last_point, keys = True) if ((item[0] == last_point) & (item[1] == temp_last_point))], default=-1) + 1
                        G_proj.add_edge(last_point, temp_last_point, osmid = 'Edge: ' + str(trip) + "_" + str(p) + "_r", new = True, driven = True, DatesVelocities = "|"+curDateVel, ref = None, highway = None, oneway= not settings[3], length = computeLengthLinestring(LineString(currentLine[::-1]), method = 'euclidean'), geometry = LineString(currentLine[::-1]), close_to_point_start = 'End', close_to_point_end = close_to_point_start, maxspeed = None, service = None, bridge= None, lanes = None, u = last_point, v = temp_last_point, key = keyNew)
                    """ _________________________________________________________________________________"""
            else:
                # Extend the geometry of the currently forming edge. Note that in the case that we are creating a new edge, we do include ALL point in its geometry. The idea is to start with the best approximation possible, and then be more strict when adding points to the geometry.   
                currentLine = currentLine + [(x_point, y_point)]
                if do_print: print("||||| Node incorporated into the current edge.") 
                
                curDateVels = add_datvels(curDateVels, curDateVel)
                #curDateVels += "|"+curDateVel
                # Update parameters
                started_new_edge = False # If we just added a new corner point, it means that the current point could not have been absorbed.  
                
        # Go to the next point in the trace
        p += 1
        if do_print: print()
        if do_print: print()
    





    


  
def ExtendGraphWithOSM_NEW(used_crs, ex_poly, points, buf = 500, merging = False):
    """
    Create a polygon that covers all points in the [points] dataset. Use a 
    default buffer of 500 meters here. Project using the same CRS [used_crs] 
    that is used during the algorithm. 
    
    Then, check if this polygon is contained in the existing polygon [ex_poly]
    in the algorithm. If so, this means that all streets of OSM are within the 
    graph and we can use this new set of points to extend the graph. If not, 
    we may want to extend the graph (#TODO). Currently, if this is not the case, 
    we do not use this [points] data to extend the graph. 

    Parameters
    ----------
    used_crs : string
    ex_poly : (Multi)Polygon
    points : DataFrame 
        
    Returns
    -------
    ex_poly :  Polygon
        The existing large polygon that is used. 
    ch_polygon : Polygon
        Small polygon that covers the points in the [points] dataset (including a buffer)
    
    """
    # First, create a new polygon that includes all new points
    test_temp = geopandas.GeoDataFrame(points, geometry = geopandas.points_from_xy(points.Longitude, points.Latitude), crs="EPSG:4326")
    test_temp = ox.project_gdf(test_temp, to_crs = used_crs)#, crs = "EPSG:4326")    
    ch_polygon = MultiPoint(list(map(lambda x, y:(x,y), test_temp.geometry.x, test_temp.geometry.y))).convex_hull.buffer(buf)
    ch_polygon_proj = ox.projection.project_geometry(ch_polygon, crs = test_temp.crs, to_latlong = True)[0]
    # If the existing polygon contains the new polygon, just return the existing polygon
    if (ex_poly.contains(ch_polygon_proj) | merging): return ex_poly, None, ch_polygon
    else: return (ex_poly, None, None)  

