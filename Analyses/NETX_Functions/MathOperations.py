from math import radians, cos, sin, asin, sqrt, atan2, pi

"""--------------------------------------------- BEARING FUNCTIONS -----------------------------------------"""
"""---------------------------------------------------------------------------------------------------------"""
#TODO DONE    
def computeBearing(start_point, end_point):
    """
    This function computes the bearing between two points: [start_point] and [end_point].
    
    Parameters
    ----------
    start_point : Point(x,y)
    end_point : Point(x,y)
    
    Returns
    -------
    brng_new : float
    
    """
    brng_new = (180/pi) * atan2(end_point.x-start_point.x, end_point.y - start_point.y)
    if brng_new >= 0: return brng_new
    else: return brng_new+360

#TODO DONE
def computeAngularDifference(alpha, beta):
    """
     Returns the angular difference between the angles [alpha] and [beta] (in degrees)
    
     Parameters
     ----------
     alpha : float
         Angle 1 in degrees
     beta : float
         Angle 2 in degrees

     Returns
     -------
     check : float
         angular difference in degrees
    
    """
    check = abs(float(alpha)-float(beta))
    if check > 180: check = abs(check- 360)
    return check

#TODO DONE
def meanCourse(alpha, beta): 
    """
     Returns the mean bearing of two angles: [alpha] and [beta] (in degrees)
    
     Parameters
     ----------
     alpha : float
         Angle 1 in degrees
     beta : float
         Angle 2 in degrees

     Returns
     -------
     mean_1/2 : float
         mean bearing between the two angles in degrees (sharp angle)
    """
    # At first, there are two means, one with a sharp angle and the other with a wide angle
    mean_1 = (alpha + beta)/2
    mean_2 = mean_1 + 180
    if (mean_2 > 360):
        mean_2 = mean_2 - 360
    
    # We choose the mean angle from the sharpest angle
    diff_1 = computeAngularDifference(alpha, mean_1)
    diff_2 = computeAngularDifference(alpha, mean_2)
    
    if diff_1 < diff_2: return mean_1
    else: return mean_2
    
"""-------------------------------------------- DISTANCE FUNCTIONS -----------------------------------------"""
"""---------------------------------------------------------------------------------------------------------""" 
#TODO DONE
def dist(x1, y1, x2, y2, x3, y3): 
    """
     Returns the projected distance between the point (x3,y3) and the edge (x1,y1) -> (x2,y2)
     ** Note that this is done in a cartesian system. 
    """
    px = x2-x1
    py = y2-y1

    norm = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = (dx*dx + dy*dy)**.5

    return x, y, dist

#TODO DONE
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified as lat/long combinations
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000

#TODO DONE
def computeLengthLinestring(line, method = "haversine"): 
    """
     Returns the length of the linestring [line], using the [method].
    
     Parameters
     ----------
     line : LineString
     method : string 
        determines how to compute the length between points (e.g, haversine or euclidean)

     Returns
     -------
     distance : float
         length of the [line]
    
    """
    # Extract all coordinates
    numCoords = len(line.coords) - 1
    
    distance = 0
    for i in range(0, numCoords):
        point1 = line.coords[i]
        point2 = line.coords[i + 1]
        if method == "haversine": distance += haversine(point1[0], point1[1], point2[0], point2[1])
        else: distance += sqrt((point1[0]-point2[0])**2 + (point1[1] - point2[1])**2)
    return distance
"""---------------------------------------------------------------------------------------------------------"""
"""---------------------------------------------------------------------------------------------------------"""    