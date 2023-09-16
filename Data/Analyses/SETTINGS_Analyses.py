"""-------------------------------------------------------------------------"""
"""---------------------- PARAMETERS OF THE ANALYSIS -----------------------"""
"""-------------------------------------------------------------------------"""
#_______________________ Initial/Extended graphs _____________________#
import os
main_path = "/".join(os.getcwd().split("/")[:-1])

# Graphs 
PATH_TO_GRAPH = "Data/_Graph_SIMPLE_ENHANCED/" # Extended graph

CASENAME = "CASE00"
#_____________________________________________________________________#

#______________________________ OD pairs _____________________________#
PATH_TO_FROMTO_COMBINATIONS = "Data/PEMPEM_stops_125_3_80_10_5_5_ARTCOURSE_DIRECTION_False_1311.csv"
#_____________________________________________________________________#

#________________________ Distance thresholds ________________________#
MAX_PROJECTION = 30 # 30, 50, 70, 90
MAX_DISTANCE_OPPOSITE_EDGE = 5
#_____________________________________________________________________#

#_________________________ Selection settings ________________________#
SELECTION = None # None, means all OD pairs in PATH_TO_FROMTO_COMBINATIONS
PLOTPATH = False # Recommendation: only use when 1 ID selected
#_____________________________________________________________________#

#__________________________ Savings settings _________________________#
FNAME = "SP_diffs_"+CASENAME+"_"+str(MAX_PROJECTION)+"_"+str(MAX_DISTANCE_OPPOSITE_EDGE)+"_NEW_0716_BEST.csv"
#_____________________________________________________________________#

#________________________ Mapbox accesstoken _________________________#
with open('../mapbox_accesstoken.txt') as f: mapbox_accesstoken = f.readlines()[0]
#_____________________________________________________________________#
"""-------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------"""



