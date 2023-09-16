import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import osmnx as ox
import pickle
import itertools
import os
import networkx as nx
from SETTINGS import *


def testModel(model, df_test_KR, df_test_UR, scaler):
    ###############################################################
    #################### EVALUATE PERFORMANCE #####################
    ###############################################################
    # Test data
    x_test_info_KR, x_test_image_KR, y_test_KR = df_test_KR
    x_test_info_UR, x_test_image_UR, y_test_UR = df_test_UR
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # #  KNOWN ROADS  # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Make predictions on the testing data and define the true speeds as a list
    preds_KR = model.predict([x_test_image_KR, x_test_info_KR])
    pred_KR = [item for sublist in preds_KR for item in sublist]
    true_KR = list(y_test_KR.iloc[:,0])

    # Performance metrics
    se = (np.asarray(true_KR) - np.asarray(pred_KR))**2 # Squared errors (SE)
    ae = abs(np.asarray(true_KR) - np.asarray(pred_KR)) # Absolute errors (AE)
    print("-------------- Known roads --------------")
    print("Mean squared error (MSE): " + str(se.mean())) # or use mean_squared_error(true,pred)
    print("Root mean squared error (RMSE): " + str(sqrt(se.mean()))) # or use mean_squared_error(true,pred)
    print("Mean absolute error : " + str(ae.mean()))
    print("-----------------------------------------")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # #  UNKNOWN ROADS  # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Make predictions on the testing data and define the true speeds as a list
    preds_UR = model.predict([x_test_image_UR, x_test_info_UR])
    pred_UR = [item for sublist in preds_UR for item in sublist]
    true_UR = list(y_test_UR.iloc[:,0])

    # Performance metrics
    se = (np.asarray(true_UR) - np.asarray(pred_UR))**2 # Squared errors (SE)
    ae = abs(np.asarray(true_UR) - np.asarray(pred_UR)) # Absolute errors (AE)
    print("------------- Unknown roads -------------")
    print("Mean squared error (MSE): " + str(se.mean())) # or use mean_squared_error(true,pred)
    print("Root mean squared error (RMSE): " + str(sqrt(se.mean()))) # or use mean_squared_error(true,pred)
    print("Mean absolute error : " + str(ae.mean()))
    print("-----------------------------------------")
    ###############################################################
    ###############################################################
    
    # Combine predictions
    res = pd.DataFrame()
    res['pred'] = pred_KR + pred_UR
    res['true'] = true_KR + true_UR
    res['mse'] = (res['pred']-res['true'])**2
       
    for col in y_test_KR.columns[1::]:
        res[col] = list(y_test_KR.loc[:,col])+list(y_test_UR.loc[:,col])
    
    res['cut'] = pd.qcut(res['AmountObs'], 9, labels=None, retbins=False, precision=3, duplicates='drop')
    res.sort_values(by= ['edge'], inplace = True)
    res.reset_index(drop=True, inplace= True)
    print(res)

    ################################################################################
    ######################## EXAMINE THE SPEED DISTRIBUTION ########################
    ################################################################################
    if EXAMINE_SPEED_DISTRIBUTION:
        fig = plt.Figure()

        bins = np.linspace(0, 65, 30)
        # Histograms
        counts_2,bins,bars = plt.hist(res.true, bins = bins, alpha = 0.5)
        counts_1,bins,bars = plt.hist(res.pred, bins = bins, alpha = 0.5)
        print(counts_1)
        print(counts_2)
        print(bins)
        plt.legend(['Observed', 'Predicted'])
        plt.title("Speed predictions versus observations")
        plt.xlabel('Velocity')
        plt.ylabel('Frequency')
        
        # bins = [(y + x)/2 for x, y in zip(bins, bins[1:])]
        # print(bins)
        
        # Save data (for use in tikz)
        pd.DataFrame({'bins': bins[1::], 'pred':list(counts_1[1::])+[0],'true': counts_2}).to_csv("Results/"+CASENAME+"/PredTrue_DATA_0627.csv", sep = ";")
    
        plt.show()
    ################################################################################
    ################################################################################
    
    ################################################################################
    ###################### EXAMINE THE AMOUNT OF OBSERVATIONS ######################
    ################################################################################
    if EXAMINE_NUM_OBSERVATIONS:
        res.sort_values(by= ['cut'], inplace = True)
        tes = res.groupby(by = 'cut').agg({'AmountObs': 'count', 'mse': 'mean'})
        print(tes)
        
        # Get the last 4 rows of the DataFrame
        last_four_rows = tes.tail(4)

        # Calculate the weighted average
        weighted_average = (last_four_rows['mse'] * last_four_rows['AmountObs']).sum() / sum(last_four_rows['AmountObs'])
        print("Weighted Average:", weighted_average)
        
        
        randomDists = res['cut'].unique()
        numDists = len(randomDists)

        data = []
        for i in randomDists:
            datatemp = list(res.loc[res.cut == i, 'mse'])
            data += [datatemp]
      
        fig, ax1 = plt.subplots(figsize=(10, 6))

        fig.canvas.set_window_title('\# Observations Analysis')
        plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

        bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
        
        plt.plot([None] +list(tes.mse*8), color = 'blue')
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='red', marker='+')

        # Add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    
        # Hide these grid behind plot objects
        ax1.set_axisbelow(True)
        ax1.set_title('Comparison of MSEs across different amounts of observations.')
        ax1.set_xlabel('Amount of observations')
        ax1.set_ylabel('MSE')

        # Now fill the boxes with desired colors
        boxColors = ['darkkhaki', 'royalblue']
        numBoxes = numDists
        medians = list(range(numBoxes))

        # Set the axes ranges and axes labels
        ax1.set_xlim(0.5, numBoxes + 0.5)
        ax1.set_ylim(-1, 1000)
        
        def divide10(x, factor=8):
            return x/factor
        def times10(x, factor=8):
            return x*factor
        ax1.secondary_yaxis('right', functions = (divide10, times10), color='blue')

        xtickNames = plt.setp(ax1, xticklabels=randomDists)
        plt.setp(xtickNames, rotation=45, fontsize=8)
        plt.show()
    ################################################################################
    ################################################################################

    ################################################################################
    ##################### CREATE PREDICTIONS FOR A ROAD NETWORK ####################
    ################################################################################
    if CREATE_ROAD_NETWORK_PREDICTION:
        #FULL_SET = True # If True: Requirement df_roads_FULL

        if FULL_SET: 
            # Load the full set of data (FULL SET)
            #with open("Input data/CustomData/df_roads_1106_FULL.pickle", "rb") as input_file: df = pickle.load(input_file)#0817
            #print(df)
            
            #from ast import literal_eval
            def image_converter(image_list_str):
                if image_list_str == "-": return image_list_str
                else:
                    image_list = eval(image_list_str)
                    return [np.array(image).astype(int) for image in image_list]
            df = pd.read_csv(PATH_TO_DF_FULL, sep = ";", index_col = 0, converters={'image': image_converter})
            # print(res.image)
            # print(res.image[0][0])
            # print(df.image)
            # print(df.image[0][0])
            # print(np.vstack([tuple(df.image[0:5].values)]).astype("float32")/255)
            # stop
            
            #df = pd.read_csv(PATH_TO_DF_FULL, sep = ";")
            #df.image = df.image.apply(lambda x: np.array(literal_eval(x)))
            
            # df.image_5 = df.image_5.apply(lambda x: x[0].tolist())
            # df = df[['u','v','key', 'image_5', 'midlon', 'midlat']]
            # df.to_csv("Input data/CustomData/df_roads_FULL.csv", sep = ";", index = False)
            #df.to_csv("df_roads_FULL_2.csv", index = False)
            
      
            # Add some relevant data
            df['edge'] = "(" + df['u'].map(str) + ',' + df['v'].map(str) + ',' + df['key'].map(str) + ")"
            df_test_all = df.loc[:,["edge","midlon","midlat",'image','vegetation_percentage','swir_min', 'swir_max', 'swir_mean']]
            #df_test_all.dropna(subset = ["image_5"], inplace=True)
            #df_test_all['image_5'] = [i[0] for i in df_test_all.image_5]
        else: 
            # Define all roads situations (TEST SET) (only consider roads that occur in the test set)

            df_test_all = res #pd.DataFrame()
            # df_test_all['edge'] = res.edge
            #             df_test_all['image'] = res.image
            #             df_test_all['midlon'] = res.midlon
            #             df_test_all['midlat'] = res.midlat
            df_test_all = df_test_all.groupby('edge').agg({'edge':'first','image': 'first', 'midlon': 'first', 'midlat':'first'}).reset_index(drop=True)
            df_test_all.sort_values(by=['edge'], inplace = True)
            
        amount_cats= []

        for e in ENV_FEATURES:
            amount_cats.append(len(res[e].unique()))
        amount_envs = np.prod(amount_cats)

        
        # Add dark/light and rain classes
        colnames = df_test_all.columns
        df_test_all = pd.DataFrame(np.repeat(df_test_all.values, amount_envs, axis=0))
        df_test_all.columns = colnames
        
        
        test = [range(ac) for ac in amount_cats]
        

        def generate_unique_combinations(lists):
            combinations = list(itertools.product(*lists))
            separated_lists = [[] for _ in range(len(lists))]
            for combination in combinations:
                for i, element in enumerate(combination):
                    separated_lists[i].append(element)
            return separated_lists
        
        result = generate_unique_combinations(test)
        
        ind = 0
        for e in ENV_FEATURES:
            df_test_all[e] = result[ind]*int(len(df_test_all)/amount_envs)
            ind+=1
        
        df_test_all=df_test_all.loc[:, ['edge'] + ENV_FEATURES + INFO_FEATURES + IMAGE_FEATURE]
        
        # Note that this only works if we have an uneven amount of rain classes, otherwise duplicate rows
        # ind = 0
        # for e in ENV_FEATURES:
        #     ac = amount_cats[0]
        #     df_test_all[e] = list(range(0,ac)) * int(len(df_test_all)/ac)
        #     ind+=1
        # df_test_all[ENV_FEATURES[1]] = [0,1]*number_rain_categories*int((len(df_test_all)/(2*number_rain_categories)))
        # df_test_all[ENV_FEATURES[0]] = [item for sublist in [[i]*2 for i in range(number_rain_categories)] for item in sublist]*int((len(df_test_all)/(2*number_rain_categories)))
        
        # Define the feature variables.
        print(df_test_all)

        # x_test_all_info = df_test_all.loc[:,["midlon","midlat"]+ENV_FEATURES]
        # x_test_all_info = df_test_all.loc[:,INFO_FEATURES]
        x_test_all_info = df_test_all.loc[:,~df_test_all.columns.isin(['edge','image'])]
        
        # print("-----------------------------------------------------------------")
        # print(x_test_all_info)
        # print("-----------------------------------------------------------------")
        
        x_test_all_info = scaler.transform(x_test_all_info).astype("float32")
        
        
        # print(df_test_all)
        # print(np.vstack([tuple(df_test_all.image.values)]).astype("float32")/255)
        x_test_all_image = np.vstack([tuple(df_test_all.image.values)]).astype("float32")/255
        
        preds_all = model.predict([x_test_all_image, x_test_all_info])
        preds_all = [item for sublist in preds_all for item in sublist]
        
        # Make the predictions
        df_test_all['Prediction'] = preds_all
        
        df_test_all.to_csv('Results/'+CASENAME+'/df_test_all_'+CHECK_MODEL+'.csv', sep = ";")
        print(df_test_all)
    ################################################################################
    ################################################################################
    
    # If CREATE_ROAD_NETWORK_PREDICTION is already ran
    try: 
        df_test_all = pd.read_table("Results/"+CASENAME+"/df_test_all_"+CHECK_MODEL+".csv", sep = ";", index_col = 0)
    except: 
        print('<<<<<< df_test_all.csv is not (yet?) created. First, run CREATE_ROAD_NETWORK_PREDICTION! >>>>>>')
        return
    
    ################################################################################
    ####################### CREATE HEATMAPS PER SPEED CATEGORY #####################
    ################################################################################
    if CREATE_HEATMAP_SPEED:
        if CUSTOM_SPEED_CATS: 
            df_test_all['cat'] = None
            ind = 0
            for i in df_test_all['Prediction']:
                cat = 0
                for c in SPEED_CATS+[1000]:
                    if i < c: 
                        df_test_all.loc[ind,'cat'] = cat
                        break
                    cat+=1
                ind+=1
        else:
            df_test_all['cat'] = pd.cut(df_test_all['Prediction'], SPEED_CATS, labels=None, retbins=False, precision=3, duplicates='drop')
        
        edges_stds = df_test_all.groupby(by = ['edge']).agg({'Prediction': 'std'})
        df_test_all = df_test_all.merge(edges_stds, on='edge', suffixes=('', '_std'))
        
        print(np.min(df_test_all['Prediction_std']))
        print(np.max(df_test_all['Prediction_std']))
        df_test_all['cat_var'] = pd.cut(df_test_all['Prediction_std'], [0, 1.5, 2.5, 10], labels=None, retbins=False, precision=3, duplicates='drop')
        print(df_test_all)

        ######### Determine vulnerable edges ##########
        t = df_test_all.groupby(by =["edge"]).agg({'Prediction': [('MinV', 'min'), ('MaxV', 'max')]}).reset_index(drop=False)
        t.columns = t.columns.droplevel(0)
        t['Relative diff'] = t['MaxV'] - t['MinV']
        t['Percentage diff'] = ((t['MaxV'] - t['MinV'])/t['MinV'])*100
        t.columns =['edge', 'MinV', 'MaxV', 'Relative diff', 'Percentage diff']
        
        vulnerable_edges = list(t[t['Percentage diff']>-1].edge)
        df_test_all_vulnerable = df_test_all[df_test_all.edge.isin(vulnerable_edges)]

        heatmap_var = 'cat_var' #cat
        t = pd.DataFrame(df_test_all_vulnerable.groupby(by =[heatmap_var]+ENV_FEATURES[0:2]).agg({'Prediction': 'mean'})).reset_index(drop = False)
        ###############################################
        
        # # Fill in any missing elements
        # for c in df_test_all['cat'].unique():
        #     for r in df_test_all[ENV_FEATURES[0]].unique():
        #         for d in df_test_all[ENV_FEATURES[1]].unique():
        #             relevant_row = t[(t.cat==c) & (t[ENV_FEATURES[1]] ==d) & (t[ENV_FEATURES[0]] == r)]
        #             if len(relevant_row)==0:
        #                 t.loc[len(t.index)] = [c,r,d,0]
        
        t.sort_values(by=[heatmap_var, ENV_FEATURES[1], ENV_FEATURES[0]], inplace = True)
        t.reset_index(drop=True, inplace = True)
        print(t)
          
        # Plot the categories in a heatmap
        if CUSTOM_SPEED_CATS: num_plots = len(SPEED_CATS)+1
        else: num_plots = SPEED_CATS
        
        rain_cats = t[heatmap_var].unique()
        titles = ['Low std', 'Moderate std', 'High std']
        #titles = ['Low speed', 'Moderate speed', 'High speed']
        # Show subplots | shape: (1,3) 
        
        rc = 0
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
        fig.tight_layout(pad=1.0)
        for i, ax in enumerate(axs.flatten()):
            plt.sca(ax)
            a = np.array(t[t[heatmap_var] == rain_cats[rc]].loc[:,'Prediction']).reshape((2,3))
            im = plt.imshow(a, cmap='autumn')
            if rc ==0: ax.set_yticks(np.arange(len(t[ENV_FEATURES[1]].unique())), labels=['Dark', 'Light']) 
            else: ax.set_yticks([]) 
            ax.set_xticks(np.arange(len(t[ENV_FEATURES[0]].unique())), labels=['No/Light', 'Moderate', 'Heavy'])
            #plt.colorbar()
            plt.title(titles[i])
            ax.figure.colorbar(im,  ax = ax, shrink=0.5)
            rc+=1
        plt.show()
    ################################################################################
    ################################################################################
    
    ################################################################################
    ####################### EXAMINE THE SPREAD OF VELOCITIES #######################
    ################################################################################
    if EXAMINE_PREDICTION_SPREAD:
        """ REQUIREMENT: CREATE_ROAD_NETWORK_PREDICTION """
        # Examine the spread of TRUE speed obsrevations
        t = res.groupby(by =["edge"]).agg({'true': [('MinV', 'min'), ('MaxV', 'max')]}).reset_index(drop=True)
        t.columns = t.columns.droplevel(0)
        t['Relative diff'] = t['MaxV'] - t['MinV']
        t['Percentage diff'] = ((t['MaxV'] - t['MinV'])/t['MinV'])*100
        t.sort_values(by='Relative diff', inplace = True)
        print(t)
        
        t = df_test_all.groupby(by =["edge"]).agg({'Prediction': [('MinV', 'min'), ('MaxV', 'max')]}).reset_index(drop=True)
        t.columns = t.columns.droplevel(0)
        t['Relative diff'] = t['MaxV'] - t['MinV']
        t['Percentage diff'] = ((t['MaxV'] - t['MinV'])/t['MinV'])*100
        t.sort_values(by='Relative diff', inplace = True)
        print(t)
        t.sort_values(by='Percentage diff', inplace = True)
        print(t)
        
        fig = plt.Figure()
        bins = np.linspace(0, 50, 20)
        # Histograms
        counts_2,bins,bars = plt.hist(t['Percentage diff'], bins = bins, alpha = 0.5)
        
        print(counts_2)
        print(bins[1::])
        
        #plt.legend(['Observed'])
        plt.title("Percentage diffs")
        plt.xlabel('Percentage diff')
        plt.ylabel('Frequency')
        
        # Save data (for use in tikz)
        #pd.DataFrame({'pred':t['Percentage diff']}).to_csv("Pred_DATA_2.csv", sep = ";")
        pd.DataFrame({'bins':[0] + list(bins[1::]), 'pred': list(counts_2) + [0]}).to_csv("Results/"+CASENAME+"/Pred_DATA_2.csv", sep = ";")
        plt.show()
    ################################################################################
    ################################################################################
    
    ################################################################################
    ################### CREATE NEW SHAPEFILES (INCL SPEED INFO) ####################
    ################################################################################
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
    if CREATE_ENHANCED_ROAD_NETWORK: 
        """ REQUIREMENT: CREATE_ROAD_NETWORK_PREDICTION """
        
        from shapely import wkt
        import geopandas
        def csvs2graph(path_nodes, path_edges, project = True):
            edges = pd.read_csv(path_edges, sep=";", index_col = 0, low_memory = False) #PEMPEM: 14960, OE: 1100, EO: 8000
            nodes = pd.read_csv(path_nodes, sep=";", index_col = 0, low_memory = False) #PEMPEM: 14960, OE: 1100, EO: 8000
            # print(edges.driven.unique())

            edges['highway'] = edges['highway'].replace(np.nan, "None")
    
            nodes['geometry'] = nodes['geometry'].apply(wkt.loads)
            edges['geometry'] = edges['geometry'].apply(wkt.loads)
        
            gdf_nodes = geopandas.GeoDataFrame(nodes, geometry=nodes.geometry, crs='epsg:4326')
            gdf_edges = geopandas.GeoDataFrame(edges, geometry=edges.geometry, crs='epsg:4326')
            gdf_edges.set_index(['u', 'v', 'key'], inplace=True)
            
            # Selection of attributes
            #gdf_nodes = gdf_nodes[['x','y', 'geometry']]
            #gdf_edges = gdf_edges[['geometry', 'u', 'v', 'key', 'length']]
            G = ox.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs = {'crs': 'epsg:4326', 'simplified': True})
            if project: G = ox.projection.project_graph(G)# to_crs="+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +type=crs"
            return G
    
        """ ------------------  Load graph  ------------------ """
        # PATH_TO_GRAPH = "Input data/Data/_Graphs_SIMPLE/"
        G = csvs2graph(path_nodes = PATH_TO_GRAPH+"Nodes.csv", path_edges = PATH_TO_GRAPH+"Edges.csv", project = False)
        used_crs = "+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +type=crs"
        """ ------------------------------------------------- """

        envs = []
        for r in df_test_all[ENV_FEATURES[0]].unique():
            for d in df_test_all[ENV_FEATURES[1]].unique():
                envs+= ['Velocity_'+str(r)+str(d)]
                nx.set_edge_attributes(G, None, name = 'Velocity_'+str(r)+str(d))

        nx.set_edge_attributes(G, None, name = 'Average velocity')
        
        from progress.bar import Bar

        bar = Bar('Processing', max=len(G.edges))
        
        i=1
        
        for e in G.edges(data = 'geometry', keys = True):
            edge_string = "("+str(e[0])+","+str(e[1])+","+str(e[2])+")"
            predictions = list(df_test_all.loc[df_test_all.edge==edge_string, 'Prediction'])
            for s, p in zip(envs, predictions):
                G.edges[e[0], e[1], e[2]][s] = p
            try:
                G.edges[e[0], e[1], e[2]]['Average velocity'] = sum(predictions)/len(predictions)
            except: G.edges[e[0], e[1], e[2]]['Average velocity'] = None
            bar.next()
            
        nodes, edges = ox.graph_to_gdfs(G)
        nodes = nodes.reset_index(drop = True)
        print(nodes)
        print(edges)
        
        if not os.path.exists("Results/"+CASENAME+"/_Graph_SIMPLE_ENHANCED"): os.makedirs("Results/"+CASENAME+"/_Graph_SIMPLE_ENHANCED")
        nodes.to_csv("Results/"+CASENAME+"/_Graph_SIMPLE_ENHANCED/Nodes_ENHANCED.csv", sep = ";")
        edges.to_csv("Results/"+CASENAME+"/_Graph_SIMPLE_ENHANCED/Edges_ENHANCED.csv", sep = ";")
        
        if not os.path.exists("Results/"+CASENAME+"/_Shapefiles"): os.makedirs("Results/"+CASENAME+"/_Shapefiles")
        create_shapefile(nodes, "Results/"+CASENAME+"/_Shapefiles/SF_nodes_FULL.shp")
        create_shapefile(edges, "Results/"+CASENAME+"/_Shapefiles/SF_edges_FULL.shp")
    ################################################################################
    ################################################################################

def testModel_SAT(model, df_test_KR, df_test_UR, scaler):
    ###############################################################
    #################### EVALUATE PERFORMANCE #####################
    ###############################################################
    # Test data
    x_test_info_KR, x_test_image_KR, y_test_KR = df_test_KR
    x_test_info_UR, x_test_image_UR, y_test_UR = df_test_UR
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # # # # #  KNOWN ROADS  # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Make predictions on the testing data and define the true speeds as a list
    preds_KR = model.predict(x_test_image_KR)
    pred_KR = [item for sublist in preds_KR for item in sublist]
    true_KR = list(y_test_KR.iloc[:,0])

    # Performance metrics
    se = (np.asarray(true_KR) - np.asarray(pred_KR))**2 # Squared errors (SE)
    ae = abs(np.asarray(true_KR) - np.asarray(pred_KR)) # Absolute errors (AE)
    print("-------------- Known roads --------------")
    print("Mean squared error (MSE): " + str(se.mean())) # or use mean_squared_error(true,pred)
    print("Root mean squared error (RMSE): " + str(sqrt(se.mean()))) # or use mean_squared_error(true,pred)
    print("Mean absolute error : " + str(ae.mean()))
    print("-----------------------------------------")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # #  UNKNOWN ROADS  # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Make predictions on the testing data and define the true speeds as a list
    preds_UR = model.predict(x_test_image_UR)
    pred_UR = [item for sublist in preds_UR for item in sublist]
    true_UR = list(y_test_UR.iloc[:,0])

    # Performance metrics
    se = (np.asarray(true_UR) - np.asarray(pred_UR))**2 # Squared errors (SE)
    ae = abs(np.asarray(true_UR) - np.asarray(pred_UR)) # Absolute errors (AE)
    print("------------- Unknown roads -------------")
    print("Mean squared error (MSE): " + str(se.mean())) # or use mean_squared_error(true,pred)
    print("Root mean squared error (RMSE): " + str(sqrt(se.mean()))) # or use mean_squared_error(true,pred)
    print("Mean absolute error : " + str(ae.mean()))
    print("-----------------------------------------")
    ###############################################################
    ###############################################################
    
    # Combine predictions
    res = pd.DataFrame()
    res['pred'] = pred_KR + pred_UR
    res['true'] = true_KR + true_UR
    res['mse'] = (res['pred']-res['true'])**2
       
    for col in y_test_KR.columns[1::]:
        res[col] = list(y_test_KR.loc[:,col])+list(y_test_UR.loc[:,col])
    
    res['cut'] = pd.qcut(res['AmountObs'], 9, labels=None, retbins=False, precision=3, duplicates='drop')
    res.sort_values(by= ['edge'], inplace = True)
    res.reset_index(drop=True, inplace= True)
    print(res)
    
    ################################################################################
    ######################## EXAMINE THE SPEED DISTRIBUTION ########################
    ################################################################################
    if EXAMINE_SPEED_DISTRIBUTION:
        fig = plt.Figure()

        bins = np.linspace(0, 65, 30)
        # Histograms
        counts_2,bins,bars = plt.hist(res.true, bins = bins, alpha = 0.5)
        counts_1,bins,bars = plt.hist(res.pred, bins = bins, alpha = 0.5)
        print(counts_1)
        print(counts_2)
        print(bins)
        plt.legend(['Observed', 'Predicted'])
        plt.title("Speed predictions versus observations")
        plt.xlabel('Velocity')
        plt.ylabel('Frequency')
        
        # Save data (for use in tikz)
        pd.DataFrame({'bins': bins[1::], 'pred':counts_1,'true': counts_2}).to_csv("Results/"+CASENAME+"/PredTrue_DATA_1219.csv", sep = ";")
    
        plt.show()
    ################################################################################
    ################################################################################
    
    ################################################################################
    ###################### EXAMINE THE AMOUNT OF OBSERVATIONS ######################
    ################################################################################
    if EXAMINE_NUM_OBSERVATIONS:
        res.sort_values(by= ['cut'], inplace = True)
        tes = res.groupby(by = 'cut').agg({'AmountObs': 'count', 'mse': 'mean'})
    
        randomDists = res['cut'].unique()
        numDists = len(randomDists)

        data = []
        for i in randomDists:
            datatemp = list(res.loc[res.cut == i, 'mse'])
            data += [datatemp]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        fig.canvas.set_window_title('\# Observations Analysis')
        plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

        bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
        
        plt.plot([None] +list(tes.mse*8), color = 'blue')
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['fliers'], color='red', marker='+')

        # Add a horizontal grid to the plot, but make it very light in color
        # so we can use it for reading data values but not be distracting
        ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    
        # Hide these grid behind plot objects
        ax1.set_axisbelow(True)
        ax1.set_title('Comparison of MSEs across different amounts of observations.')
        ax1.set_xlabel('Amount of observations')
        ax1.set_ylabel('MSE')

        # Now fill the boxes with desired colors
        boxColors = ['darkkhaki', 'royalblue']
        numBoxes = numDists
        medians = list(range(numBoxes))

        # Set the axes ranges and axes labels
        ax1.set_xlim(0.5, numBoxes + 0.5)
        ax1.set_ylim(-1, 1000)
        
        def divide10(x, factor=8):
            return x/factor
        def times10(x, factor=8):
            return x*factor
        ax1.secondary_yaxis('right', functions = (divide10, times10), color='blue')

        xtickNames = plt.setp(ax1, xticklabels=randomDists)
        plt.setp(xtickNames, rotation=45, fontsize=8)
        plt.show()
    ################################################################################
    ################################################################################

    ################################################################################
    ##################### CREATE PREDICTIONS FOR A ROAD NETWORK ####################
    ################################################################################
    if CREATE_ROAD_NETWORK_PREDICTION:
        #FULL_SET = True # If True: Requirement df_roads_FULL

        if FULL_SET: 
            # Load the full set of data (FULL SET)
            #with open("Input data/CustomData/df_roads_1106_FULL.pickle", "rb") as input_file: df = pickle.load(input_file)#0817
            #print(df)
            
            #from ast import literal_eval
            def image_converter(image_list_str):
                if image_list_str == "-": return image_list_str
                else:
                    image_list = eval(image_list_str)
                    return [np.array(image).astype(int) for image in image_list]
            df = pd.read_csv(PATH_TO_DF_FULL, sep = ";", index_col = 0, converters={'image': image_converter})
            # print(res.image)
            # print(res.image[0][0])
            # print(df.image)
            # print(df.image[0][0])
            # print(np.vstack([tuple(df.image[0:5].values)]).astype("float32")/255)
            # stop
            
            #df = pd.read_csv(PATH_TO_DF_FULL, sep = ";")
            #df.image = df.image.apply(lambda x: np.array(literal_eval(x)))
            
            # df.image_5 = df.image_5.apply(lambda x: x[0].tolist())
            # df = df[['u','v','key', 'image_5', 'midlon', 'midlat']]
            # df.to_csv("Input data/CustomData/df_roads_FULL.csv", sep = ";", index = False)
            #df.to_csv("df_roads_FULL_2.csv", index = False)
            
       
            print("HOOOOOLLO")
            # Add some relevant data
            df['edge'] = "(" + df['u'].map(str) + ',' + df['v'].map(str) + ',' + df['key'].map(str) + ")"
            df_test_all = df.loc[:,["edge","midlon","midlat",'image','vegetation_percentage','swir_min', 'swir_max', 'swir_mean']]
            #df_test_all.dropna(subset = ["image_5"], inplace=True)
            #df_test_all['image_5'] = [i[0] for i in df_test_all.image_5]
        else: 
            # Define all roads situations (TEST SET) (only consider roads that occur in the test set)

            df_test_all = res #pd.DataFrame()
            # df_test_all['edge'] = res.edge
            #             df_test_all['image'] = res.image
            #             df_test_all['midlon'] = res.midlon
            #             df_test_all['midlat'] = res.midlat
            df_test_all = df_test_all.groupby('edge').agg({'edge':'first','image': 'first', 'midlon': 'first', 'midlat':'first'}).reset_index(drop=True)
            df_test_all.sort_values(by=['edge'], inplace = True)
            
        amount_cats= []

        for e in ENV_FEATURES:
            amount_cats.append(len(res[e].unique()))
        amount_envs = np.prod(amount_cats)
        

        
        # Add dark/light and rain classes
        colnames = df_test_all.columns
        df_test_all = pd.DataFrame(np.repeat(df_test_all.values, amount_envs, axis=0))
        df_test_all.columns = colnames
        

        
        test = [range(ac) for ac in amount_cats]
        

        def generate_unique_combinations(lists):
            combinations = list(itertools.product(*lists))
            separated_lists = [[] for _ in range(len(lists))]
            for combination in combinations:
                for i, element in enumerate(combination):
                    separated_lists[i].append(element)
            return separated_lists
        
        result = generate_unique_combinations(test)
        
        ind = 0
        for e in ENV_FEATURES:
            df_test_all[e] = result[ind]*int(len(df_test_all)/amount_envs)
            ind+=1


        
        
        
        # Note that this only works if we have an uneven amount of rain classes, otherwise duplicate rows
        # ind = 0
        # for e in ENV_FEATURES:
        #     ac = amount_cats[0]
        #     df_test_all[e] = list(range(0,ac)) * int(len(df_test_all)/ac)
        #     ind+=1
        # df_test_all[ENV_FEATURES[1]] = [0,1]*number_rain_categories*int((len(df_test_all)/(2*number_rain_categories)))
        # df_test_all[ENV_FEATURES[0]] = [item for sublist in [[i]*2 for i in range(number_rain_categories)] for item in sublist]*int((len(df_test_all)/(2*number_rain_categories)))
        
        # Define the feature variables.
        print(df_test_all)

        # x_test_all_info = df_test_all.loc[:,["midlon","midlat"]+ENV_FEATURES]
        x_test_all_info = df_test_all.loc[:,INFO_FEATURES]
        
        print(x_test_all_info)
        
        x_test_all_info = scaler.transform(x_test_all_info).astype("float32")
        # print(df_test_all)
        # print(np.vstack([tuple(df_test_all.image.values)]).astype("float32")/255)
        x_test_all_image = np.vstack([tuple(df_test_all.image.values)]).astype("float32")/255
        print("HApLO")
        preds_all = model.predict([x_test_all_image, x_test_all_info])
        preds_all = [item for sublist in preds_all for item in sublist]
        
        # Make the predictions
        df_test_all['Prediction'] = preds_all
        df_test_all.to_csv('Results/'+CASENAME+'/df_test_all.csv', sep = ";")
        print(df_test_all)
    ################################################################################
    ################################################################################
    
    # If CREATE_ROAD_NETWORK_PREDICTION is already ran
    try: 
        df_test_all = pd.read_table("Results/"+CASENAME+"/df_test_all.csv", sep = ";", index_col = 0)
    except: 
        print('<<<<<< df_test_all.csv is not (yet?) created. First, run CREATE_ROAD_NETWORK_PREDICTION! >>>>>>')
        return
    
    ################################################################################
    ####################### CREATE HEATMAPS PER SPEED CATEGORY #####################
    ################################################################################
    if CREATE_HEATMAP_SPEED:
        
        # """----------- Settings -----------"""
        # CUSTOM_SPEED_CATS = False
        # if CUSTOM_SPEED_CATS:
        #     SPEED_CATS = [15,30]
        # if not CUSTOM_SPEED_CATS:
        #     SPEED_CATS = 3
        # """--------------------------------"""
        # df_test_all = res
        # df_test_all['Prediction'] = res.true
        if CUSTOM_SPEED_CATS: 
            df_test_all['cat'] = None
            ind = 0
            for i in df_test_all['Prediction']:
                cat = 0
                for c in SPEED_CATS+[1000]:
                    if i < c: 
                        df_test_all.loc[ind,'cat'] = cat
                        break
                    cat+=1
                ind+=1
        else:
            df_test_all['cat'] = pd.cut(df_test_all['Prediction'], SPEED_CATS, labels=None, retbins=False, precision=3, duplicates='drop')
        
        edges_stds = df_test_all.groupby(by = ['edge']).agg({'Prediction': 'std'})
        df_test_all = df_test_all.merge(edges_stds, on='edge', suffixes=('', '_std'))
        df_test_all['cat_var'] = pd.cut(df_test_all['Prediction_std'], [0, 1.5, 2.5, 4], labels=None, retbins=False, precision=3, duplicates='drop')
        print(df_test_all)
        
        
        
       

        #print(df_test_all.groupby(by=['cat']+ENV_FEATURES).agg({'cat':'count'}))

        ######### Determine vulnerable edges ##########
        t = df_test_all.groupby(by =["edge"]).agg({'Prediction': [('MinV', 'min'), ('MaxV', 'max')]}).reset_index(drop=False)
        t.columns = t.columns.droplevel(0)
        t['Relative diff'] = t['MaxV'] - t['MinV']
        t['Percentage diff'] = ((t['MaxV'] - t['MinV'])/t['MinV'])*100
        t.columns =['edge', 'MinV', 'MaxV', 'Relative diff', 'Percentage diff']
        
        vulnerable_edges = list(t[t['Percentage diff']>-1].edge)
        df_test_all_vulnerable = df_test_all[df_test_all.edge.isin(vulnerable_edges)]

        heatmap_var = 'cat_var' #cat
        t = pd.DataFrame(df_test_all_vulnerable.groupby(by =[heatmap_var]+ENV_FEATURES[0:2]).agg({'Prediction': 'mean'})).reset_index(drop = False)
        ###############################################
        
        
        
        
        # # Fill in any missing elements
        # for c in df_test_all['cat'].unique():
        #     for r in df_test_all[ENV_FEATURES[0]].unique():
        #         for d in df_test_all[ENV_FEATURES[1]].unique():
        #             relevant_row = t[(t.cat==c) & (t[ENV_FEATURES[1]] ==d) & (t[ENV_FEATURES[0]] == r)]
        #             if len(relevant_row)==0:
        #                 t.loc[len(t.index)] = [c,r,d,0]
        
        t.sort_values(by=[heatmap_var, ENV_FEATURES[1], ENV_FEATURES[0]], inplace = True)
        t.reset_index(drop=True, inplace = True)
        print(t)
        

                        
        # Plot the categories in a heatmap
        if CUSTOM_SPEED_CATS: num_plots = len(SPEED_CATS)+1
        else: num_plots = SPEED_CATS
        
        rain_cats = t[heatmap_var].unique()
        titles = ['Low speed', 'Moderate speed', 'High speed']
        # Show subplots | shape: (1,3) 
        
        rc = 0
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
        fig.tight_layout(pad=1.0)
        for i, ax in enumerate(axs.flatten()):
            
            plt.sca(ax)
            a = np.array(t[t[heatmap_var] == rain_cats[rc]].loc[:,'Prediction']).reshape((2,3))
            im = plt.imshow(a, cmap='autumn')
            if rc ==0: ax.set_yticks(np.arange(len(t[ENV_FEATURES[1]].unique())), labels=['Light', 'Dark']) 
            else: ax.set_yticks([]) 
            ax.set_xticks(np.arange(len(t[ENV_FEATURES[0]].unique())), labels=['No/Light', 'Moderate', 'Heavy'])
            #plt.colorbar()
            plt.title(titles[i])
            ax.figure.colorbar(im,  ax = ax, shrink=0.5)
            rc+=1

        #plt.tight_layout()
        plt.show()
        
        
        
        
        # fig, ax = plt.subplots(num_plots)
        #
        # df_test_all_vulnerable
        # p = 0
        # to = 131
        #
        # for i in t['cat'].unique():
        #     a = np.array(t[t.cat == i].loc[:,'Prediction']).reshape((2,3))
        #     if len(a[a>0])>0: min_a = np.min(a[a>0])
        #     else: min_a = 0
        #     im = ax[p].imshow(a, cmap = 'autumn' , interpolation = 'nearest', vmin=min_a, vmax=np.max(a))
        #
        #     ax[p].set_yticks(np.arange(len(t[ENV_FEATURES[1]].unique())), labels=['Light', 'Dark'])
        #
        #
        #     if p == (num_plots-1):
        #         ax[p].set_xticks(np.arange(len(t[ENV_FEATURES[0]].unique())), labels=['No/Light', 'Moderate', 'Heavy'])
        #
        #     ax[p].figure.colorbar(im,  ax = ax[p], shrink=0.3)
        #     p+=1
        #     to+=1
        #
        # plt.show()
    ################################################################################
    ################################################################################
    
    ################################################################################
    ####################### EXAMINE THE SPREAD OF VELOCITIES #######################
    ################################################################################
    if EXAMINE_PREDICTION_SPREAD:
        """ REQUIREMENT: CREATE_ROAD_NETWORK_PREDICTION """
        # Examine the spread of TRUE speed obsrevations
        t = res.groupby(by =["edge"]).agg({'true': [('MinV', 'min'), ('MaxV', 'max')]}).reset_index(drop=True)
        t.columns = t.columns.droplevel(0)
        t['Relative diff'] = t['MaxV'] - t['MinV']
        t['Percentage diff'] = ((t['MaxV'] - t['MinV'])/t['MinV'])*100
        t.sort_values(by='Relative diff', inplace = True)
        print(t)
        
        t = df_test_all.groupby(by =["edge"]).agg({'Prediction': [('MinV', 'min'), ('MaxV', 'max')]}).reset_index(drop=True)
        t.columns = t.columns.droplevel(0)
        t['Relative diff'] = t['MaxV'] - t['MinV']
        t['Percentage diff'] = ((t['MaxV'] - t['MinV'])/t['MinV'])*100
        t.sort_values(by='Relative diff', inplace = True)
        print(t)
        t.sort_values(by='Percentage diff', inplace = True)
        print(t)
        
        fig = plt.Figure()
        bins = np.linspace(0, 50, 20)
        # Histograms
        counts_2,bins,bars = plt.hist(t['Percentage diff'], bins = bins, alpha = 0.5)
        
        #plt.legend(['Observed'])
        plt.title("Percentage diffs")
        plt.xlabel('Percentage diff')
        plt.ylabel('Frequency')
        
        # Save data (for use in tikz)
        #pd.DataFrame({'pred':t['Percentage diff']}).to_csv("Pred_DATA_2.csv", sep = ";")
        pd.DataFrame({'bins':bins[1::], 'pred': counts_2}).to_csv("Results/"+CASENAME+"/Pred_DATA_2.csv", sep = ";")
        plt.show()
    ################################################################################
    ################################################################################
    
    ################################################################################
    ################### CREATE NEW SHAPEFILES (INCL SPEED INFO) ####################
    ################################################################################
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
    if CREATE_ENHANCED_ROAD_NETWORK: 
        """ REQUIREMENT: CREATE_ROAD_NETWORK_PREDICTION """
        
        from shapely import wkt
        import geopandas
        def csvs2graph(path_nodes, path_edges, project = True):
            edges = pd.read_csv(path_edges, sep=";", index_col = 0, low_memory = False) #PEMPEM: 14960, OE: 1100, EO: 8000
            nodes = pd.read_csv(path_nodes, sep=";", index_col = 0, low_memory = False) #PEMPEM: 14960, OE: 1100, EO: 8000
            # print(edges.driven.unique())

            edges['highway'] = edges['highway'].replace(np.nan, "None")
    
            nodes['geometry'] = nodes['geometry'].apply(wkt.loads)
            edges['geometry'] = edges['geometry'].apply(wkt.loads)
        
            gdf_nodes = geopandas.GeoDataFrame(nodes, geometry=nodes.geometry, crs='epsg:4326')
            gdf_edges = geopandas.GeoDataFrame(edges, geometry=edges.geometry, crs='epsg:4326')
            gdf_edges.set_index(['u', 'v', 'key'], inplace=True)
            
            # Selection of attributes
            #gdf_nodes = gdf_nodes[['x','y', 'geometry']]
            #gdf_edges = gdf_edges[['geometry', 'u', 'v', 'key', 'length']]
            G = ox.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs = {'crs': 'epsg:4326', 'simplified': True})
            if project: G = ox.projection.project_graph(G)# to_crs="+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +type=crs"
            return G
    
        """ ------------------  Load graph  ------------------ """
        # PATH_TO_GRAPH = "Input data/Data/_Graphs_SIMPLE/"
        G = csvs2graph(path_nodes = PATH_TO_GRAPH+"Nodes.csv", path_edges = PATH_TO_GRAPH+"Edges.csv", project = False)
        used_crs = "+proj=utm +zone=48 +ellps=WGS84 +datum=WGS84 +units=m +no_defs +type=crs"
        """ ------------------------------------------------- """
        #with open("Input data/Data/graph_0-14960.pickle", "rb") as input_file: G = pickle.load(input_file)
        
        # nodes, edges = ox.graph_to_gdfs(G)
        # print(nodes)
        # print(edges)
        envs = []
        for r in df_test_all[ENV_FEATURES[0]].unique():
            for d in df_test_all[ENV_FEATURES[1]].unique():
                envs+= ['Velocity_'+str(r)+str(d)]
                nx.set_edge_attributes(G, None, name = 'Velocity_'+str(r)+str(d))

        nx.set_edge_attributes(G, None, name = 'Average velocity')
        i=1
        for e in G.edges(data = 'geometry', keys = True):
            edge_string = "("+str(e[0])+","+str(e[1])+","+str(e[2])+")"
            predictions = list(df_test_all.loc[df_test_all.edge==edge_string, 'Prediction'])
            
            for s, p in zip(envs, predictions):
                G.edges[e[0], e[1], e[2]][s] = p
            
            try:
                G.edges[e[0], e[1], e[2]]['Average velocity'] = sum(predictions)/len(predictions)
            except: G.edges[e[0], e[1], e[2]]['Average velocity'] = None
            
            i+=1
            if i%1000 == 0: print("Processing edge:",i)
            
        nodes, edges = ox.graph_to_gdfs(G)
        nodes = nodes.reset_index(drop = True)
        print(nodes)
        print(edges)
        
        if not os.path.exists("Results/"+CASENAME+"/_Graph_SIMPLE_ENHANCED"): os.makedirs("Results/"+CASENAME+"/_Graph_SIMPLE_ENHANCED")
        nodes.to_csv("Results/"+CASENAME+"/_Graph_SIMPLE_ENHANCED/Nodes_ENHANCED.csv", sep = ";")
        edges.to_csv("Results/"+CASENAME+"/_Graph_SIMPLE_ENHANCED/Edges_ENHANCED.csv", sep = ";")
        
        if not os.path.exists("Results/"+CASENAME+"/_Shapefiles"): os.makedirs("Results/"+CASENAME+"/_Shapefiles")
        create_shapefile(nodes, "Results/"+CASENAME+"/_Shapefiles/SF_nodes_FULL.shp")
        create_shapefile(edges, "Results/"+CASENAME+"/_Shapefiles/SF_edges_FULL.shp")
    ################################################################################
    ################################################################################
