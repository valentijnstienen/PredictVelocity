# Date       : 2022-07-11
# Environment: conda activate SAT
# Location   : cd "Desktop/PEMPEM PAPER 2/Code/Input data"
# Run        : python additionalObs.py
import pandas as pd

###########################################################################
##################### PERCENTAGE ROADS WITH MAX_SPEED #####################
###########################################################################
# Load the data
def list_converter(l):
    try: return eval(l)
    except: return [l]
df = pd.read_table('Edges.csv', sep=";", index_col=0, low_memory=False, converters={'highway':list_converter})

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
used_road_types = ['trunk','primary','secondary','tertiary', 'unclassified', 'residential', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'] # Roads + Road links
EL = [i for i in range(len(df)) if (elementList(df.loc[i, 'highway'], used_road_types) | df.loc[i,'driven'])]
df_relevant = df.iloc[EL,:].reset_index(drop=True)
#print(df_relevant.DatesVelocities)
#print(df_relevant[~df_relevant.DatesVelocities.isnull()].groupby(['u', 'v', 'key']).agg({'u':'count'}))

a = sum(df_relevant.length)
print("Total length:", a)

# Define roads for which a max speed is given (altough may be unreliable)
filtered_df = df_relevant[df_relevant['maxspeed'].notnull()]
b = sum(filtered_df.length)

# Show for how many roads the max speed is known (incorporating speed)
print("Total length (known max_speed):", b)
print("Percentage known max_speed:", (b/a)*100)

# Show for how many roads the max speed is known (amount)
print("Total amount (known max_speed):", len(filtered_df))
print("Percentage known max_speed:", (len(filtered_df)/len(df_relevant))*100)
###########################################################################
###########################################################################
