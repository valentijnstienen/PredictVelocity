import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("RoutingResults/SP_diffs_CASE00_30_5_NEW_0716_BEST.csv", sep = ";", index_col = 0, low_memory = False)

# Drop rows for which we did not know the traveltime of edges in the orignal route (no comparison possible)
df = df[df['tt_00'] != "-"].reset_index(drop=True)
df = df[df['SP'] > 2500].reset_index(drop=True)

# Drop rows with 'inf' in Column2
df = df.loc[df.SP < float('inf'), :].reset_index(drop =True)
print(df)


cols = ['tt_00', 'tt_01', 'tt_10', 'tt_11', 'tt_20', 'tt_21', 'tt_AVG'] + ['tt_00_optimal', 'tt_01_optimal', 'tt_10_optimal', 'tt_11_optimal', 'tt_20_optimal', 'tt_21_optimal', 'tt_AVG_optimal']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
df = df.round(dict.fromkeys(cols, 5))

df_new = pd.DataFrame(columns = ['tt_00', 'tt_01', 'tt_10', 'tt_11', 'tt_20', 'tt_21', 'tt_AVG'])



for i in df_new.columns:
    df_new[i] = (pd.to_numeric(df[i]) - pd.to_numeric(df[i+'_optimal']))
    df_new[i+"_perc"] = (pd.to_numeric(df[i]) - pd.to_numeric(df[i+'_optimal']))/pd.to_numeric(df[i])*100

print(df_new)

#print("KONT")# df_new = df_new[(df_new > 1).any(axis=1)]
# df_new = df_new[df_new.tt_00>0]
df_new = df_new[df_new[df_new.loc[:,['tt_00', 'tt_01', 'tt_10', 'tt_11', 'tt_20', 'tt_21']]>0].any(axis=1)]
print(df_new)



#stop
# df_new = df_new[df_new[df_new[['tt_00', 'tt_01', 'tt_10', 'tt_11', 'tt_20', 'tt_21'] > 0].any(axis=1)]


df_new['test'] = df_new.tt_21 - df_new.tt_00
df_new['test_2'] = 100*(df_new.tt_21 - df_new.tt_00)/(df_new.tt_00)

df_new = df_new.sort_values(by = ['test']).reset_index(drop = True)
print(df_new)

stop
# print(list(df_new.test_2))
#
# stop


# plt.Figure()
# plt.hist(df_new.test, bins = 25)
# plt.show()
# stop


print(df_new)

####################################################################################
####################################### PLOT #######################################
####################################################################################
# Create a list of column names to plot
columns_to_plot = ['tt_00', 'tt_01', 'tt_10', 'tt_11', 'tt_20', 'tt_21', 'tt_AVG']

columns_to_plot = [s + "_perc" for s in columns_to_plot]
#columns_to_plot = [s for s in columns_to_plot]

####################################################################################
####################################################################################
# # Create a list to store sorted data
# sorted_data = []
# # Iterate over the columns and sort the data
# for column in columns_to_plot:
#     sorted_data.append(sorted(df_new[column]))
# # Generate x-axis values (indices of sorted data)
# x = range(len(sorted_data[0]))  # Assuming all columns have the same length
# # Plot the sorted data
# for i, column in enumerate(columns_to_plot):
#     plt.plot(x, sorted_data[i], label=column)
# # Add labels and title
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Plot of Columns')
# # Add legend
# plt.legend()
# # Show the plot
# plt.show()
####################################################################################
####################################################################################

####################################################################################
####################################################################################
sorted_data = df_new[columns_to_plot].values


print(sorted_data)
print(sorted_data.shape)
# Create the histogram
fig, ax = plt.subplots()



# Sort each column from high to low
sorted_arr = np.sort(sorted_data, axis=0)[::-1]


print(sorted_arr)
stop


data_mean = np.mean(sorted_arr, axis = 1)
print(np.mean(data_mean, axis = 0))




print(len(np.where(data_mean>20)[0])/len(data_mean))
print(len(np.where(sorted_arr[:,6]>20)[0])/len(data_mean))

np.savetxt('data_mean.txt', data_mean)
# stop
# stop
np.savetxt('data_NEW_2500.txt', sorted_arr)
# stop
stop
# Generate colors for each column
colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan'] #['red', 'blue', 'lime', 'orange', 'black', 'yellow']

# Create a line plot
# for i in range(sorted_arr.shape[0]):
#     plt.plot(sorted_arr[i, :], color=colors[i], label=f'Column {i+1}')
# Create a line plot with transposed axes
# for i in range(sorted_arr.shape[1]):
#     plt.plot(sorted_arr[i, :], color=colors[i], label=f'Column {i+1}')

# Create a line plot with transposed axes
for i in range(sorted_arr.shape[1]):
    plt.plot(sorted_arr[:, i], color=colors[i], label=columns_to_plot[i])

# n, bins, patches = ax.hist(sorted_data, 40, histtype='bar', color = ['red', 'blue', 'lime', 'orange', 'black', 'yellow'], label=columns_to_plot)
# print(bins)

# stop
# Set x-labels at bin locations
# ax.set_xticks(bins)
# Optionally, you can format the x-tick labels
# For example, you can rotate them for better readability
plt.xticks(rotation=45)
# Add other labels, title, legend, etc.
ax.set_xlabel('Bins')
ax.set_ylabel('Frequency')
ax.set_title('Histogram')
ax.legend()

# Show the plot
plt.show()
####################################################################################
####################################################################################

