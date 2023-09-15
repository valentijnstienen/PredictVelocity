# PemPem-paper 2
Supplements to the paper "Better routing in developing regions: weather and satellite-informed road speed prediction".

[comment]: <> (Include authors)

In this paper, we predict the expected velocity a vehicle can drive under different circumstances based on the information from GPS trajectory data. The starting point of this method is a road network. For some of the edges, we require that vehicles have been driving these roads under different circumstances (e.g., different weather, time of day, etc.). We will use this information to predict the velocity a vehicle is expected to drive on all the edges in all different circumstances.  

More specifically, this information should be stored as follows. 








The road network is represented as a NetworkX graph. One of the edge attributes, specifically, represents information about previous drivings. 

| Edge                | Velocity observations|
| :------------------ |:-------|
| (a, b, 0) |(2020-02-17 12:43:30,13.27), (2020-02-17 12:44:00,13.07)|
| (c, d, 1) |(2020-02-17 12:43:30,13.27)|
| (b, d, 0) |(2020-02-17 12:43:30,13.27), (2020-02-17 12:43:30,13.27), ...|
| ... | ... | 


## Feature extension
The next step is to create an extended dataset that we will use for our prediction algorithm. 

### Features
Image of the road, (historical) weather at the time of driving, and wether it is dark outside (related to the time of driving).





The details of this process can be found here (!). The resulting dataset looks as follows:

|Image (100m x 100m) | Location | Rainfall 1h (cat) | Dark | Velocity (km/h) |
| :------------------ |-------:|--------:|------:|------:|
|[[[122, 114, 91], [139, 131, 103 ...| (102.41 -0.54) | 0 | Yes | 33 |
|[[[122, 114, 91], [139, 131, 103 ...| (102.41 -0.54) | 1 | No | 42 |
|[[[122, 114, 91], [139, 131, 103 ...| (102.41 -0.54) | 1 | No | 46|
| ... | ... | ... | ... | ... |



## Set up the environment
In order to use this tool, you need to set up the appropriate environment. The packages that are used are stored in ([requirements.txt](https://github.com/valentijnstienen/PemPem-paper/tree/main/requirements.txt)). If you use [conda](https://conda.io), you can clone the virtual environments by executing the following code in you command line. Make sure that you first look in the channel conda-forge before turning to the anaconda channel (otherwise conflicts may occur). 

```
$ conda create --name ENV_NAME --file requirements.txt
```

## Running the algorithm
Before running the algorithm, you need to define the settings of the algorithm ([SETTINGS.py](https://github.com/valentijnstienen/PemPem-paper/tree/main/SETTINGS.py)). After filling in these settings, you can run the algorithm by executing the following commands:

```
$ python mainRun.py
```
