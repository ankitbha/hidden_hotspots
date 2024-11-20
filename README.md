# Comprehensive Monitoring of Air Pollution Hotspots Using Sparse Sensor Networks
This repository houses the code used for the analysis of data, model training, and generation of figures for our paper with the above title. The data used for this project consists of publically available datasets as well as privately collected data. The public data used is cited accordingly in the paper. The privately collected data is governed by the data sharing agreement between NYU, Yale, and Kaiterra. 
The repository is structured as:
```bash
./
├── code
│   ├── field_estimation
│   ├── generating_figures
│   └── source_apportionment
│       
└── data
    ├── govdata
    └── kaiterra
```
The top level files CombinedLocations.xlsx and combined_distances.csv have the locations and distances between different deployed sensors (See Figure 1 in paper). The file distances_pilot2_20180130_to_20180202.xlsx has the distances for our pilot experiment (see Figure 2 in paper).

The data folder, not provided with this repository contains the folders govdata and kaiterra, each having the data collected from the respective sensor network as csv files.

The environment file for the code environment is provided to replicate the results.

Inside the code folder, the three subfolders are field_estimation, generating_figures, and source_apportionment. The generating_figures folder consists of notebooks that were used to generate the figures in the paper. The field_estimation folder contains the modeling code for Kriging and Neural Network approaches, while the source_apportionment folder contains the code for Gaussian dispersion model. 

