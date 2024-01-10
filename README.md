# DISCOVERING HIDDEN POLLUTION HOTSPOTS USING SPARSE SENSOR MEASUREMENTS
This repository houses the code used for the analysis of data, model training, and generation of figures for our paper with the above title. The data used for this project consists of publically available datasets as well as privately collected data. The public data used is cited accordingly in the paper. The privately collected data is governed by the data sharing agreement between NYU, Yale, and Kaiterra. For access to the private data, please reach out to Lakshminarayanan Subramanian at lakshmi@nyu.edu.

The repository is structured as:
```bash
./
├── code
│   ├── field_estimation
│   ├── policy_recs
│   └── source_apportionment
│       └── fnl_20180501_00-20201031_18_00.grib2
└── data
    ├── govdata
    └── kaiterra
```
The top level files CombinedLocations.xlsx and combined_distances.csv have the locations and distances between different deployed sensors (See Figure 1 in paper). The file distances_pilot2_20180130_to_20180202.xlsx has the distances for our pilot experiment (see Figure 2 in paper).

The data folder, not provided with this repository contains the folders govdata and kaiterra, each having the data collected from the respective sensor network as csv files.