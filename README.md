# Hydrogen Sentinel Project


The hydrogen Sentinel project was a project highly assisted by Dr Jiajia Sun from UH University


Important files:

Code:
./sider/integrator.py - the bulk of the project is here, contains ALL data extraction and processing
functions: gets band data from sentinel 2, data preprocessing, spectral analysis and normalization, outputs to datafiles and csv
prerequisites: associated USGS data (eg S07SNTL2)

./sider/scatterplot.py
function: generation of mineral image files
prerequisites: integrator.py must be run, example run already completed within this repo

./sider/barplot.py
function: generation of figure 1
prerequisites: integrator.py must be run

./sider/cluster.py
function: generation of spectra stuff image
prerequisites: integrator.py must be run


Images:
./sider/Olivine.png
./sider/Brucite.png
./sider/Cummingtonite.png
./sider/Serpentine.png
./sider/Figure_1.png
./sider/spectrastuff.png
were used on the poster and are fundemental to my research

Data files
./api_features_coord.csv - coords extracted from arcgis
./merged.cvv - combined point data

