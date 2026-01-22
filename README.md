# TL-NDMnet

This repository includes the codes for the manuscript entitled "Time-lapse seismic processing using NDM-net", which was submitted to Computers & Geosciences for peer review. A neural network-based method for accelerating seismic monitoring. Given that the seismogram data at each monitoring stage is calculated using a coarse computational grid, and a portion of the seismograms (up to 20% of the total number of sources) are calculated using a fine computational grid, it is possible to utilize a neural network (NDM-net) in order to accelerate seismic monitoring.

# Code introduction
- [pix2pix.py](pix2pix.py): NDM-net model based on pix2pix architecture.
- [dataset.py](dataset.py): a code for converting a dataset into suitable format for NDM-net.
- [Time_Lapse_NDMnet.py)[Time_Lapse_NDMnet.py]: the test code for generating a training dataset using seismograms, training the NDM-net, and analyzing the results of applying the algorithm.

 
# Requirements
- cuda 8.0
- Python 3.9.18
- conda 23.3.1
- pytorch 1.12.1
- matplotlib 3.7.0
- numpy 1.23.1
- natsorted 8.2.0
- obspy
