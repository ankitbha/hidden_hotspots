%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMIS - GNSS Missing Data Interpolation Software                      %
%                                                                      %
% Author: Ning Liu, Wujiao Dai, Rock Santerre, Cuilin Kuang            %
% E-mail: nliucsu@csu.edu.cn                                           %
% Affiliation: Central South University                                %
%              Dept. of Surveying and Remote Sensing Science           %
% Code website:                                                        %
%  http://faculty.csu.edu.cn/daiwujiao/en/lwcg/35988/content/11646.htm %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

There are two ways to use this software:
(1)check that your computer has an installed MATLAB Compiler Runtime (MCR) for version 8.1, 
   then double-click the executable program to run the software;
(2)add the source code folder into MATLAB search path, 
   type GMIS in the MATLAB command window;
   
Description of each file in this compressed folder:
- GNSS Missing Data Interpolation Software User Manual.docx
  This User Manual clearly describe each widget in each GUI with its function, and use two sample datasets to describe how to install and use the GMIS software
  
- software folder:
  There are two subfolders in this folder and four sample datasets.
  (a) exe subfolder contains: GMIS executable program
  (b) source code subfolder contains: GMIS source code
  
  (c) AP.mat: dataset example from Antarctic Peninsula GPS Network
  (d) scign.mat: dataset example from Southern California Integrated GPS Network(SCIGN,http://www.scign.org/)
  (e) tjcors.mat: dataset example from Tian Jin CORS net in China
  (f) ukcors.mat: dataset example from NERC NERC British Isles continuous GNSS Facility in United Kingdom(BIGF,http://bigf.ac.uk)
  
- software\exe\GMIS.exe:
  GMIS executable program, double-click it and run the software
  
- software\source code:
  (a) GMIS.m and GMIS.fig: main GMIS GUI source code file. It is used to load GNSS data, save the filter and interpolated result, display the three dimensional source and result GNSS data, and invoke other subGUI.
  (b) SemiVariog.m and SemiVariog.fig: SemiVarigFit GUI source code file. It is used to calculate the empirical semi-variogram value and fit a semi-variogram function. The empirical semi-variogram value and fitting result can also be displayed in this GUI.
  (c) EMInterp.m and EMInterp.fig: InterpolationMode GUI source code file. It is used to set some parameters of EM algorithm and KKF model.
  (d) About.m and About.fig: About GUI source code file. It is used to show GMIS software version.
  
  (e) EMEst_filter.m: EM algorithm with Kalman filter source code
  (f) EMEst_smooth.m: EM algorithm with Kalman smooth source code
  (g) SpatialFiled.m: This source code is used to calculate spatial filed of KKF
  (h) trendpoly0.m: This source code is used to calculate constant trend filed
  (i) trendpoly1.m: This source code is used to calculate linear trend filed
  (j) trendpoly3.m: This source code is used to calculate quadratic trend filed
  (k) variog.m: This source code is used to calculate empirical semi-variogram value
  (l) variogramfit.m: This source code is used to fit empirical semi-variogram value
  (m) fminsearchbnd.m: a sub function which is needed in variogramfit.m
  