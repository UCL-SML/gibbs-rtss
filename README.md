# gibbs-rtss
This software package provides a Matlab implementation of the Gibbs-filter and Gibbs-RTSS as described in

Marc Peter Deisenroth and Henrik Ohlsson
"A General Perspective on Gaussian Filtering and Smoothing:
Explaining Current and Deriving New Algorithms"
in Proceedings of the 2011 American Control Conference (ACC 2011)

The software package also contains implementations of the following filters/smoothers:
- Gibbs-filter/Gibbs-RTSS
- EKF/EKS
- UKF/URTSS
- CKF/CKS

Running 
				demo_nonlinear_model				
reproduces the results from the paper for the nonlinear example in the paper (figures and numbers).

The code requires MatlabR2007a or newer.

(C) Copyright 2016 by Marc Deisenroth
 
Permission is granted for anyone to copy, use, or modify this
software and accompanying documents for any uncommercial
purposes, provided this copyright notice is retained, and note is
made of any changes that have been made. This software and
documents are distributed without any warranty.

I'd appreciate any feedback on the code (useful, buggy, inefficient, ...)
