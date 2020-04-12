# covid19-modelling

### A Outline of Project for MATH371 Wi2020}

by Natasha Ting <br/>

BA Hons Economics <br/>
University of Alberta 2020 <br/>


### Objective
Develope method of estimation of parameters of the SIRD Epidemiology model without assumption of population. 

### Result
Method estimates parameters by fitting the SIRD and SIRDQ model on data. However, it is not robust to the changes in initial guess in N (population) fed to the optimizer.

### Usage
1. use `system_estimation.py` to estimate parameters. 
2. `getfiles.py` gets the data in format needed for estimation
3. `dataset.py` updates daily data from https://github.com/canghailan/Wuhan-2019-nCoV

### Development
This repository is under development to optimize speed and to increase robustness. 

