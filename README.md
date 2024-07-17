# mfpred

Flux rope prediction with machine learning building up on the code of Reiss et al. 2021, for real time deployment.

Copy these files into the data folder
- 
- 


## Installation 

Install python with miniconda:

on Linux:

	  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	  bash Miniconda3-latest-Linux-x86_64.sh

on MacOS:

	  curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
	  bash Miniconda3-latest-MacOSX-x86_64.sh

go to a directory of your choice

	  git clone https://github.com/cmoestl/mfpred
	  

Create a conda environment using the "environment.yml", and activate the environment:

	  conda env create -f environment.yml
      
	  pip install requirements.txt      

	  conda activate mfpred

