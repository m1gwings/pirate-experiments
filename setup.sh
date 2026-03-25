# setup.sh
conda env create -f environment.yml
conda activate pirate_env
Rscript -e 'install.packages("BayesMallows", repos="http://cran.r-project.org")'
