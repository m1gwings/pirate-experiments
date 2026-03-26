#!/bin/bash

# 1. Create the environment from the yml file
# We use --prune to ensure it matches the file exactly
echo "Creating/Updating Conda environment: pirate_env..."
conda env create -f environment.yml || conda env update -f environment.yml

# 2. Find the path to the new environment's R executable
# This ensures we install the package into the right place
CONDA_R_PATH=$(conda info --base)/envs/pirate_env/bin/R

# 3. Use R to install the latest BayesMallows from CRAN
echo "Installing latest BayesMallows via R..."
$CONDA_R_PATH -e "install.packages('BayesMallows', repos='https://cloud.r-project.org')"

# 4. Verify the version
echo "Verifying installation..."
$CONDA_R_PATH -e "cat('Installed BayesMallows Version: ', as.character(packageVersion('BayesMallows')), '\n')"

echo "Setup complete! Activate your environment with: conda activate pirate_env"
