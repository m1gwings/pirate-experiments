#!/bin/bash

# Default value for the flag
SUPPORT_R_BASELINE=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --support-r-baseline) SUPPORT_R_BASELINE=true; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# 1. Create the environment from the yml file
# We use --prune to ensure it matches the file exactly
echo "Creating/Updating Conda environment: pirate_env..."
conda env create -f environment.yml || conda env update -f environment.yml

# Execute R-related setup only if the flag is true
if [ "$SUPPORT_R_BASELINE" = true ]; then
    echo "R baseline support enabled. Proceeding with R setup..."

    # 2. Find the path to the new environment's R executable
    # This ensures we install the package into the right place
    CONDA_R_PATH=$(conda info --base)/envs/pirate_env/bin/R

    # 3. Use R to install the latest BayesMallows algorithm from CRAN
    echo "Installing latest BayesMallows algorithm via R..."
    $CONDA_R_PATH -e "install.packages('BayesMallows', repos='https://cloud.r-project.org')"

    # 4. Verify the version
    echo "Verifying installation..."
    $CONDA_R_PATH -e "cat('Installed BayesMallows algorithm version: ', as.character(packageVersion('BayesMallows')), '\n')"
fi

echo "Setup complete! Activate your environment with: conda activate pirate_env"
