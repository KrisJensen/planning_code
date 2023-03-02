## A recurrent network model of planning explains hippocampal replay and human behavior

In this repository, we provide code for training and analysing the recurrent planning agent described by Jensen et al. (2023).
To run the code, julia >= 1.7 should be installed together with all the packages from the Manifest.toml file.
To install these packages:\
    `cd ./computational_model`\
    `julia --project=.`\
    `using Pkg`\
    `Pkg.instantiate()`\
To run the pretrained models, BSON 0.3.5 and Flux 0.13.5 should be installed since backwards compatibility was not preserved for the latest versions of these packages.
The primary script used to train models is './computational_model/walls_train.jl'.
The primary script used for downstream analyses is './computational_model/analysis_scripts/generate_planning_data.jl'.

Please note that this repository is still in the process of being cleaned up and documented properly.
More thorough installation instructions and descriptions of different scripts will be added in due time.

For any questions, comments, or suggestions, please reach out to Kris Jensen (ktj21@cam.ac.uk).

### Relevant references:
Jensen et al. (2023)\
Wang & Kurth-Nelson et al. (2018)\
Mattar & Daw (2018)\
Agrawal et al. (2021)


