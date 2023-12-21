## A recurrent network model of planning explains hippocampal replay and human behavior

In this repository, we provide code for training and analysing the reinforcement learning agent described in Jensen et al. (2023): "A recurrent network model of planning explains hippocampal replay and human behavior" (https://www.biorxiv.org/content/10.1101/2023.01.16.523429v2).

Human data for the behavioural experiments both with and without periodic boundaries can be found in the `human_data/` directory.
Code for training and analysing the reinforcement learning models, as well as generating the figures in the paper, can be found in the `computational_model/` directory.
A collection of pretrained base models is provided in `./computational_model/models/`.

To run the code, julia >= 1.7 should be installed together with all the packages from the Manifest.toml file.
To install these packages:\
    `cd ./computational_model`\
    `julia --project=.`\
    `using Pkg`\
    `Pkg.instantiate()`\
To run the pretrained models, BSON 0.3.5 and Flux 0.13.5 should be installed since backwards compatibility was not preserved for the latest versions of these packages.
Julia 1.8.0 was used for all analyses in the paper.

The primary script used to train RL agents is './computational_model/walls_train.jl'.
A useful script for getting started on downstream analyses of the computational model is './computational_model/analysis_scripts/analyse_rollout_timing.jl'.
The primary script used for analyses of the human data is './computational_model/analysis_scripts/analyse_human_data.jl'.

Please note that this repository is still in the process of being cleaned up and documented properly.
More thorough installation instructions and descriptions of different scripts will be added in due time.
Pretrained models will also be added soon to facilitate analyses of the RL agents without having to train them from scratch.

For any questions, comments, or suggestions, please reach out to Kris Jensen (kris.torp.jensen@gmail.com).

