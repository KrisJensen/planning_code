# functions

# plotting.jl
export plot_progress

# loss_hyperparameters.jl
export LossHyperparameters

# model.jl
export ModelProperties, ModularModel
export create_model_name
export build_model

# environment.jl
export EnvironmentDimensions, EnvironmentProperties, Environment, WorldState

# a2c.jl
export GRUind
export run_episode, model_loss
export forward_modular, sample_actions, construct_ahot, zeropad_data

# wall_build.jl
export Arena
export build_arena, build_environment
export act_and_receive_reward
export update_agent_state

# walls.jl
export onehot_from_state,
    onehot_from_loc,
    state_from_loc,
    state_ind_from_state,
    state_from_onehot,
    gen_input,
    get_wall_input,
    comp_π
export WallState

# train.jl
export gmap, gmap_grads_pred

# io.jl
export recover_model, save_model

# initializations.jl
export initialize_arena, gen_maze_walls, gen_wall_walls


# plotting
export arena_lines,
    plot_arena,
    plot_weiji_gif

#planners
export build_planner
export PlanState
export model_planner

#analyses of human data
export get_wall_rep
export extract_maze_data

#priors
export prior_loss

#baseline policies
export random_policy, dist_to_rew, optimal_policy