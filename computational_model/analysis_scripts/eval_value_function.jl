## in this script we analyse how the value estimate of the agent changes with rollouts

## load scripts and model
include("anal_utils.jl")
using ToPlanOrNotToPlan

println("comparing performance with and without rollouts")

##

loss_hp = LossHyperparameters(0, 0, 0, 0) #we're not computing a loss

greedy_actions = true
epoch = plan_epoch
results = Dict() #dictionary to store results
batch_size = 10000 #number of environments to run
data = Dict()
for seed = seeds
    results[seed] = Dict() #results for this model
    Random.seed!(1) #set a seed for reproducibility

    filename = "N100_T50_Lplan8_seed$(seed)_$epoch" #model to load
    println("running $filename")
    network, opt, store, hps, policy, prediction = recover_model(loaddir*filename) #load model parameters
    Larena = hps["Larena"]

    ## construct environment, noting whether rollouts are allowed or now
    model_properties, wall_environment, model_eval = build_environment(
        Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions
    )
    m = ModularModel(model_properties, network, policy, prediction, forward_modular) #construct model
    Nstates = Larena^2

    tic = time()
    L, ys, rews, as, world_states, hs = run_episode(
        m, wall_environment, loss_hp; batch=batch_size, calc_loss = false
    ) #let the agent act in the environments (parallelized)

    #print a brief summary
    println("\n", seed)
    println(sum(rews .> 0.5) / batch_size, " ", time() - tic)
    println("planning fraction: ", sum(as .> 4.5) / sum(as .> 0.5))

    ##

    # extract time-within-episode
    ts = reduce(hcat, [state.environment_state.time for state = world_states])
    # compute all reward-to-gos
    rew_to_go = sum(rews .> 0.5, dims = 2) .- [zeros(batch_size) cumsum(rews[:, 1:size(rews,2)-1], dims = 2)]

    # compute all value functions
    Naction = wall_environment.dimensions.Naction
    Vs = ys[Naction+1, :, :]
    accuracy = abs.(Vs - rew_to_go) # accuracy of the value function

    ##
    plan_nums = Int.(zeros(size(accuracy))) # rollout iteration
    tot_plans = Int.(zeros(size(accuracy))) # total number of rollouts
    suc_rolls = Int.(zeros(size(accuracy))) # is this a response to a successful rollout
    num_suc_rolls = Int.(zeros(size(accuracy))) # number of successful rollouts in this sequence
    for b = 1:batch_size # for each episode
        plan_num, init_plan = 0, 0 # initialize
        if sum(rews[b, :]) > 0.5 # if we found the goal at least once
            for anum = findfirst(rews[b, :] .== 1)+1:sum(as[b, :] .> 0.5) # for each iteration
                a = as[b, anum]

                if (a == 5) && (rews[b, anum-1] != 1) # planning and didn't just get reward
                    plan_num += 1 # update rollout number within sequence
                    if plan_num == 1 # just started planning
                        init_plan = anum # iteration at which this rollout sequence started
                        # didn't plan on previous iteration, should have no plan input
                        @assert sum(world_states[anum].planning_state.plan_input[:, b]) < 0.5
                    else
                        # planned on previous iteration, should have planning input
                        @assert sum(world_states[anum].planning_state.plan_input[:, b]) > 0.5
                    end
                    plan_nums[b, anum] = plan_num-1 # number of rollouts before generating this output
                    suc_rolls[b, anum] = world_states[anum].planning_state.plan_input[end, b] # is this a response to a successful rollout?
                else
                    if plan_num > 0 # just finished planning
                        tot_plans[b, (init_plan):(anum)] .= plan_num # total number of rollouts in this sequence
                        plan_nums[b, anum] = plan_num
                        # double check that we've just planned
                        @assert sum(world_states[anum].planning_state.plan_input[:, b]) > 0.5
                        suc_rolls[b, anum] = world_states[anum].planning_state.plan_input[end, b] # is this a response to a successful rollout?
                        num_suc_rolls[b, (init_plan):(anum)] .= sum(suc_rolls[b, (init_plan):(anum)]) # total number of successful rollouts in this sequence
                    end
                    plan_num = 0 # reset planning counter
                end
            end
        end
    end

    # save relevant data
    data[seed] = Dict("tot_plans" => tot_plans,
                        "plan_nums" => plan_nums,
                        "suc_rolls" => suc_rolls,
                        "num_suc_rolls" => num_suc_rolls,
                        "Vs" => Vs,
                        "rew_to_go" => rew_to_go,
                        "as" => as,
                        "ts" => ts)

end
@save datadir * "value_function_eval.bson" data # store result
