# in this script we compare a model with and without rollouts.
# this allows us to investigate whether rollouts improve performance, taking into account the opportunity cost.

# load scripts and model
include("anal_utils.jl")
using ToPlanOrNotToPlan

println("comparing performance with and without rollouts")

loss_hp = LossHyperparameters(0, 0, 0, 0) #we're not computing a loss

greedy_actions = true
epoch = plan_epoch
results = Dict() #dictionary to store results
batch_size = 50000 #number of environments to run

for seed = seeds #for each independently trained model

    results[seed] = Dict() #results for this model
    for plan = [false; true] #no rollouts (false) or rollouts (true)
        Random.seed!(1) #set a seed for reproducibility

        filename = "N100_T50_Lplan8_seed$(seed)_$epoch" #model to load
        network, opt, store, hps, policy, prediction = recover_model(loaddir*filename) #load model parameters
        Larena = hps["Larena"]
        #construct environment, noting whether rollouts are allowed or now
        model_properties, wall_environment, model_eval = build_environment(
            Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions,no_planning = (~plan)
        )
        m = ModularModel(model_properties, network, policy, prediction, forward_modular) #construct model
        Nstates = Larena^2

        tic = time()
        L, ys, rews, as, world_states, hs = run_episode(
            m, wall_environment, loss_hp; batch=batch_size, calc_loss = false
        ) #let the agent act in the environments (parallelized)

        #print a brief summary
        println(seed, " ", plan)
        println(sum(rews .> 0.5) / batch_size, " ", time() - tic)
        println("planning fraction: ", sum(as .> 4.5) / sum(as .> 0.5))
        results[seed][plan] = rews #write result before moving on
    end
end

#save results
@save "$(datadir)/performance_with_out_planning.bson" results



