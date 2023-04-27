#in this script, we consider how rewards and rollout probabilities change over learning
#for models of different sizes.

#load some stuff
include("anal_utils.jl")
using ToPlanOrNotToPlan

println("analyzing learning and rollouts for different model sizes")

loss_hp = LossHyperparameters(0, 0, 0, 0) #not computing losses
Nhiddens = 40:20:120 #number of hidden units
epochs = 0:50:1000 #training epochs to consider


meanrews, pfracs = [zeros(length(Nhiddens), length(seeds), length(epochs)) for _ = 1:2] #containers for storing results
for (ihid, Nhidden) = enumerate(Nhiddens) #for each network size
    for (iseed, seed) = enumerate(seeds) #for each random seed
        for (iepoch, epoch) = enumerate(epochs)

            filename = "N$(Nhidden)_T50_Lplan8_seed$(seed)_$epoch" #model to load
            network, opt, store, hps, policy, prediction = recover_model(loaddir*filename) #load model parameters
            
            #instantiate environment and agent
            Larena = hps["Larena"]
            model_properties, wall_environment, model_eval = build_environment(
                Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions
            )
            m = ModularModel(model_properties, network, policy, prediction, forward_modular)

            Random.seed!(1) #set random seed for reproducibility
            batch_size = 5000 #number of environments to consider
            tic = time()
            #run the experiment
            L, ys, rews, as, world_states, hs = run_episode(
                m, wall_environment, loss_hp; batch=batch_size, calc_loss = false
            )
            plan_frac = sum(as .== 5)/sum(as .> 0.5) #fraction of actions that were rollouts
            mean_rew = sum(rews .> 0.5) / batch_size #average reward per episode

            println("N=$Nhidden, seed=$seed, epoch=$epoch, avg rew=$(mean_rew), rollout fraction=$(plan_frac)")
            #store results
            meanrews[ihid, iseed, iepoch] = mean_rew
            pfracs[ihid, iseed, iepoch] = plan_frac

        end
    end
end

#save all results
res_dict = Dict("seeds" => seeds,
                "Nhiddens" => Nhiddens,
                "epochs" => epochs,
                "meanrews" => meanrews,
                "planfracs" => pfracs)
@save datadir * "rew_and_plan_by_n_new.bson" res_dict

