#in this script, we look at the causal effect of replays on behaviour
#we do this separately for successful and unsuccessful replays

#load some scripts
include("anal_utils.jl")
using ToPlanOrNotToPlan

try
    println("running default analyses: ", run_default_analyses)
catch e
    global run_default_analyses = true
end

function run_causal_rollouts(;seeds, N, Lplan, epoch, prefix = "")
println("analysing behaviour after successful and unsuccessful replays")

for seed = seeds #iterate through independently trained models

    fname = "N$(N)_T50_Lplan$(Lplan)_seed$(seed)_$(epoch)" #model name
    println("loading $fname")
    network, opt, store, hps, policy, prediction = recover_model("$loaddir$fname") #load parameters

    Larena = hps["Larena"] #arena size
    #construct environment
    model_properties, wall_environment, model_eval = build_environment(
        Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions
    )
    m = ModularModel(model_properties, network, policy, prediction, forward_modular) #construct model

    # set some basic parameters
    batch = 1
    ed = wall_environment.dimensions
    Nout, Nhidden = m.model_properties.Nout, m.model_properties.Nhidden
    Nstates, T = ed.Nstates, ed.T
    nreps = 1000 #number of environments to consider

    #intialize some containers for storing variables of interest
    p_continue_sim = zeros(3, nreps) .+ NaN #probability of doing another rollout
    p_initial_sim = zeros(3, nreps) .+ NaN #initial probability of rollout
    p_simulated_actions = zeros(3, nreps) .+ NaN #probability of taking a_hat
    p_simulated_actions_old = zeros(3, nreps) .+ NaN #initial probability of taking a_hat
    time_to_second_rew = zeros(3, nreps) .+ NaN #time to get to reward
    plan_dists = zeros(3, nreps) .+ NaN #distance to reward at rollout time
    sim_lengths = zeros(3, nreps) .+ NaN #number of rollout actions
    hidden_old = zeros(3, hps["Nhidden"], nreps) .+ NaN #hidden state before rollout
    hidden_new = zeros(3, hps["Nhidden"], nreps) .+ NaN #hidden state after rollout
    V_old, V_new = [zeros(3, nreps) .+ NaN for _ = 1:2] #value functions before and after rollout
    planning_is = zeros(3, nreps) .+ NaN #number of attempts to get this scenario

    for rep = 1:nreps #for each repetition (i.e. newly sampled environment)
        if rep % 200 == 0 println("environment $rep of $nreps") end

        for (irew, plan_to_rew) = enumerate([true; false; nothing]) #for successful/unsuccessful/no rollout
            Random.seed!(rep) #set random seed for consistent environment

            #initialize environment
            world_state, agent_input = wall_environment.initialize(
                zeros(2), zeros(2), batch, m.model_properties
            )
            agent_state = world_state.agent_state
            h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch)) #expand hidden state
            rew = zeros(batch)
            tot_rew = zeros(batch)
            t_first_rew = 0 #time of first reward

            # let the agent act in the world

            exploit = Bool.(zeros(batch)) #are we in the exploitation phase
            planner, initial_plan_state = build_planner(Lplan, Larena) #initialize planning module
            all_rews = []
            #instantiate some variables
            sim_a, Vt_old, h_old, path, πt_old = NaN, NaN, NaN, NaN, NaN
            iplan = 0
            just_planned, stored_plan = false, false

            
            while any(world_state.environment_state.time .< (40 - 1e-2)) #iterate through some trials


                h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn) #RNN step
                active = (world_state.environment_state.time .< (T+1 - 1e-2)) #active episodes
                πt = exp.(agent_output[1:5, :]) #current policy
                Vt = agent_output[6, 1] #current value function

                if just_planned && (~stored_plan) #if we just performed a rollout and this is the first one
                    stored_plan = true #we have now performed a rollout
                    p_continue_sim[irew, rep] = πt[5] #store the new policy
                    p_initial_sim[irew, rep] = πt_old[5] #store the old policy
                    #pre/post probabilities of taking the imagined action
                    if isnan(sim_a) p_a, p_a_old = NaN, NaN else p_a, p_a_old = πt[sim_a], πt_old[sim_a] end
                    p_simulated_actions[irew, rep] = p_a
                    p_simulated_actions_old[irew, rep] = p_a_old
                    sim_lengths[irew, rep] = sum(path[:, :, 1]) #rollout length

                    ps = world_state.environment_state.reward_location
                    ws = world_state.environment_state.wall_loc
                    state = Int.(world_state.agent_state)
                    plan_dists[irew, rep] = dist_to_rew(ps, ws, Larena)[state[1], state[2]] #distance to goal
                    
                    #store value functions, hidden states, and planning iteration
                    V_old[irew, rep] = Vt_old
                    V_new[irew, rep] = Vt
                    hidden_old[irew, :, rep] = h_old[:]
                    hidden_new[irew, :, rep] = h_rnn[:]
                    planning_is[irew, rep] = iplan
                end
                #store most recent variables for an additional iteration
                πt_old = copy(πt)
                Vt_old = copy(Vt)
                h_old = copy(h_rnn)

                if (tot_rew[1] == 1) && stored_plan && (a[1] > 4.5)
                    #if we're in trial two and have already planned, don't plan any more
                    if greedy_actions
                        a[1] = argmax(πt[1:4]) #only consider physical actions
                    else
                        a[1] = rand(Categorical(πt[1:4] / sum(πt[1:4])))
                    end
                end

                just_planned = false #have we just done a rollout
                donot_plan = (.~(active .& (a[:] .> 4.5) .& (rew[:] .< 0.5))) #whether to plan
                if isnothing(plan_to_rew) && exploit[1] && (~stored_plan) && (~donot_plan[1])
                    #if we should have planned but are in the no-planning condition, store some data
                    if greedy_actions
                        a[1] = argmax(πt[1:4]) #skip planning
                    else
                        a[1] = rand(Categorical(πt[1:4] / sum(πt[1:4])))
                    end
                    sim_a = a[1]
                    just_planned = true #as if we have planned
                    donot_plan[1] = true #now don't plan again
                end

                #explicitly run planning and return states
                ahot = zeros(Float32, 5, batch); for b = 1:batch ahot[Int(a[b]), b] = 1f0 end #1hot action representation
                iplan = 0 #number of resamples
                cont = true #whether we should continue resampling
                planning_state = nothing #result of rollout
                while cont #while we haven't finished
                    iplan += 1 #iterate attempt number
                    # sample a rollout
                    planning_state, plan_inds, (path, all_Vs, found_rew, plan_states) = planner.planning_algorithm(world_state,
                                                                    ahot,
                                                                    wall_environment.dimensions,
                                                                    agent_output,
                                                                    donot_plan,
                                                                    planner,
                                                                    m,
                                                                    h_rnn,
                                                                    m.model_properties,
                                                                    returnall = true)
                    
                    pinput = planning_state.plan_input #feedback from rollout
                    found_rew = (pinput[end, :] .> 0.5) #was it successful
                    if ~exploit[1] || donot_plan[1] || stored_plan
                        # if (i) in exploration, (ii) not planning, or (iii) already stored a plan, continue as normal
                        cont = false
                    elseif plan_to_rew && found_rew[1] #in 'successful' condition and sampled successful rollout
                        cont = false #don't sample anymore
                        sim_a = argmax(path[:, 1, 1]) #first sampled action
                        just_planned = true #just did a rollout
                    elseif (~plan_to_rew) && (~found_rew[1]) #in 'unsuccessful' condition and samples unsuccessful rollout
                        cont = false #don't sample anymore
                        sim_a = argmax(path[:, 1, 1]) #first sampled action
                        just_planned = true #just did a rollout
                    elseif iplan > 100 #if we've exceeded our limit
                        cont = false #don't sample anymore
                        sim_a = NaN #no sampled action
                        just_planned = true #as if we just did a rollout (give up)
                    end
                end

                if rew[1] > 0.5 #if we found reward
                    if tot_rew[1] == 0 #first reward
                        t_first_rew = world_state.environment_state.time[1] #store time of first reward
                    elseif (tot_rew[1] == 1) && stored_plan #second reward and we did a rollout to start the trial
                        time_to_second_rew[irew, rep] = (world_state.environment_state.time[1] - t_first_rew) #store time to rew
                    end
                end

                tot_rew += Float64.(rew .> 0.5) #total reward accumulated
                exploit[rew[:] .> 0.5] .= true #exploitation phase if we've found reward

                # now perform a step of the environment dynamics
                rew, world_state, predictions, ahot, teleport = act_and_receive_reward(
                    a, world_state, planner, wall_environment.dimensions, agent_output, m, h_rnn, m.model_properties
                )

                #overwrite the rollout
                world_state = WorldState(agent_state=world_state.agent_state,
                        environment_state=world_state.environment_state,
                        planning_state=planning_state #the one sampled above
                    )
                #check that we successfully overwrote the rollout
                @assert all(planning_state.plan_input .== world_state.planning_state.plan_input)
                
                #generate input for the agent
                agent_input = gen_input(world_state, ahot, rew, wall_environment.dimensions, m.model_properties)
                rew[.~active] .= 0f0 #zero out for finished episodes

            end
            if (tot_rew[1] == 1) && stored_plan
                #did a rollout during trial 2 but didn't find reward
                time_to_second_rew[irew, rep] = (world_state.environment_state.time[1] - t_first_rew)
            end
        end
    end

    ## evaluation and plotting 

    no_nans = findall(.~isnan.(plan_dists[3, :])) #indices where we stored data
    @assert all( (plan_dists[1, :] .== plan_dists[2, :])[no_nans] ) #check that we used the same environments

    #collect data
    data = Dict("plan_dists" => plan_dists, "p_simulated_actions" => p_simulated_actions,
                "p_simulated_actions_old" => p_simulated_actions_old,
                "p_continue_sim" => p_continue_sim, "sim_lengths" => sim_lengths,
                "time_to_second_rew" => time_to_second_rew,
                "p_initial_sim" => p_initial_sim, "V_old" => V_old, "V_new" => V_new,
                "hidden_old" => hidden_old, "hidden_new" => hidden_new, "planning_is" => planning_is)

    #write data
    @save "$datadir/$(prefix)causal_N$(N)_Lplan$(Lplan)_$(seed)_$epoch.bson" data

end

end

#run_default_analyses is a global parameter in anal_utils.jl
run_default_analyses && run_causal_rollouts(;seeds, N, Lplan, epoch)