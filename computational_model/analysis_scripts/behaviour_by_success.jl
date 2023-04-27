#in this script, we look at the causal effect of replays on behaviour
#we do this separately for successful and unsuccessful replays

#load some scripts
include("anal_utils.jl")
using ToPlanOrNotToPlan

println("analysing behaviour after successful and unsuccessful replays")

function run_planning_causal(;seeds, prefix, N, Lplan, epoch, greedy_actions)


for seed = seeds #iterate through independently trained models

    fname = "$(prefix)N$(N)_T50_Lplan$(Lplan)_seed$(seed)_$(epoch)" #model name
    network, opt, store, hps, policy, prediction = recover_model("$datadir$fname") #load parameters

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
    V_old, V_new, V_sim = [zeros(3, nreps) .+ NaN for _ = 1:3] #value functions before, after and during rollout
    planning_is = zeros(3, nreps) .+ NaN #number of attempts to get this scenario

    for rep = 1:nreps #for each repetition (i.e. newly sampled environment)
        if rep % 100 == 0 println("environment $rep of $nreps") end

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
            sim_a, sim_V, Vt_old, h_old, path, πt_old = NaN, NaN, NaN, NaN, NaN, NaN
            iplan = 0
            just_planned, stored_plan = false, false

            
            while any(world_state.environment_state.time .< (40 - 1e-2)) #iterate through some trials


                h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn) #RNN step
                active = (world_state.environment_state.time .< (T+1 - 1e-2)) #active episodes
                πt = exp.(agent_output[1:5, :]) #current policy
                Vt = agent_output[6, 1] #current value function

                if just_planned && (~stored_plan) #if we just performed a rollout and this is the first one
                    stored_plan = true #we have now stored our plan
                    Print && println("storing plan $iplan $sim_a")
                    p_continue_sim[irew, rep] = πt[5]
                    p_initial_sim[irew, rep] = πt_old[5]
                    if isnan(sim_a) p_a, p_a_old = NaN, NaN else p_a, p_a_old = πt[sim_a], πt_old[sim_a] end
                    p_simulated_actions[irew, rep] = p_a
                    p_simulated_actions_old[irew, rep] = p_a_old
                    sim_lengths[irew, rep] = sum(path[:, :, 1])

                    ps = world_state.environment_state.reward_location
                    ws = world_state.environment_state.wall_loc
                    state = Int.(world_state.agent_state)
                    plan_dists[irew, rep] = dist_to_rew(ps, ws, Larena)[state[1], state[2]]

                    V_old[irew, rep] = Vt_old
                    V_new[irew, rep] = Vt
                    V_sim[irew, rep] = sim_V
                    hidden_old[irew, :, rep] = h_old[:]
                    hidden_new[irew, :, rep] = h_rnn[:]
                    planning_is[irew, rep] = iplan
                end
                πt_old = copy(πt)
                Vt_old = copy(Vt)
                h_old = copy(h_rnn)

                if (tot_rew[1] == 1) && stored_plan && (a[1] > 4.5)
                    #if we're in trial two and have already planned, don't plan any more
                    if greedy_actions
                        a[1] = argmax(πt[1:4])
                    else
                        a[1] = rand(Categorical(πt[1:4] / sum(πt[1:4])))
                    end
                end

                just_planned = false
                donut_plan = (.~(active .& (a[:] .> 4.5) .& (rew[:] .< 0.5))) #whether to plan
                if isnothing(plan_to_rew) && exploit[1] && (~stored_plan) && (~donut_plan[1])
                    #skip planning
                    Print && println("skipping")
                    if greedy_actions
                        a[1] = argmax(πt[1:4])
                    else
                        a[1] = rand(Categorical(πt[1:4] / sum(πt[1:4])))
                    end
                    sim_a = a[1]
                    sim_V = NaN
                    just_planned = true
                end

                ### explicitly run planning and return states!!! ###
                donut_plan = (.~(active .& (a[:] .> 4.5) .& (rew[:] .< 0.5))) #whether to plan
                ahot = zeros(Float32, 5, batch); for b = 1:batch ahot[Int(a[b]), b] = 1f0 end
                iplan = 0
                cont = true
                planning_state = nothing
                while cont
                    #println("planning ", isnothing(plan_to_rew), " ", (~stored_plan), " ", (a[1] > 4.5))
                    iplan += 1
                    ###run planning and return states!!!
                    #if iplan % 10 == 1 println("planning ", iplan) end
                    planning_state, plan_inds, (path, all_Vs, found_rew, plan_states) = planner.planning_algorithm(world_state,
                                                                    ahot,
                                                                    wall_environment.dimensions,
                                                                    agent_output,
                                                                    donut_plan,
                                                                    planner,
                                                                    m,
                                                                    h_rnn,
                                                                    m.model_properties,
                                                                    returnall = true,
                                                                    true_transition = false)
                    
                    pinput = planning_state.plan_input
                    found_rew = (pinput[end, :] .> 0.5)

                    if ~exploit[1] || donut_plan[1] || stored_plan
                        # if (i) in exploitation, (ii) not planning, (iii) already stored a plan, continue as normal
                        cont = false #run normal dynamics during exploration or right after reward or after storing plan
                    elseif plan_to_rew && found_rew[1]
                        cont = false #consider trajectories that go to 
                        sim_a = argmax(path[:, 1, 1])
                        sim_V = pinput[size(pinput, 1)-1, 1]
                        just_planned = true
                    elseif (~plan_to_rew) && (~found_rew[1])
                        cont = false
                        sim_a = argmax(path[:, 1, 1])
                        sim_V = pinput[size(pinput, 1)-1, 1]
                        just_planned = true
                    elseif iplan > 100
                        cont = false
                        sim_a = NaN
                        sim_V = NaN
                        just_planned = true
                    end
                end

                if rew[1] > 0.5
                    if tot_rew[1] == 0
                        #store time of first reward
                        t_first_rew = world_state.environment_state.time[1]
                    elseif (tot_rew[1] == 1) && stored_plan
                        #second reward and we planned before this
                        time_to_second_rew[irew, rep] = (world_state.environment_state.time[1] - t_first_rew)
                    end
                end

                tot_rew += Float64.(rew .> 0.5)
                exploit[rew[:] .> 0.5] .= true #exploitation phase
                Print && if (rew[1] > 0.5) && (isnan(p_continue_sim[irew, rep])) println("found rew!") end

                ### now update environment dynamics ###
                rew, world_state, predictions, ahot, teleport = act_and_receive_reward(
                    a, world_state, planner, wall_environment.dimensions, agent_output, m, h_rnn, m.model_properties
                )

                @assert sum(planning_state.plan_input[1:4]) == sum(world_state.planning_state.plan_input[1:4])
                #overwrite planning

                #println(world_state.planning_state.plan_input[28:34])
                world_state = WorldState(agent_state=world_state.agent_state,
                        environment_state=world_state.environment_state,
                        planning_state=planning_state
                    )
                #println(world_state.planning_state.plan_input[28:34])
    
                agent_input = gen_input(world_state, ahot, rew, wall_environment.dimensions, m.model_properties)
                rew[rew .< 0f0] .= 0f0
                rew[.~active] .= 0f0

            end
            if (tot_rew[1] == 1) && stored_plan
                #planned during trial 2 but didn't find reward
                time_to_second_rew[irew, rep] = (world_state.environment_state.time[1] - t_first_rew)
            end
        end
    end

    ## evaluation and plotting 

    no_nans = findall(.~isnan.(plan_dists[3, :]))
    @assert all( (plan_dists[1, :] .== plan_dists[2, :])[no_nans] )
    @assert all( abs.(plan_dists[1, :] .- plan_dists[3, :])[no_nans] .< 1.5 )

    data = Dict("plan_dists" => plan_dists, "p_simulated_actions" => p_simulated_actions,
                "p_simulated_actions_old" => p_simulated_actions_old,
                "p_continue_sim" => p_continue_sim, "sim_lengths" => sim_lengths,
                "time_to_second_rew" => time_to_second_rew,
                "p_initial_sim" => p_initial_sim, "V_old" => V_old, "V_new" => V_new, "V_sim" => V_sim,
                "hidden_old" => hidden_old, "hidden_new" => hidden_new, "planning_is" => planning_is)

    @save "$datadir/causal_N$(N)_Lplan$(Lplan)_$(seed)_$epoch.bson" data

end

end

run = false
run && run_planning_causal(;seeds, prefix, N, Lplan, epoch, greedy_actions)