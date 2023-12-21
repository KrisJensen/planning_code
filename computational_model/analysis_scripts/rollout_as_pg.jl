#in this script, we compare the hidden state updates of the models to those predicted by policy gradients

#load some stuff
include("anal_utils.jl")
using ToPlanOrNotToPlan

println("comparing hidden state updates to policy gradients")

res_dict = Dict() #instantiate dictionary for storing results
for seed = seeds #for each RL agent
res_dict[seed] = Dict() #results for this agent

filename = "N100_T50_Lplan8_seed$(seed)_$epoch" #model to load
println("loading model: $filename")
network, opt, store, hps, policy, prediction = recover_model(loaddir*filename) #load model parameters

#construct environment and RL agent
Larena = hps["Larena"]
model_properties, wall_environment, model_eval = build_environment(
                Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions
            )
m = ModularModel(model_properties, network, policy, prediction, forward_modular)

#define some useful parameters
Nstates = Larena^2
Naction = wall_environment.dimensions.Naction
Nin_base = Naction + 1 + 1 + Nstates + 2 * Nstates #number of 'physical' input channels
Nout, Nhidden, Nin = m.model_properties.Nout, m.model_properties.Nhidden, m.model_properties.Nin
Lplan = model_properties.Lplan
ed = wall_environment.dimensions
Nstates, Naction, T = ed.Nstates,  ed.Naction, ed.T

#instantiate some arrays for storing results
all_sim_as, all_sim_a2s = [], [] #first and second rollout actions
all_jacs, all_jacs_shift, all_jacs_shift2 = [], [], [] #Jacobians (dh/R_tau), Jacs with fake action 1, and Jacs with fake action 2
all_gs, all_gs2 = [], [] #Gradients and gradients for second action
all_pis = [] # log policies over actions
full_inps = [] #rollout feedback
meangv = [] #instantiate

Random.seed!(2) #set random seed for reproducibility
for (i_mode, mode) = enumerate(["R_tau", "test"]) #first estimate the direction of R_tau, then actual test episodes
    if mode == "R_tau" println("estimating R_tau") else println("computing policy gradients") end
    all_rews = [] #container for storing rewards
    agent_output = nothing # instantiate

    # run a handful of steps
    batch = 1002 #number of episodes to consider

    world_state, agent_input = wall_environment.initialize(
        zeros(2), zeros(2), batch, m.model_properties
    )
    agent_state = world_state.agent_state
    h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch)) #expand hidden state
    rew = zeros(batch) #instantiate reward vector to zeros

    exploit, just_planned = Bool.(zeros(batch)), Bool.(zeros(batch)) #in exploitation phase and did I just do a rollout?
    tmax = 50 #maximum number of iterations
    planner, initial_plan_state = build_planner(Lplan, Larena) #instantiate planning module
    all_ts, all_as = zeros(batch, tmax), zeros(batch, tmax) #timepoints and actions

    for t = 1:tmax
        if t % 20 == 0 println("iteration $t of $tmax") end
        #copy over local variables
        agent_input = agent_input
        world_state = world_state
        rew = rew
        all_ts[:, t] = world_state.environment_state.time[:] #store environment time

        plan_bs = findall(exploit[:] .& just_planned) #episodes where I just did a rollout during exploitation
        Nps = length(plan_bs) #number of episodes to consider
        if (mode == "test") && Nps > 0.5 #if we're testing and there are episodes to compute PGs for

            logπ = agent_output[1:4, plan_bs]
            push!(all_pis, exp.(Float64.(logπ .- Flux.logsumexp(logπ; dims=1)))')

            newgs, newgs2 = [zeros(Nps, Nhidden, 4) for _ = 1:2] #instantiate gradients
            sim_as, sim_a2s = [zeros(Nps) .+ NaN for _ = 1:2] #instantiate array of rollout actions
            jacs, jacs_shift, jacs_shift2 = [zeros(Nps, Nhidden) for _ = 1:3] #instantiate jacobians

            for (ib, b) = enumerate(plan_bs) # for each episode where we planned

                sim_a = argmax(agent_input[Nin_base+1:Nin_base+4, b]) #rollout action
                sim_as[ib] = sim_a #store rollout action
                sim_a2 = argmax(agent_input[Nin_base+5:Nin_base+8, b]) #second rolled-out action
                shift_a, shift_a2 = Int(sim_a)%4 + 1, Int(sim_a2)%4 + 1 #construct 'wrong' actions
                @assert agent_input[Nin_base+sim_a, b] == 1 #check this action was rolled out

                #construct the input direction to compute gradients w.r.t.
                pert = zeros(Float32, Nin)
                pert[Nin_base+1:end] = meangv

                #delta vectors applying the change in simulated action
                shifta = zeros(Float32, Nin); shifta[Nin_base+sim_a] = -1f0; shifta[Nin_base+shift_a] = 1f0
                shifta2 = zeros(Float32, Nin); shifta2[Nin_base+4+sim_a2] = -1f0; shifta2[Nin_base+4+shift_a2] = 1f0

                #function that computes the RNN update step with a potential shift and input perturbation
                function fh(x; shift = zeros(Float32, Nin))
                    return m.network[GRUind].cell(h_rnn[:, b:b], agent_input[:, b:b]+shift+pert*x)[1]
                end
                # gradients of hidden state w.r.t. reward input
                jac = jacobian(fh, 0f0)[1] #gradient w.r.t the reward perturbation at zero perturbation
                jacs[ib, :] = jac #store jacobian

                # here we compute dh/dR
                fh_shift = (x -> fh(x, shift = shifta)) #function where a1 has been permuted
                jacs_shift[ib, :] = jacobian(fh_shift, 0f0)[1] #corresponding gradient
                fh_shift2 = (x -> fh(x, shift = shifta2)) #function where a2 has been permuted
                jacs_shift2[ib, :] = jacobian(fh_shift2, 0f0)[1] #corresponding gradient

                #function mapping hidden state to policy (for _old_ hidden state)
                function fp(x, a)
                    logπ = m.policy(x)[1:4] #log policy
                    return logπ[a] - Flux.logsumexp(logπ) #normalized log policy
                end

                for ia = 1:4 #for each action
                    fa(x) = fp(x, ia) #pi[a] as a function of h
                    gs = gradient(fa, h_rnn[:, b:b])[1][:] #dlogpi[a]/dh
                    newgs[ib, :, ia] = gs #save
                end

                if agent_input[Nin_base+4+sim_a2, b] == 1 #rollout contained at least two actions
                    sim_a2s[ib] = sim_a2 #store simulated action
                    #input on next iteration (current iteration in the outer loop)
                    nextinp = zeros(Float32, Nin); nextinp[:, 1] = agent_input[:, b]
                    nextinp[Nin_base+1:end] .= 0f0 #imagine that we took an action instead of a rollout
                    nextinp[Naction + 2, :] .+= (1f0-0.3f0) #increment time as if we took an action
                    nextinp[1:5] .= 0f0; nextinp[sim_a] = 1f0 #overwrite with imagined action (ahat_1)
                    newstate = update_agent_state(world_state.agent_state[:, b:b], nextinp[1:5, :], Larena) #state we would have been at
                    #imagined input if we took the imagined action and moved to the corresponding state
                    nextinp[(Naction + 3):(Naction + 2 + Nstates), :] = Float32.(onehot_from_state(Larena, newstate))

                    function f2(x, a)
                        #function outputting logpi[a2] as a function of h1 (i.e. through the network dynamics)
                        newh = m.network[GRUind].cell(x, nextinp)[1] #update as if we took simulated action instead of planning
                        logπ2 = m.policy(newh)[1:4] #log policy from this state
                        logπa2 = logπ2[a] - Flux.logsumexp(logπ2) #probability of taking a2 from this state
                        return logπa2
                    end

                    for ia = 1:4 #for each action
                        fa(x) = f2(x, ia) #construct function outputting policy for this action
                        gs = gradient(fa, h_rnn[:, b:b])[1][:] #compute gradient of log \pi(a) w.r.t. current hidden state
                        newgs2[ib, :, ia] = gs #store gradient
                    end
                end
            end

            #save all parameters for future use
            push!(all_sim_as, sim_as); push!(all_sim_a2s, sim_a2s)
            push!(all_jacs, jacs)
            push!(all_jacs_shift, jacs_shift); push!(all_jacs_shift2, jacs_shift2)
            push!(all_gs, newgs); push!(all_gs2, newgs2)
            
        end

        push!(full_inps, agent_input[Nin_base+1:end, :]) #store agent inputs
        h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn) #RNN step
        active = (world_state.environment_state.time .< (T+1 - 1e-2)) #active episodes
        teleport = (.~(active .& exploit .& (a[:] .> 4.5) .& (rew[:] .< 0.5))) #no rollouts if true
        all_as[:, t] = a[:] #store actual physical actions

        #perform rollouts and return states
        exploit[rew[:] .> 0.5] .= true #exploitation phase if we have found a reward
        just_planned = Bool.(zeros(batch)) #keep track of whether we just performed a rollout
        just_planned[ .~teleport ] .= true #just performed a rollout
        #take a step through the environment
        old_world_state = world_state # store this
        rew, agent_input, world_state, predictions = wall_environment.step(
                    agent_output, a, world_state, wall_environment.dimensions, m.model_properties, m, h_rnn
                )
        plan_states = world_state.planning_state.plan_cache #imagined rollout states

        rew[.~active] .= 0f0 #zero reward if finished episode
        push!(all_rews, rew) #store rewards

    end

    if mode == "R_tau" #if we're in the first phase of estimating R_tau
        rews = reduce(vcat, all_rews) #all rewards
        inps = reduce((x1, x2) -> cat(x1, x2, dims = 3), full_inps) #all rollout feedback
        inp_sums = sum(inps, dims = 1)[1, :, :] #total rollout feedback (just to check whether there is any)
        t_to_rew = zeros(size(rews)) .+ NaN #time to next reward
        for b = 1:batch #for each episode
            if sum(rews[:, b]) > 1.5 #if the agent finished at least two trials
                ks = findall(rews[:, b] .== 1) #find the iterations at which the agent got a reward
                k1, k2 = ks[1]+2, ks[end] 
                ts = all_ts[b, ks] #elapsed 'time' within the episode
                for k = k1:k2 #for each iteration between a1 of trial2 and the last time reward is found
                    dts = ts .- all_ts[b, k] #time between now and all rewards
                    t_to_rew[k, b] = dts[dts .>= 0][1] #time between now and next reward
                end
            end
        end
        inds = findall(.~isnan.(t_to_rew') .& (inp_sums .> 0.5)) #iterations where we performed a rollout and got a later reward
        X = inps[:, inds] #rollout feedback at these iterations
        y = t_to_rew'[inds] #time to reward at these iterations
        y = (-y) #negate  -> want a lower time
        beta = (X * X')^(-1) * X * y #regression coefficients
        beta[1:4] = beta[1:4] .- mean(beta[1:4])  #zero out effect of the first action DC mode since this is always 1 (this is the intercept)
        meangv = beta / sqrt(sum(beta.^2)) #normalize
    else #if we're in the test phase

        #stack our result arrays and store result
        arrs = [all_sim_as, all_sim_a2s, all_jacs, all_jacs_shift, all_jacs_shift2, all_gs, all_gs2, all_pis]
        labels = ["sim_as", "sim_a2s", "jacs", "jacs_shift", "jacs_shift2", "sim_gs", "sim_gs2", "all_pis"]
        cat_arrs = [reduce(vcat, arr) for arr = arrs]
        for arr_ind = 1:length(arrs) res_dict[seed][labels[arr_ind]] = cat_arrs[arr_ind] end

    end
end
end

# write our results to a file
@save datadir * "planning_as_pg.bson" res_dict

