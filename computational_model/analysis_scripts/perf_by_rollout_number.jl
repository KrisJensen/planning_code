# in this script, we analyse how model performance depends on the number of rollouts
# this allow us to investigate whether rollouts improve the policy

# load some packages
include("anal_utils.jl")
using ToPlanOrNotToPlan

"""
    run_perf_by_plan_number(;seeds, N, Lplan, epoch, prefix = "")
analyses the performance (in terms of steps to goal) on trial 2
as a function of the number of enforced rollouts.
"""
function run_perf_by_rollout_number(;seeds, N, Lplan, epoch, prefix = "")
println("quantifying trial 2 performance by number of rollouts")

res_dict = Dict() #dictionary to store results

for seed = seeds #iterate through random seeds
println("\n new seed $(seed)!")
res_dict[seed] = Dict() #results for this seed
filename = prefix*"N$(N)_T50_Lplan$(Lplan)_seed$(seed)_$epoch" #model to load
network, opt, store, hps, policy, prediction = recover_model(loaddir*filename) #load model parameters

Larena = hps["Larena"] #size of the arena
model_properties, wall_environment, model_eval = build_environment(
    Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions
) #construct RL environment
m = ModularModel(model_properties, network, policy, prediction, forward_modular) #construct model

# set some parameters
batch = 1 #one batch at a time
ed = wall_environment.dimensions
Nstates, Naction = ed.Nstates,  ed.Naction
Nin_base = Naction + 1 + 1 + Nstates + 2 * Nstates #'physical' input dimensions
Nhidden = m.model_properties.Nhidden
tmax = 50
Lplan = model_properties.Lplan
nreps = 1000 #number of random environments to consider
nplans = 0:15 #number of plans enforced
dts = zeros(2, nreps, length(nplans)) .+ NaN; #time to goal
policies = zeros(2, nreps, length(nplans), 10, 5) .+ NaN; #store policies
mindists = zeros(nreps, length(nplans));

for ictrl = [1;2] #plan input or not (zerod out)
    for nrep = 1:nreps #for each repetition
        if nrep % 100 == 0 println(nrep) end
        for (iplan, nplan) = enumerate(nplans) #for each number of rollouts enforced
            Random.seed!(nrep) #set random seed for consistent environment across #rollouts
            world_state, agent_input = wall_environment.initialize(
                zeros(2), zeros(2), batch, m.model_properties
            ) #initialize environment
            agent_state = world_state.agent_state #initialize agent location
            h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch)) #expand hidden state
            exploit = Bool.(zeros(batch)) #keep track of exploration bs exploitation
            rew = zeros(batch) #keep track of reward
            if iplan == 1
                ps, ws = world_state.environment_state.reward_location, world_state.environment_state.wall_loc
                global dists = dist_to_rew(ps, ws, Larena) #compute distances to goal for this arena
            end

            tot_n = nplan #rollouts to go
            t = 0 #timestep
            finished = false #have we finished this environment
            nact = 0 #number of physical actions
            while ~finished #until finished
                t += 1 #update iteration number
                agent_input, world_state, rew = agent_input, world_state, rew
                if ictrl == 2 agent_input[Nin_base+1:end, :] .= 0 end #no planning input if ctrl
                h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn) #RNN step

                plan = false #have we just performed a rollout
                if exploit[1] && (tot_n > 0.5) #exploitation phase and more plans to go
                    plan, tot_n = true, tot_n-1 #we will perform a rollout; decrease counter-to-go
                    if tot_n  == 0 #if we're done
                        state = world_state.agent_state[:] #current location
                        mindists[nrep, iplan] = dists[state[1], state[2]] #distance to goal
                    end
                end
                if plan #we need to do a rollout
                    a[1] = 5 #perform a rollout
                elseif exploit[1] #exploitation phase
                    nact += 1 #increment action number to goal
                    a[1] = argmax(agent_output[1:4, 1]) #greedy action selection
                    if nact <= 10 policies[ictrl, nrep, iplan, nact, :] = agent_output[1:5, 1] end #store policy
                else #exploration phase (trial 1)
                    a[1] = argmax(agent_output[1:5, 1]) #greedy and don't store anything
                end

                exploit[rew[:] .> 0.5] .= true #we're in the exploitation phase if we have found reward
                #pass action to environment
                rew, agent_input, world_state, predictions = wall_environment.step(
                            agent_output, a, world_state, wall_environment.dimensions, m.model_properties, m, h_rnn
                        )

                if rew[1] > 0.5 # if we just found a reward
                    if exploit[1] #already found reward before
                        finished = true #now finished since we only consider trial 2
                        dts[ictrl, nrep, iplan] = t - t1 - 1 - nplan #store the number of actions to goal
                    else #first time
                        global t1 = t #reset timer at the end of first trial
                    end
                end
                if t > tmax finished = true end #impose maximum time steps
            end
        end
    end
end

#store some data for this seed
res_dict[seed]["dts"] = dts
res_dict[seed]["mindists"] = mindists
res_dict[seed]["nplans"] = nplans
res_dict[seed]["policies"] = policies

end

#save our data
savename = "$(prefix)N$(N)_Lplan$(Lplan)"
@save datadir * "perf_by_n_$(savename).bson" res_dict

end

#run_default_analyses is a global parameter in anal_utils.jl
run_default_analyses && run_perf_by_rollout_number(;seeds,N,Lplan,epoch)
