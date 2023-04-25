
include("anal_utils.jl")
using ToPlanOrNotToPlan
using Random
using BSON: @save, @load

seeds = 61:65
greedy_actions = true
epoch = plan_epoch
prefix = ""
N = 100; Lplan = 8

function run_perf_by_plan_number(;seeds, N, Lplan, epoch, prefix = "")

loaddir = "../models/"
res_dict = Dict()

for seed = seeds
println("\n new seed $(seed)!")
res_dict[seed] = Dict()
filename = prefix*"N$(N)_T50_Lplan$(Lplan)_seed$(seed)_$epoch"
network, opt, store, hps, policy, prediction = recover_model(loaddir*filename)

Larena = hps["Larena"]
model_properties, wall_environment, model_eval = build_environment(
    Larena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = true
)
m = ModularModel(model_properties, network, policy, prediction, forward_modular)


# run a handful of steps
batch = 1
ed = wall_environment.dimensions
Nstates, Naction = ed.Nstates,  ed.Naction
Nin_base = Naction + 1 + 1 + Nstates + 2 * Nstates
Nhidden = m.model_properties.Nhidden

tmax = 50
Lplan = model_properties.Lplan
nreps = 1000
nplans = 0:15
dts = zeros(2, nreps, length(nplans)) .+ NaN;
policies = zeros(2, nreps, length(nplans), 10, 5) .+ NaN;
mindists = zeros(nreps, length(nplans));

for ictrl = [1;2] #plan input or not
    for nrep = 1:nreps
        if nrep % 100 == 0 println(nrep) end
        for (iplan, nplan) = enumerate(nplans)
            Random.seed!(nrep)
            world_state, agent_input = wall_environment.initialize(
                zeros(2), zeros(2), batch, m.model_properties
            )
            agent_state = world_state.agent_state
            h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch)) #expand hidden state
            exploit = Bool.(zeros(batch))
            rew = zeros(batch)
            if iplan == 1
                ps, ws = world_state.environment_state.reward_location, world_state.environment_state.wall_loc
                global dists = dist_to_rew(ps, ws, Larena)
            end

            tot_n = nplan
            t = 0
            finished = false
            nact = 0
            while ~finished
                t += 1
                agent_input, world_state, rew = agent_input, world_state, rew
                if ictrl == 2 agent_input[Nin_base+1:end, :] .= 0 end #no planning input
                h_rnn, agent_output, a = m.forward(m, ed, agent_input, h_rnn) #RNN step

                plan = false
                if exploit[1] && (tot_n > 0.5) #exploitation phase and more plans to go
                    plan, tot_n = true, tot_n-1 #do a plan
                    if tot_n  == 0
                        state = world_state.agent_state[:]
                        mindists[nrep, iplan] = dists[state[1], state[2]]
                    end
                end
                if plan
                    a[1] = 5 #perform a rollout
                elseif exploit[1] #exploitation phase
                    nact += 1 #action number
                    a[1] = argmax(agent_output[1:4, 1]) #greedy
                    if nact <= 10 policies[ictrl, nrep, iplan, nact, :] = agent_output[1:5, 1] end #store policy
                else
                    a[1] = argmax(agent_output[1:5, 1]) #greedy and don't store anything
                end

                exploit[rew[:] .> 0.5] .= true #exploitation phase if we have found reward
                rew, agent_input, world_state, predictions = wall_environment.step(
                            agent_output, a, world_state, wall_environment.dimensions, m.model_properties, m, h_rnn
                        )

                if rew[1] > 0.5
                    if exploit[1] #already found reward before
                        finished = true
                        dts[ictrl, nrep, iplan] = t - t1 - 1 - nplan
                    else
                        global t1 = t #reset timer at the end of first trial
                    end
                end
                if t > tmax finished = true end #impose maximum time steps
            end
        end
    end
end

#store some data
res_dict[seed]["dts"] = dts
res_dict[seed]["mindists"] = mindists
res_dict[seed]["nplans"] = nplans
res_dict[seed]["policies"] = policies

end

savename = "$(prefix)N$(N)_Lplan$(Lplan)"
@save datadir * "perf_by_n_$(savename).bson" res_dict

end

run = true
run && run_perf_by_plan_number(;seeds,N,Lplan,epoch)
