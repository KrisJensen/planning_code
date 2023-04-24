
include("anal_utils.jl")
using ToPlanOrNotToPlan
using NaNStatistics, MultivariateStats, Flux, PyCall, PyPlot, Random, Statistics
using BSON: @save, @load

seeds = 61:65
greedy_actions = true
seed = 61
epoch = plan_epoch
prefix = ""

function run_perf_by_plan_number(;seeds, prefix, N, Lplan, epoch, greedy_actions)

Save = true
loaddir = "../models/maze/"
resdir, figdir = "./results/", "../figs/maze/"
res_dict = Dict()

for seed = seeds
println("\n new seed $(seed)!")
res_dict[seed] = Dict()
filename = "$loaddir/$(prefix)N$(N)_T50_seed$(seed)_Lplan$(Lplan)_$epoch"
network, opt, store, hps, policy, prediction = recover_model(filename)

Larena = hps["Larena"]
model_properties, wall_environment, model_eval = build_environment(
    arena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions
)
m = ModularModel(model_properties, network, policy, prediction, forward_modular)

# run a handful of steps
batch = 1
ed = wall_environment.dimensions
Nstates, Naction, T = ed.Nstates,  ed.Naction, ed.T
Nin_base = Naction + 1 + 1 + Nstates + 2 * Nstates
Nout, Nhidden = m.model_properties.Nout, m.model_properties.Nhidden

tmax = 50
Lplan = model_properties.Lplan
nreps = 5000
nreps = 1000
nplans = 0:15
dts = zeros(2, nreps, length(nplans)) .+ NaN
policies = zeros(2, nreps, length(nplans), 10, 5) .+ NaN
meandists = zeros(nreps)
mindists = zeros(nreps, length(nplans))
Print = false
Nin_base = Naction + 1 + 1 + Nstates + 2 * Nstates

for ictrl = [1;2] #plan input or not
    Random.seed!(2)
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
                meandists[nrep] = sum(dists) / (Nstates - 1)
            end

            tot_n = nplan
            t = 0
            finished = false
            nact = 0
            while ~finished
                t += 1
                Print && println(t)
                agent_input, world_state, rew = agent_input, world_state, rew
                if ictrl == 2 agent_input[Nin_base+1:end, :] .= 0 end #no planning input
                h_rnn, agent_output, a = m.forward(m, ed, agent_input; h_rnn=h_rnn) #RNN step agent_output_t = (pi_t(s_t), V_t(s_t)) (nout x batch)

                plan = false
                if exploit[1] && (tot_n > 0.5) #exploitation phase and more plans to go
                    plan, tot_n = true, tot_n-1 #do a plan
                    Print && println("planning! ", t, " ", tot_n+1)
                    if tot_n  == 0
                        state = world_state.agent_state[:]
                        mindists[nrep, iplan] = dists[state[1], state[2]]
                    end
                end
                if plan
                    a[1] = 5
                elseif exploit[1]
                    nact += 1 #action number
                    a[1] = argmax(agent_output[1:4, 1]) #greedy
                    if nact <= 10 policies[ictrl, nrep, iplan, nact, :] = agent_output[1:5, 1] end
                else
                    a[1] = argmax(agent_output[1:5, 1])
                end

                exploit[rew[:] .> 0.5] .= true #exploitation phase
                rew, agent_input, world_state, predictions = wall_environment.step(
                            agent_output, a, world_state, wall_environment.dimensions, m.model_properties, m, h_rnn#, plan = (planning_state, plan_inds)
                        )
                if rew[1] > 0.5
                    Print && println("found rew!!!")
                    if exploit[1] #already found reward before
                        finished = true
                        dts[ictrl, nrep, iplan] = t - t1 - 1 - nplan
                    else
                        Print && println("setting t1")
                        global t1 = t
                    end
                end
                if t > tmax finished = true end
            end
        end
    end
end

res_dict[seed]["dts"] = dts
res_dict[seed]["mindists"] = mindists
res_dict[seed]["nplans"] = nplans
res_dict[seed]["policies"] = policies

keepinds = findall(.~isnan.(sum(dts, dims = (1,3))[:]))
#keepinds = findall(.~isnan.(sum(dts, dims = 2)[:]) .& (mindists[:, 2] .== 7))

new_dts = dts[:, keepinds, :]
new_mindists = mindists[keepinds, 2]
ms1, ss1 = mean(new_dts[1,:,:], dims = 1)[:], std(new_dts[1,:,:], dims = 1)[:]/sqrt(length(keepinds))
ms2, ss2 = mean(new_dts[2,:,:], dims = 1)[:], std(new_dts[2,:,:], dims = 1)[:]/sqrt(length(keepinds))
baseline = mean(new_mindists)
println(baseline)
println(ms1); println(ss1)

plot_dyn = true
figure(figsize = (4, 3))
plot_dyn && plot(nplans,ms2, "b-", label = "dynamics")
plot_dyn && fill_between(nplans,ms2-ss2,ms2+ss2, color = "b", alpha = 0.2)
plot(nplans,ms1, "k-", label = "agent")
fill_between(nplans,ms1-ss1,ms1+ss1, color = "k", alpha = 0.2)
plot([nplans[1]; nplans[end]], ones(2)*baseline, color = ones(3)*0.5, ls = "-", label = "optimal")
legend(frameon = false)
xlabel("# plans")
ylabel("steps to goal")
ylim(0, 1.1*maximum(ms1+ss1))
savefig("$figdir/model/perf_by_nplan.png", bbox_inches = "tight")
close()

end

savename = "$(prefix)N$(N)_Lplan$(Lplan)"
@save resdir * "perf_by_n_$(savename).bson" res_dict


end

run = true
run && run_perf_by_plan_number(;seeds,N,Lplan,epoch,greedy_actions)
