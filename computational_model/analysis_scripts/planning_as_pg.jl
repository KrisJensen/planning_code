include("anal_utils.jl")
using Revise
using ToPlanOrNotToPlan
using NaNStatistics, MultivariateStats, Flux, PyCall, PyPlot, Random, Statistics, Zygote
using BSON: @save, @load

epoch = plan_epoch
loss_hp = LossHyperparameters(0, 0, 0, 0, 0, 0, 1000, true, 0f0, () -> ())
Save = true
greedy_actions = true

loaddir = "../models/maze/"
resdir, figdir = "./results/", "../figs/maze/"

res_dict = Dict()

for seed = seeds
println("\n new seed $(seed)!")
res_dict[seed] = Dict()
filename = "$loaddir/N100_T50_seed$(seed)_Lplan8_$epoch"
network, opt, store, hps, policy, prediction = recover_model(filename)
Larena = hps["Larena"]
model_properties, wall_environment, model_eval = build_environment(
    arena, hps["Nhidden"], hps["T"], Lplan = hps["Lplan"], greedy_actions = greedy_actions
)
m = ModularModel(model_properties, network, policy, prediction, forward_modular)
Nstates = Larena^2
Naction = wall_environment.properties.dimensions.Naction
Nin_base = Naction + 1 + 1 + Nstates + 2 * Nstates
Nout, Nhidden, Nin = m.model_properties.Nout, m.model_properties.Nhidden, m.model_properties.Nin
Lplan = model_properties.Lplan
 
all_cosses, all_cosses2, all_sim_as, all_sim_a2s = [], [], [], []
all_jacs, all_jacs_shift, all_jacs_shift2 = [], [], []
all_gs, all_gs2 = [], []
all_reprews = []
all_h0s, all_h1s = [], []
all_gvs, all_inps, all_Vs = [], [], []
full_inps = []
meangv = []

for (i_mode, mode) = enumerate(["R_tau", "test"]) #first estimate the direction of R_tau, then actual test
    all_rews = []

    # run a handful of steps
    batch = 1002
    Random.seed!(2)
    ed = wall_environment.properties.dimensions
    Nstates, Naction, T = ed.Nstates,  ed.Naction, ed.T
    world_state, agent_input = wall_environment.initialize(
        zeros(2), zeros(2), batch, m.model_properties
    )
    agent_state = world_state.agent_state
    h_rnn = m.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch)) #expand hidden state
    rew = zeros(batch)

    exploit, just_planned = Bool.(zeros(batch)), Bool.(zeros(batch))
    tmax = 50
    planner, initial_plan_state = build_planner(Lplan, Larena)
    rewlocs = [argmax(world_state.environment_state.reward_location[:, i]) for i = 1:batch]
    all_ts, all_as = zeros(batch, tmax), zeros(batch, tmax)
    t = 0

    for t = 1:tmax
        #t += 1
        agent_input = agent_input
        world_state = world_state
        rew = rew
        all_ts[:, t] = world_state.environment_state.time[:]

        plan_bs = findall(exploit[:] .& just_planned) #if I just did a plan
        Nps = length(plan_bs)
        #plan_bs = []
        if Nps > 0.5
            cosses, cosses2 = [zeros(Nps, 4) for _ = 1:2]
            newgs, newgs2 = [zeros(Nps, Nhidden, 4) for _ = 1:2]
            sim_as, sim_a2s = [zeros(Nps) .+ NaN for _ = 1:2]
            jacs, h0s, h1s = [zeros(Nps, Nhidden) for _ = 1:3]
            jacs_shift, jacs_shift2 = [zeros(Nps, Nhidden) for _ = 1:2]
            reprews = zeros(Nps); gvs = zeros(Nps, Nin-Nin_base); inps = zeros(Nps, Nin); vs = zeros(Nps)
            for (ib, b) = enumerate(plan_bs)
                #println(ib, " ", b)
                #path = reshape(agent_input[Nin_base+1:Nin-2, b], 4, 8)
                sim_a = argmax(agent_input[Nin_base+1:Nin_base+4, b]); sim_as[ib] = sim_a
                sim_a2 = argmax(agent_input[Nin_base+5:Nin_base+8, b])
                shift_a, shift_a2 = Int(sim_a)%4 + 1, Int(sim_a2)%4 + 1 #'wrong' action
                @assert agent_input[Nin_base+sim_a, b] == 1

                #find the reward input dimension as dV/dxplan
                function fv(x)
                    vec = [zeros(Float32, Nin_base); x]
                    newh = m.network[GRUind].cell(h_rnn[:, b:b], agent_input[:, b:b]+vec)[1]
                    V = m.policy(newh)[6]
                end
                if mode == "R_tau" #find R_tau direction
                    newgv = gradient(fv, zeros(Float32, Nin-Nin_base))[1]

                    newgv = zeros(Float32, Nin-Nin_base)
                    for i = 1:(Nin-Nin_base) #finite difference 1 minus 0
                        pert = zeros(Float32, Nin-Nin_base); pert[i] = 1f0
                        ind = Nin+i
                        if agent_input[ind] == 1
                            newgv[i] = fv(0f0*pert) - fv(-pert) #1 minus 0 everything else constant
                        else
                            newgv[i] = fv(pert) - fv(0f0*pert) #1 minus 0
                        end
                    end

                    gvs[ib, :] = newgv; vs[ib] = fv(zeros(Float32, Nin-Nin_base));
                    inps[ib, :] = agent_input[:, b:b]
                else

                    pert = zeros(Float32, Nin)
                    pert[Nin_base+1:end] = meangv

                    shifta = zeros(Float32, Nin); shifta[Nin_base+sim_a] = -1f0; shifta[Nin_base+shift_a] = 1f0
                    shifta2 = zeros(Float32, Nin); shifta2[Nin_base+4+sim_a2] = -1f0; shifta2[Nin_base+4+shift_a2] = 1f0

                    fh(x; shift = zeros(Float32, Nin)) = m.network[GRUind].cell(h_rnn[:, b:b], agent_input[:, b:b]+shift+pert*x)[1]
                    # gradients of hidden state w.r.t. reward input
                    jac = jacobian(fh, 0f0)[1]
                    jacs[ib, :] = jac

                    fh_shift = (x -> fh(x, shift = shifta))
                    jacs_shift[ib, :] = jacobian(fh_shift, 0f0)[1]
                    fh_shift2 = (x -> fh(x, shift = shifta2))
                    jacs_shift2[ib, :] = jacobian(fh_shift2, 0f0)[1]

                    h0s[ib, :] = h_rnn[:, b]
                    h1s[ib, :] = fh(0)
                    reprews[ib] = agent_input[end, b]

                    #function mapping hidden state to policy
                    function fp(x, a) logπ = m.policy(x)[1:4]; logπ[a] - Flux.logsumexp(logπ) end
                    for ia = 1:4
                        fa(x) = fp(x, ia)
                        # gradient of policy w.r.t hidden state
                        gs = gradient(fa, h_rnn[:, b:b])[1][:]
                        cosθ = sum(jac .* gs) / sqrt(sum(jac.^2) * sum(gs.^2))
                        cosses[ib, ia] = cosθ
                        newgs[ib, :, ia] = gs
                    end

                    if agent_input[Nin_base+4+sim_a2, b] == 1 #simulated two actions
                        #effect on next action - construct input
                        sim_a2s[ib] = sim_a2
                        nextinp = zeros(Float32, Nin); nextinp[:, 1] = agent_input[:, b]
                        nextinp[Nin_base+1:end] .= 0f0; nextinp[Naction + 2, :] .+= (1f0-0.3f0) #no planning and increment time
                        nextinp[1:5] .= 0f0; nextinp[sim_a] = 1f0 #previous action
                        newstate = update_agent_state(world_state.agent_state[:, b:b], nextinp[1:5, :], Larena) 
                        nextinp[(Naction + 3):(Naction + 2 + Nstates), :] = Float32.(onehot_from_state(Larena, newstate))
                        function f2(x, a)
                            #logπ1 = m.policy(x)[1:4]; pia1 = logπ1[sim_a] - Flux.logsumexp(logπ1)
                            newh = m.network[GRUind].cell(x, nextinp)[1] #update as if we took simulated action instead of planning
                            logπ2 = m.policy(newh)[1:4]; pia2 = logπ2[a] - Flux.logsumexp(logπ2)
                            return pia2 #+ pia1
                        end
                        for ia = 1:4
                            fa(x) = f2(x, ia)
                            gs = gradient(fa, h_rnn[:, b:b])[1][:]
                            cosses2[ib, ia] = sum(jac .* gs) / sqrt(sum(jac.^2) * sum(gs.^2))
                            newgs2[ib, :, ia] = gs
                        end
                    end
                end
            end

            if mode == "R_tau"
                push!(all_gvs, gvs); push!(all_inps, inps); push!(all_Vs, vs)
            else
                push!(all_cosses, cosses); push!(all_cosses2, cosses2)
                push!(all_sim_as, sim_as); push!(all_sim_a2s, sim_a2s)
                push!(all_jacs, jacs); push!(all_h0s, h0s); push!(all_h1s, h1s)
                push!(all_jacs_shift, jacs_shift); push!(all_jacs_shift2, jacs_shift2)
                push!(all_gs, newgs); push!(all_gs2, newgs2); push!(all_reprews, reprews)
            end
            
        end

        push!(full_inps, agent_input[Nin_base+1:end, :])
        h_rnn, agent_output, a = m.forward(m, ed, agent_input; h_rnn=h_rnn) #RNN step agent_output_t = (pi_t(s_t), V_t(s_t)) (nout x batch)
        active = (world_state.environment_state.time .< (T+1 - 1e-2)) #active heads
        teleport = (.~(active .& exploit .& (a[:] .> 4.5) .& (rew[:] .< 0.5))) #whether to plan
        all_as[:, t] = a[:]

        ###run planning and return states!!!
        exploit[rew[:] .> 0.5] .= true #exploitation phase
        just_planned = Bool.(zeros(batch))
        just_planned[ .~teleport ] .= true #just planned with no reward
        rew, agent_input, world_state, predictions = wall_environment.step(
                    agent_output, a, world_state, wall_environment.properties, m.model_properties, m, h_rnn
                )
        path = reshape(world_state.planning_state.plan_input[1:(4*Lplan), :], 4, Lplan, batch)
        plan_states = world_state.planning_state.plan_cache

        rew[.~active] .= 0f0
        push!(all_rews, rew)

        πt = exp.(agent_output[1:5, :])
        println(t, " ", mean(maximum(πt, dims = 1)))

    end

    if mode == "R_tau" #want to know reward density
        rews = reduce(vcat, all_rews)
        inps = reduce((x1, x2) -> cat(x1, x2, dims = 3), full_inps)
        inp_sums = sum(inps, dims = 1)[1, :, :]
        t_to_rew = zeros(size(rews)) .+ NaN
        for b = 1:batch
            if sum(rews[:, b]) > 1.5
                ks = findall(rews[:, b] .== 1)
                k1, k2 = ks[1]+2, ks[end]
                ts = all_ts[b, ks] #real time
                for k = k1:k2 #for each completed trial
                    dts = ts .- all_ts[b, k]
                    t_to_rew[k, b] = dts[dts .>= 0][1]
                    #t_to_rew[t, b] = (ts .- t)[(ts .- t) .>= 0][1]
                end
            end
        end
        inds = findall(.~isnan.(t_to_rew') .& (inp_sums .> 0.5))
        X = inps[:, inds]
        y = t_to_rew'[inds]
        y = 0.0*mean(y) .- y
        beta = (X * X')^(-1) * X * y
        beta[1:4] = beta[1:4] .- mean(beta[1:4])
        meangv = beta / sqrt(sum(beta.^2))
        #meangv = zeros(length(beta)); meangv[end] = 1
    else
        wall_loc, ps = world_state.environment_state.wall_loc, world_state.environment_state.reward_location
        println(mean(sum(all_rews)))
        rews = reduce(vcat, all_rews)'
        arrs = [all_cosses, all_cosses2, all_sim_as, all_sim_a2s, all_jacs, all_jacs_shift, all_jacs_shift2, all_gs, all_gs2, all_h0s, all_h1s, all_reprews, all_gvs]
        cat_cosses, cat_cosses2, all_sim_as, all_sim_a2s, all_jacs, all_jacs_shift, all_jacs_shift2, all_gs, all_gs2, all_h0s, all_h1s, all_reprews, all_gvs = [reduce(vcat, arr) for arr = arrs]
        
        res_dict[seed]["cosses"] = cat_cosses; res_dict[seed]["cosses2"] = cat_cosses2
        res_dict[seed]["sim_as"] = all_sim_as; res_dict[seed]["sim_a2s"] = all_sim_a2s
        res_dict[seed]["jacs"] = all_jacs; res_dict[seed]["sim_gs"] = all_gs; res_dict[seed]["sim_gs2"] = all_gs2
        res_dict[seed]["h0s"] = all_h0s; res_dict[seed]["h1s"] = all_h1s; res_dict[seed]["rews"] = all_reprews
        res_dict[seed]["gvs"] = all_gvs
        res_dict[seed]["jacs_shift"] = all_jacs_shift; res_dict[seed]["jacs_shift2"] = all_jacs_shift2;

        for (ia, cosses) = enumerate([cat_cosses, cat_cosses2])
            if ia == 1 cosses = [cosses[i, Int(all_sim_as[i])] for i = 1:length(all_sim_as)] end
            if ia == 2
                cosses = [cosses[i, Int(all_sim_as[i])] for i = findall(.~isnan.(all_sim_a2s))]
            end
            println(ia, " ", mean(cosses, dims = 1), " ", std(cosses, dims = 1)/sqrt(size(cosses, 1)))
            bins = -0.31:0.02:0.31
            figure(figsize = (4, 3))
            hist(cosses, color = col_p, alpha = 0.6, bins = bins)
            #hist(cosses[:, 2], color = ones(3)*0.4, alpha = 0.6, bins = bins)
            axvline(mean(cosses), color = col_p, label = "sim. a")
            #axvline(mean(cosses[:, 2]), color = ones(3)*0.4, label = "ctrl a")
            xlabel(L"$\cos \theta$")
            ylabel("frequency")
            #legend(frameon = false, fontsize = 14)
            savefig("$figdir/plan_state_angles$ia.png", bbox_inches = "tight")
            close()
        end

    end

end

end

@save resdir * "planning_as_pg_new.bson" res_dict

