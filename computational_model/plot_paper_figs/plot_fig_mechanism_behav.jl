include("plot_utils.jl")
using ToPlanOrNotToPlan
using BSON: @load
using Random, NaNStatistics, Statistics, Flux

cm = 1/2.54

bot, top = 0.0, 1.0
fig = figure(figsize = (17*cm, 3.0*cm))

### plot performance by # replays
@load "$datadir/perf_by_n.bson" res_dict
seeds = sort([k for k = keys(res_dict)])
Nseed = length(seeds)
ms1, ss2, ms2, ss2, bs, es1, es2 = [], [], [], [], [], [], []
for (is, seed) = enumerate(seeds)
    dts, mindists, policies = [res_dict[seed][k] for k = ["dts"; "mindists"; "policies"]]
    keepinds = findall((.~isnan.(sum(dts, dims = (1,3))[:])) .& (mindists[:, 2] .>= 0))
    new_dts = dts[:, keepinds, :]
    new_mindists = mindists[keepinds, 2]
    policies = policies[:, keepinds, :, :, :]
    m1, m2 = mean(new_dts[1,:,:], dims = 1)[:], mean(new_dts[2,:,:], dims = 1)[:]
    push!(ms1, m1); push!(ms2, m2); push!(bs, mean(new_mindists))
    p1, p2 = policies[1, :, :, :, :], policies[2, :, :, :, :]
    p1, p2 = [p .- Flux.logsumexp(p, dims = 4) for p = [p1, p2]]
    e1, e2 = [-sum(exp.(p) .* p, dims = 4)[:, :, :, 1] for p = [p1, p2]]
    m1, m2 = [mean(e[:,:,1], dims = 1)[:] for e = [e1,e2]] #only consider entropy of first action
    push!(es1, m1); push!(es2, m2)
end
ms1, ms2, es1, es2 = [reduce(hcat, arr) for arr = [ms1, ms2, es1, es2]]
m1, s1 = mean(ms1, dims = 2)[:], std(ms1, dims = 2)[:]/sqrt(Nseed)
m2, s2 = mean(ms2, dims = 2)[:], std(ms2, dims = 2)[:]/sqrt(Nseed)
me1, se1 = mean(es1, dims = 2)[:], std(es1, dims = 2)[:]/sqrt(Nseed)
me2, se2 = mean(es2, dims = 2)[:], std(es2, dims = 2)[:]/sqrt(Nseed)
nplans = (1:length(m1)) .- 1

grids = fig.add_gridspec(nrows=1, ncols=2, left=0.00, right=0.33, bottom = bot, top = 1.0, wspace=0.50)

### performance ###
ax = fig.add_subplot(grids[1,1])
ax.plot(nplans,m1, ls = "-", color = col_p, label = "agent")
ax.fill_between(nplans,m1-s1,m1+s1, color = col_p, alpha = 0.2)
plot([nplans[1]; nplans[end]], ones(2)*mean(bs), color = col_c, ls = "-", label = "optimal")
legend(frameon = false, loc = "upper right", fontsize = fsize_leg, handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.05)
xlabel("# rollouts")
ylabel("steps to goal")
ylim(0.9*mean(bs), maximum(m1+s1)+0.1*mean(bs))
xticks([0;5;10;15])

### entropy ###
ax = fig.add_subplot(grids[1,2])
ax.plot(nplans,me1, ls = "-", color = col_p, label = "agent")
ax.fill_between(nplans,me1-se1,me1+se1, color = col_p, alpha = 0.2)
plot([nplans[1]; nplans[end]], ones(2)*log(4), color = col_c, ls = "-", label = "uniform")
legend(frameon = false, fontsize = fsize_leg, handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.05)
xlabel("# rollouts")
ylabel("entropy (nats)", labelpad = 1)
ylim(0, 1.1*log(4))
xticks([0;5;10;15])
yticks([0; 1])

### performance with and without planning ###

@load "$(datadir)/performance_with_out_planning.bson" results
ress = zeros(length(seeds), 2)
for (i, plan) = enumerate([true; false])
    for (iseed, seed) = enumerate(seeds)
        rews = results[seed][plan]
        ress[iseed, i] = sum(rews) / size(rews, 1)
    end
end
m, s = mean(ress, dims = 1)[:], std(ress, dims = 1)[:]/sqrt(length(seeds))
println("performance with and without rollouts:")
println(m, " ", s)
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.405, right=0.505, bottom = 0, top = 1, wspace=0.15)
ax = fig.add_subplot(grids[1,1])
plot_comparison(ax, ress; xticklabs=["rollout", "no roll"], ylab = "avg. reward", col = col_p, col2 = 1.2*col_p, yticks = [6;7;8], rotation = 45)

### print performance with and without shuffled rollouts ###

# perform a simple analysis of the performance
@load "$(datadir)/performance_shuffled_planning.bson" results
ress = zeros(length(seeds), 2)
for (i, shuffle) = enumerate([true; false])
    for (iseed, seed) = enumerate(seeds)
        rews = results[seed][shuffle]
        ress[iseed, i] = sum(rews) / size(rews, 2)
    end
end
m, s = mean(ress, dims = 1)[:], std(ress, dims = 1)[:]/sqrt(length(seeds))
println("shuffled performance: ", m, " (", s, ")")

### plot example goal directed and non-goal directed rollouts ###
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.53, right=0.69, bottom = -0.03, top = 0.80, wspace=0.15)
ax = fig.add_subplot(grids[1,1])

arena = build_arena(4)
@load "$datadir/test_example_rollout.bson" store
plan_state, ps, ws, state = store[10]
rew_loc = state_from_onehot(arena, ps)

arena_lines(ps, ws, arena, rew=false, col="k", rew_col = "k", col_arena = "k", lw_arena = lw_arena, lw_wall = lw_wall)
lss = ["-"; "-"]
labels = ["successful"; "unsuccessful"]
lw_rep = linewidth
for i = 1:2
    col = [col_p1, col_p2][i]

    plan_state, ps, ws, state = store[[4;5][i]]
    plan_state = plan_state[1:sum(plan_state .> 0)]
    states = [state state_from_loc(arena, plan_state')]

    N = size(states, 2)

    for s = 1:(N-1)
        x1, y1, x2, y2 = [states[:, s]; states[:, s+1]]
        alph = 1
        if s == 1 label = labels[i] else label = nothing end
        if x1 == 4 && x2 == 1
            ax.plot([x1; 4.5], [y1; 0.5*(y1+y2)], ls = lss[i], color = col, alpha = alph, lw = lw_rep)
            ax.plot([0.5; x2], [0.5*(y1+y2); y2], ls = lss[i], color = col, alpha = alph, lw = lw_rep)
        elseif x1 == 1 && x2 == 4
            ax.plot([x2; 4.5], [y2; 0.5*(y1+y2)], ls = lss[i], color = col, alpha = alph, lw = lw_rep)
            ax.plot([0.5; x1], [0.5*(y1+y2); y1], ls = lss[i], color = col, alpha = alph, lw = lw_rep)
        elseif y1 == 4 && y2 == 1
            ax.plot([x1; 0.5*(x1+x2)], [y1; 4.5], ls = lss[i], color = col, alpha = alph, lw = lw_rep)
            ax.plot([0.5*(x1+x2); x2], [0.5; y2], ls = lss[i], color = col, alpha = alph, lw = lw_rep)
        elseif y1 == 1 && y2 == 4
            ax.plot([x2; 0.5*(x1+x2)], [y2; 4.5], ls = lss[i], color = col, alpha = alph, lw = lw_rep)
            ax.plot([0.5*(x1+x2); x1], [0.5; y1], ls = lss[i], color = col, alpha = alph, lw = lw_rep)
        else
            ax.plot([x1; x2], [y1; y2], ls = lss[i], color = col, alpha = alph, label = label, lw = lw_rep)
        end
    end

end

ax.scatter([state[1]], [state[2]], color = col_p, s = 150, zorder = 1000)
ax.plot([rew_loc[1]], [rew_loc[2]], color = "k", marker="x", markersize=12, ls="", mew=3)

ax.legend(frameon = false, fontsize = fsize_leg, loc = "upper center", bbox_to_anchor = (0.5, 1.28), handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.05)

### plot change in p(a) with planning ###

grids = fig.add_gridspec(nrows=1, ncols=2, left=0.78, right=1.00, bottom = 0, top = top, wspace=0.10)
for i = 1:2 #rewarded and non-rewarded sim
    ms = []
    for seed = seeds
        @load "$(datadir)/causal_$(seed)$(plan_epoch)_single.bson" data
        p_simulated_actions, p_simulated_actions_old = data["p_simulated_actions"], data["p_simulated_actions_old"]
        p_initial_sim, p_continue_sim = data["p_initial_sim"], data["p_continue_sim"]
        p_simulated_actions ./= (1 .- p_continue_sim)
        p_simulated_actions_old ./= (1 .- p_initial_sim)
        inds = findall(.~isnan.(sum(p_simulated_actions, dims = 1)[:])) #data for all
        #old and new probabilities
        push!(ms, [mean(p_simulated_actions_old[i, inds]); mean(p_simulated_actions[i, inds])])
    end
    ms = reduce(hcat, ms)
    m3, s3 = mean(ms, dims = 2)[1:2], std(ms, dims = 2)[1:2] / sqrt(length(seeds))

    ax = fig.add_subplot(grids[1,i])
    ax.bar(1:2, m3, yerr = s3, color = [col_p1, col_p2][i], capsize = capsize)
    if plot_points
        shifts = 1:size(ms, 2); shifts = (shifts .- mean(shifts))/std(shifts)*0.2
        ax.scatter([1 .+ shifts; 2 .+ shifts], [ms[1, :]; ms[2, :]], color = col_point, marker = ".", s = 15)
    end
    ax.set_xticks(1:2, ["pre"; "post"])
    #xticks([1; 2], ["+rew"; "~rew"])
    if i == 1
        ax.set_ylabel(L"$\pi(\hat{a}_1)$", labelpad = 0)
        ax.set_title("succ.", fontsize = fsize)
        ax.set_yticks([0.1;0.3;0.5;0.7])
    else
        ax.set_title("unsucc.", fontsize = fsize)
        ax.set_yticks([])
    end
    ax.set_ylim(0.0, 0.8)
    ax.set_xlim([0.4; 2.6])
    ax.axhline(0.25, color = color = col_c, ls = "-")
end


### add labels and save ###

add_labels = true
if add_labels
    y1 = 1.16
    y2 = 0.46
    x1, x2, x3, x4, x5 = -0.05, 0.15, 0.34, 0.51, 0.71
    fsize = fsize_label
    plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
    plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x3,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x4,y1,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x5,y1,"E";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
end

savefig("./figs/fig_mechanism_behav.pdf", bbox_inches = "tight")
savefig("./figs/fig_mechanism_behav.png", bbox_inches = "tight")
close()


