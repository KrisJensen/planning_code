#This script plots Figure 3 of Jensen et al.

include("plot_utils.jl")
using ToPlanOrNotToPlan
using Flux

bot, top = 0.0, 1.0
fig = figure(figsize = (17*cm, 3.0*cm))

#plot performance and entropy by number of rollouts

#start by loading and extracting data
@load "$datadir/perf_by_n_N100_Lplan8.bson" res_dict
seeds = sort([k for k = keys(res_dict)])
Nseed = length(seeds)
ms1, ms2, bs, es1, es2 = [], [], [], [], []
dists = 1:6; bydist = []
for (is, seed) = enumerate(seeds) #for each model
    #time within trial, distance to goal, and policy
    dts, mindists, policies = [res_dict[seed][k] for k = ["dts"; "mindists"; "policies"]]
    #select episodes where the trial finished for all rollout numbers
    keepinds = findall((.~isnan.(sum(dts, dims = (1,3))[:])) .& (mindists[:, 2] .>= 0))
    new_dts = dts[:, keepinds, :]
    new_mindists = mindists[keepinds, 2]
    policies = policies[:, keepinds, :, :, :]
    #mean performance across episodes with (m1) and without (m2) rollout feedback
    m1, m2 = mean(new_dts[1,:,:], dims = 1)[:], mean(new_dts[2,:,:], dims = 1)[:]
    push!(bydist, reduce(vcat, [mean(new_dts[1,new_mindists .== dist,:], dims = 1) for dist = dists]))
    push!(ms1, m1); push!(ms2, m2); push!(bs, mean(new_mindists)) #also store optimal (bs))
    p1, p2 = policies[1, :, :, :, :], policies[2, :, :, :, :] #extract log policies
    p1, p2 = [p .- Flux.logsumexp(p, dims = 4) for p = [p1, p2]] #normalize
    e1, e2 = [-sum(exp.(p) .* p, dims = 4)[:, :, :, 1] for p = [p1, p2]] #entropy
    m1, m2 = [mean(e[:,:,1], dims = 1)[:] for e = [e1,e2]] #only consider entropy of first action
    push!(es1, m1); push!(es2, m2) #store entropies
end
bydist = reduce((a,b) -> cat(a, b, dims = 3), bydist)
bydist = mean(bydist, dims = 3)[:, :, 1]
#concatenate across seeds
ms1, ms2, es1, es2 = [reduce(hcat, arr) for arr = [ms1, ms2, es1, es2]]
# compute mean and std across seeds
m1, s1 = mean(ms1, dims = 2)[:], std(ms1, dims = 2)[:]/sqrt(Nseed)
m2, s2 = mean(ms2, dims = 2)[:], std(ms2, dims = 2)[:]/sqrt(Nseed)
me1, se1 = mean(es1, dims = 2)[:], std(es1, dims = 2)[:]/sqrt(Nseed)
me2, se2 = mean(es2, dims = 2)[:], std(es2, dims = 2)[:]/sqrt(Nseed)
nplans = (1:length(m1)) .- 1 #

# plot performance vs number of rollouts
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.00, right=0.33, bottom = bot, top = 1.0, wspace=0.50)
ax = fig.add_subplot(grids[1,1])
ax.plot(nplans,m1, ls = "-", color = col_p, label = "agent") #mean
ax.fill_between(nplans,m1-s1,m1+s1, color = col_p, alpha = 0.2) #standard error
plot([nplans[1]; nplans[end]], ones(2)*mean(bs), color = col_c, ls = "-", label = "optimal") #optimal baseline
ax.plot(nplans,m2, ls = ":", color = col_c, label = "ctrl") #mean
ax.fill_between(nplans,m2-s2,m2+s2, color = col_c, alpha = 0.2) #standard error
legend(frameon = false, loc = "upper right", fontsize = fsize_leg, handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.05)
xlabel("# rollouts")
ylabel("steps to goal")
ylim(0.9*mean(bs), maximum(m1+s1)+0.1*mean(bs))
xticks([0;5;10;15])

# plot entropy vs number of rollouts
ax = fig.add_subplot(grids[1,2])
ax.plot(nplans,me1, ls = "-", color = col_p, label = "agent") #mean
ax.fill_between(nplans,me1-se1,me1+se1, color = col_p, alpha = 0.2) #standard error
plot([nplans[1]; nplans[end]], ones(2)*log(4), color = col_c, ls = "-", label = "uniform") #entropy of uniform policy
legend(frameon = false, fontsize = fsize_leg, handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.05)
xlabel("# rollouts")
ylabel("entropy (nats)", labelpad = 1)
ylim(0, 1.1*log(4))
xticks([0;5;10;15])
yticks([0; 1])

# plot performance with and without rollouts

# load data across different random seeds
@load "$(datadir)/performance_with_out_planning.bson" results
ress = zeros(length(seeds), 2)
for (i, plan) = enumerate([true; false])
    for (iseed, seed) = enumerate(seeds)
        rews = results[seed][plan]
        ress[iseed, i] = sum(rews) / size(rews, 1)
    end
end
m, s = mean(ress, dims = 1)[:], std(ress, dims = 1)[:]/sqrt(length(seeds)) # mean and sem across seeds
println("performance with and without rollouts:") #print result
println(m, " ", s)

# also add shuffled rollouts
# load result across random seeds
@load "$(datadir)/performance_shuffled_planning.bson" results
ress_shuff = zeros(length(seeds), 2)
for (i, shuffle) = enumerate([true; false])
    for (iseed, seed) = enumerate(seeds)
        rews = results[seed][shuffle]
        ress_shuff[iseed, i] = sum(rews) / size(rews, 2)
    end
end
m, s = mean(ress_shuff, dims = 1)[:], std(ress_shuff, dims = 1)[:]/sqrt(length(seeds)) #mean and standard error
println("shuffled performance: ", m, " (", s, ")") #print result

ress = [ress ress_shuff[:, 1:1]]

# plot result
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.405, right=0.505, bottom = 0, top = 1, wspace=0.15)
ax = fig.add_subplot(grids[1,1])
plot_comparison(ax, ress; xticklabs=["rollout", "no roll", "shuffled"], ylab = "avg. reward", col = col_p, col2 = 1.2*col_p, yticks = [6;7;8], rotation = 45)

# plot example goal directed and non-goal directed rollouts
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.53, right=0.69, bottom = -0.03, top = 0.80, wspace=0.15)
ax = fig.add_subplot(grids[1,1])

# load example rollouts
@load "$datadir/example_rollout.bson" store
plan_state, ps, ws, state = store[10]
rew_loc = state_from_onehot(4, ps)

# plot arena
arena_lines(ps, ws, 4, rew=false, col="k", rew_col = "k", col_arena = "k", lw_arena = lw_arena, lw_wall = lw_wall)
labels = ["successful"; "unsuccessful"]
for i = 1:2 #for successful and unsuccessful
    col = [col_p1, col_p2][i] #different colours

    #extract the rollout paths
    plan_state, ps, ws, state = store[[4;5][i]]
    plan_state = plan_state[1:sum(plan_state .> 0)]
    states = [state state_from_loc(4, plan_state')] #prepend original state
    N = size(states, 2)

    for s = 1:(N-1) #plot each line segment
        x1, y1, x2, y2 = [states[:, s]; states[:, s+1]]
        if s == 1 label = labels[i] else label = nothing end #labels for legend
        if x1 == 4 && x2 == 1 #pass through right wall
            ax.plot([x1; 4.5], [y1; 0.5*(y1+y2)], color = col, lw = linewidth)
            ax.plot([0.5; x2], [0.5*(y1+y2); y2], color = col, lw = linewidth)
        elseif x1 == 1 && x2 == 4 #left wall
            ax.plot([x2; 4.5], [y2; 0.5*(y1+y2)], color = col, lw = linewidth)
            ax.plot([0.5; x1], [0.5*(y1+y2); y1], color = col, lw = linewidth)
        elseif y1 == 4 && y2 == 1 #top wall
            ax.plot([x1; 0.5*(x1+x2)], [y1; 4.5], color = col, lw = linewidth)
            ax.plot([0.5*(x1+x2); x2], [0.5; y2], color = col, lw = linewidth)
        elseif y1 == 1 && y2 == 4 #bottom wall
            ax.plot([x2; 0.5*(x1+x2)], [y2; 4.5], color = col, lw = linewidth)
            ax.plot([0.5*(x1+x2); x1], [0.5; y1], color = col, lw = linewidth)
        else #just a normal line segment
            ax.plot([x1; x2], [y1; y2], color = col, label = label, lw = linewidth)
        end
    end
end

ax.scatter([state[1]], [state[2]], color = col_p, s = 150, zorder = 1000) #original loc
ax.plot([rew_loc[1]], [rew_loc[2]], color = "k", marker="x", markersize=12, ls="", mew=3) #goal
ax.legend(frameon = false, fontsize = fsize_leg, loc = "upper center", bbox_to_anchor = (0.5, 1.28), handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.05)

# plot change in policy with successful and unsuccessful rollouts
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.78, right=1.00, bottom = 0, top = top, wspace=0.10)
for i = 1:2 #rewarded and non-rewarded rollout
    ms = []
    for seed = seeds #iterate through random seeds
        @load "$(datadir)/causal_N100_Lplan8_$(seed)_$(plan_epoch).bson" data
        #rollouts action probability under new and old policy
        p_simulated_actions, p_simulated_actions_old = data["p_simulated_actions"], data["p_simulated_actions_old"]
        #rollout probabilities
        p_initial_sim, p_continue_sim = data["p_initial_sim"], data["p_continue_sim"]
        p_simulated_actions ./= (1 .- p_continue_sim) #renormalize over actions
        p_simulated_actions_old ./= (1 .- p_initial_sim) #renormalize over actions
        inds = findall(.~isnan.(sum(p_simulated_actions, dims = 1)[:])) #make sure we have data for both scenarios
        push!(ms, [mean(p_simulated_actions_old[i, inds]); mean(p_simulated_actions[i, inds])]) #mean for new and old
    end
    ms = reduce(hcat, ms) #concatenate across seeds
    m3, s3 = mean(ms, dims = 2)[1:2], std(ms, dims = 2)[1:2] / sqrt(length(seeds)) #mean and sem across seeds

    # plot results
    ax = fig.add_subplot(grids[1,i])
    ax.bar(1:2, m3, yerr = s3, color = [col_p1, col_p2][i], capsize = capsize)
    # plot individual data points
    shifts = 1:size(ms, 2); shifts = (shifts .- mean(shifts))/std(shifts)*0.2
    ax.scatter([1 .+ shifts; 2 .+ shifts], [ms[1, :]; ms[2, :]], color = col_point, marker = ".", s = 15)
    ax.set_xticks(1:2, ["pre"; "post"])
    if i == 1 #successful rollout
        ax.set_ylabel(L"$\pi(\hat{a}_1)$", labelpad = 0)
        ax.set_title("succ.", fontsize = fsize)
        ax.set_yticks([0.1;0.3;0.5;0.7])
    else #unsuccessful rollout
        ax.set_title("unsucc.", fontsize = fsize)
        ax.set_yticks([])
    end
    #set some parameters
    ax.set_ylim(0.0, 0.8)
    ax.set_xlim([0.4; 2.6])
    ax.axhline(0.25, color = color = col_c, ls = "-")
end

# add labels and save
y1, y2 = 1.16, 0.46
x1, x2, x3, x4, x5 = -0.05, 0.15, 0.34, 0.51, 0.71
plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label, )
plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x3,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x4,y1,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x5,y1,"E";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)

savefig("./figs/fig_mechanism_behav.pdf", bbox_inches = "tight")
savefig("./figs/fig_mechanism_behav.png", bbox_inches = "tight")
close()


