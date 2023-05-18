#This script plots Figure S5 of Jensen et al.

include("plot_utils.jl")

fig = figure(figsize = (10.5*cm, 7*cm))
grids = fig.add_gridspec(nrows=2, ncols=2, left=0.00, right=1.00, bottom = 0, top = 1.0, wspace=0.61, hspace = 0.7)

# we start by plotting the accuracy of the internal world model as a function of the number of states visited during exploration

# load data from different model seeds
global ms, unums = [], []
for seed = seeds
    @load "$(datadir)model_exploration_predictions_$(seed)_$plan_epoch.bson" data
    global unums, dec_perfs = data
    push!(ms, dec_perfs)
end
ms = reduce(hcat, ms) # concatenate across seeds
m, s = mean(ms, dims = 2)[:], std(ms, dims = 2)[:]/sqrt(size(ms, 2)) # mean and standard error across seeds

# now plot our results
ax = fig.add_subplot(grids[1,1])
ax.plot(unums, 1 ./ (16 .- unums), color = col_c, label = "optimal", zorder = -1000)
ax.plot(unums, m, color = col_p, label = "agent")
ax.fill_between(unums, m-s, m+s, color = col_p, alpha = 0.2)
ax.set_xlabel("states visited")
ax.set_ylabel("accuracy")
ax.legend(frameon = false, fontsize = fsize_leg)
ax.set_ylim(0.0, 0.7)
ax.set_xlim(unums[1], unums[end])

# plot human thinking time against pi(rollout) during exploration

# load our results
@load "$datadir/RT_predictions_N100_Lplan8_explore_1000.bson" data
allsims, RTs, pplans = [data[k] for k = ["correlations"; "RTs_by_u"; "pplans_by_u"]];
RTs, pplans = [reduce(vcat, arr) for arr = [RTs, pplans]];
bins = 0.05:0.05:0.70
xs = 0.5*(bins[1:length(bins)-1] + bins[2:end])

# bin data and generate shuffled control
RTs_shuff = RTs[randperm(length(RTs))]
dat = [RTs[(pplans .>= bins[i]) .& (pplans .< bins[i+1])] for i = 1:length(bins)-1]
dat_shuff = [RTs_shuff[(pplans .>= bins[i]) .& (pplans .< bins[i+1])] for i = 1:length(bins)-1]

# mean and standard error
m, m_c = [[mean(d) for d = dat] for dat = [dat, dat_shuff]]
s, s_c = [[std(d)/sqrt(length(d)) for d = dat] for dat = [dat, dat_shuff]]

# plot result
ax = fig.add_subplot(grids[1,2])
ax.bar(xs, m, color = col_p, width = 0.04, linewidth = 0, label = "data")
ax.errorbar(xs, m, yerr = s, fmt = "none", color = "k", capsize = 2, lw = 1.5)
ax.errorbar(xs, m_c, yerr = s_c, fmt = "-", color = col_c, capsize = 2, lw = 1.5, label = "shuffle")
ax.set_xlabel(L"$\pi$"*"(rollout)")
ax.set_ylabel("thinking time (ms)")
ax.legend(frameon = false, fontsize = fsize_leg)

# print result
m = mean(allsims, dims = 1)
s = std(allsims, dims = 1) / sqrt(size(allsims, 1))
println("correlations mean and sem: ", m, " ", s)

# plot thinking time against the number of unique states visited during exploration for RL agent

uvals = 2:15 # number of unique states to consider (ignore very first action)
RTs_us = zeros(length(seeds), length(uvals)) .+ NaN # initialize array for storing thinking times

for (iseed, seed) = enumerate(seeds) # for each model seed
    @load "$(datadir)model_unique_states_$(seed)_1000.bson" data # load result
    RTs, unique_states = data # extract data
    new_us, new_rts = [], []
    for b = 1:size(RTs, 1) # for each episode
        us = unique_states[b, :] # unique state counts
        inds = 2:sum(.~isnan.(us)) # actions of episode 1
        rts = RTs[b, inds] # thinking times for these actions
        push!(new_us, us[inds]); push!(new_rts, rts) # store data
    end
    new_us, new_rts = reduce(vcat, new_us), reduce(vcat, new_rts) # concatenate results across episodes
    RTs_us[iseed, :] = [mean(new_rts[new_us .== uval]) .- 1 for uval = uvals]*120 # mean value for this model in ms
end

# plot result
m, s = mean(RTs_us, dims = 1)[:], std(RTs_us, dims = 1)[:]/sqrt(size(RTs_us, 1)) # mean and standard error across models
ax = fig.add_subplot(grids[2,1])
ax.plot(uvals, m, ls = "-", color = col_p)
ax.fill_between(uvals, m-s, m+s, color = col_p, alpha = 0.2)
ax.set_xlabel("states visited")
ax.set_ylabel("thinking time (ms)")
ax.set_xlim(uvals[1], uvals[end])

# plot thinking time against unique states for human participants

# load our data
@load "$(datadir)/human_RT_and_rews_follow.bson" data
keep = findall([nanmean(RTs) for RTs = data["all_RTs"]] .< 690)
Nkeep = length(keep) # subjects to consider
@load "$datadir/guided_lognormal_params_delta.bson" params #load prior parameters

@load "$(datadir)unique_states_play.bson" data
all_RTs, all_unique_states = data # response time and state informationduring exploration
RTs_us = zeros(Nkeep, length(uvals)) .+ NaN # initialize array to store data
for (i_u, u) = enumerate(keep) # for each participant
    new_us, new_rts = [], []
    for b = 1:size(all_RTs[u], 1) # for each episode
        us = all_unique_states[u][b, :] # unique states for this episode
        inds = 2:sum(.~isnan.(us)) # indices of first trial
        rts = all_RTs[u][b, inds] # response times
        # compute posterior mean thinking times
        later = params["later"][u, :]
        later_post_mean(r) = calc_post_mean(r, muhat=later[1], sighat=later[2], deltahat=later[3], mode = false)
        rts = later_post_mean.(all_RTs[u][b, inds]) #posterior mean
        push!(new_us, us[inds]); push!(new_rts, rts) # store our results
    end
    new_us, new_rts = reduce(vcat, new_us), reduce(vcat, new_rts) # concatenate across episodes
    RTs_us[i_u, :] = [mean(new_rts[new_us .== uval]) for uval = uvals] # mean values for this participant
end

# plot our results
m, s = nanmean(RTs_us, dims = 1)[:], nanstd(RTs_us, dims = 1)[:]/sqrt(size(RTs_us, 1)) # mean and standard error across participants
ax = fig.add_subplot(grids[2,2])
ax.plot(uvals, m, "k-")
ax.fill_between(uvals, m-s, m+s, color = "k", alpha = 0.2)
ax.set_xlabel("states visited")
ax.set_ylabel("thinking time (ms)")
ax.set_xlim(uvals[1], uvals[end])

# add labels and save figure
y1 = 1.08
y2 = 0.47
x1, x2 = -0.18, 0.43
fsize = fsize_label
plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
plt.text(x1,y2,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
plt.text(x2,y2,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
savefig("./figs/supp_exploration.pdf", bbox_inches = "tight")
savefig("./figs/supp_exploration.png", bbox_inches = "tight")
close()

