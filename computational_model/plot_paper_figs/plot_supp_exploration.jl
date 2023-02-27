include("plot_utils.jl")
using BSON: @load
using Random, NaNStatistics, Statistics
cm = 1/2.54

bot, top = 0.0, 1.0
fig = figure(figsize = (10.5*cm, 7*cm))
grids = fig.add_gridspec(nrows=2, ncols=2, left=0.00, right=1.00, bottom = 0, top = 1.0, wspace=0.61, hspace = 0.7)

## plot decoding ability ##

global ms, unums = [], []
for seed = seeds
    @load "$(datadir)model_exploration_predictions_$(seed)_$plan_epoch.bson" data
    global unums, dec_perfs = data
    push!(ms, dec_perfs)
end
ms = reduce(hcat, ms)
m, s = mean(ms, dims = 2)[:], std(ms, dims = 2)[:]/sqrt(size(ms, 2))

ax = fig.add_subplot(grids[1,1])
ax.plot(unums, 1 ./ (16 .- unums), color = col_c, label = "optimal", zorder = -1000)
ax.plot(unums, m, color = col_p, label = "agent")
ax.fill_between(unums, m-s, m+s, color = col_p, alpha = 0.2)
ax.set_xlabel("states visited")
ax.set_ylabel("accuracy")
ax.legend(frameon = false, fontsize = fsize_leg)
ax.set_ylim(0.0, 0.7)
ax.set_xlim(unums[1], unums[end])

## plot RT vs p(rollout) ##

ax = fig.add_subplot(grids[1,2])

if weiji
    savename = "RT_predictions_N100_Lplan8_explore_1000_weiji"
    @load "$datadir/$savename.bson" data
else
    @load "$(datadir)RT_predictions_new_explore_1000.bson" data
end
res, allsims, RTs, pplans, dists, steps = [data[k] for k = ["residuals"; "correlations"; "RTs"; "pplans"; "dists"; "steps"]];
bins = 0.05:0.05:0.70
xs = 0.5*(bins[1:length(bins)-1] + bins[2:end])

RTs_shuff = RTs[randperm(length(RTs))]
dat = [RTs[(pplans .>= bins[i]) .& (pplans .< bins[i+1])] for i = 1:length(bins)-1]
dat_shuff = [RTs_shuff[(pplans .>= bins[i]) .& (pplans .< bins[i+1])] for i = 1:length(bins)-1]

m, m_c = [[mean(d) for d = dat] for dat = [dat, dat_shuff]]
s, s_c = [[std(d)/sqrt(length(d)) for d = dat] for dat = [dat, dat_shuff]]

N = [length(d) for d = dat]
#ax.errorbar(xs, m, yerr = s, fmt = "", color = col_p, capsize = 4)
ax.bar(xs, m, color = col_p, width = 0.04, linewidth = 0, label = "data")
#ax.errorbar(xs, m, yerr = s, fmt = "none", color = col_p, capsize = 0, lw = 0, label = "data")
ax.errorbar(xs, m, yerr = s, fmt = "none", color = "k", capsize = 2, lw = 1.5)
ax.errorbar(xs, m_c, yerr = s_c, fmt = "-", color = col_c, capsize = 2, lw = 1.5, label = "shuffle")
ax.set_xlabel(L"$\pi$"*"(rollout)")
ax.set_ylabel("thinking time (ms)")
ax.legend(frameon = false, fontsize = fsize_leg)

m = mean(allsims, dims = 1)
s = std(allsims, dims = 1) / sqrt(size(allsims, 1))
println("correlations mean and sem: ", m, " ", s)

## plot pi(rollout) vs #unique states ##

uvals, tvals = 2:15, 2:20
cors_ts, cors_us = [], []
RTs_us, RTs_ts = zeros(length(seeds), length(uvals)) .+ NaN, zeros(length(seeds), length(tvals)) .+ NaN

for (iseed, seed) = enumerate(seeds)
    @load "$(datadir)model_unique_states_$(seed)_1000.bson" data
    RTs, unique_states = data
    new_us, new_ts, new_rts = [], [], []
    for b = 1:size(RTs, 1)
        us = unique_states[b, :]
        inds = 2:sum(.~isnan.(us))
        rts = RTs[b, inds]
        push!(new_us, us[inds]); push!(new_ts, inds); push!(new_rts, rts)
    end
    new_us, new_ts, new_rts = reduce(vcat, new_us), reduce(vcat, new_ts), reduce(vcat, new_rts)
    push!(cors_ts, cor(new_ts, new_rts)); push!(cors_us, cor(new_us, new_rts))
    RTs_us[iseed, :] = [mean(new_rts[new_us .== uval]) .- 1 for uval = uvals]*120
    RTs_ts[iseed, :] = [mean(new_rts[new_ts .== tval]) .- 1 for tval = tvals]*120
end

deltas = cors_ts - cors_us
println(mean(deltas), " ", std(deltas)/sqrt(length(deltas)))

m, s = mean(RTs_us, dims = 1)[:], std(RTs_us, dims = 1)[:]/sqrt(size(RTs_us, 1))
ax = fig.add_subplot(grids[2,1])
ax.plot(uvals, m, ls = "-", color = col_p)
ax.fill_between(uvals, m-s, m+s, color = col_p, alpha = 0.2)
ax.set_xlabel("states visited")
ax.set_ylabel("thinking time (ms)")
ax.set_xlim(uvals[1], uvals[end])

## plot RT vs #unique states ##

@load "$(datadir)/human_RT_and_rews_follow.bson" data
keep = findall([nanmean(RTs) for RTs = data["all_RTs"]] .< 690)
Nkeep = length(keep)
@load "$(datadir)process_and_action_times_mode_follow.bson" data
process_times, action_times = data
@load "$datadir/guided_lognormal_params_delta.bson" params #mu, sigma, delta

@load "$(datadir)unique_states_play.bson" data
all_RTs, all_unique_states = data
global cors_ts, cors_us = [], []

RTs_us, RTs_ts = zeros(Nkeep, length(uvals)) .+ NaN, zeros(Nkeep, length(tvals)) .+ NaN

for (i_u, u) = enumerate(keep)
    new_us, new_ts, new_rts = [], [], []
    for b = 1:size(all_RTs[u], 1)
        us = all_unique_states[u][b, :]
        inds = 2:sum(.~isnan.(us))
        rts = all_RTs[u][b, inds]
        rts = rts .- action_times[u]

        if weiji
            later = params["later"][u, :]
            #posterior mean for later actions
            later_post_mean(r) = calc_post_mean(r, muhat=later[1], sighat=later[2], deltahat=later[3], mode = false)
            rts = later_post_mean.(all_RTs[u][b, inds]) #posterior mean
        end


        push!(new_us, us[inds]); push!(new_ts, inds); push!(new_rts, rts)
    end
    new_us, new_ts, new_rts = reduce(vcat, new_us), reduce(vcat, new_ts), reduce(vcat, new_rts)
    push!(cors_ts, cor(new_ts, new_rts)); push!(cors_us, cor(new_us, new_rts))
    RTs_us[i_u, :] = [mean(new_rts[new_us .== uval]) for uval = uvals]
    RTs_ts[i_u, :] = [mean(new_rts[new_ts .== tval]) for tval = tvals]
end

deltas = cors_ts - cors_us
println(mean(deltas), " ", std(deltas)/sqrt(length(deltas)))

m, s = nanmean(RTs_us, dims = 1)[:], nanstd(RTs_us, dims = 1)[:]/sqrt(size(RTs_us, 1))
ax = fig.add_subplot(grids[2,2])
ax.plot(uvals, m, "k-")
ax.fill_between(uvals, m-s, m+s, color = "k", alpha = 0.2)
ax.set_xlabel("states visited")
ax.set_ylabel("thinking time (ms)")
ax.set_xlim(uvals[1], uvals[end])

## add labels

add_labels = true
if add_labels
    y1 = 1.08
    y2 = 0.47
    x1, x2, x3, x4 = -0.18, 0.43, 0.45, 0.70
    fsize = fsize_label
    plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
    plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x1,y2,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x2,y2,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
end

savefig("./figs/supp_exploration.pdf", bbox_inches = "tight")
savefig("./figs/supp_exploration.png", bbox_inches = "tight")
close()

