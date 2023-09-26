#This script plots Figure S1 of Jensen et al.

include("plot_utils.jl")
using ToPlanOrNotToPlan
Random.seed!(1) # set random seed (for jitter in panel D)

RTs_play, TTs_play, base1s, base2s, rews = [], [], [], [], []

for wrapstr = ["", "_euclidean"]
    # We start by loading some of our human behavioural data

    @load "$(datadir)/human_all_data_play$wrapstr.bson" data;
    _, _, _, _, all_rews_p, all_RTs_p, all_trial_nums_p, _ = data;
    @load "$(datadir)/human_all_data_follow$wrapstr.bson" data;
    _, _, _, _, all_rews_f, all_RTs_f, all_trial_nums_f, _ = data;

    means1, means2 = [[nanmean(RT) for RT = RTs] for RTs = [all_RTs_f, all_RTs_p]]
    keep = findall(means1 .< 690) # non-outlier users
    Nkeep = length(keep)
    means1, means2 = means1[keep], means2[keep]

    @load "$datadir/guided_lognormal_params_delta$wrapstr.bson" params # parameters of prior distributions
    initial_delays = (params["initial"][:, 3]+exp.(params["initial"][:, 1]+params["initial"][:, 2].^2/2))[keep]
    later_delays = (params["later"][:, 3]+exp.(params["later"][:, 1]+params["later"][:, 2].^2/2))[keep]
    push!(base1s, initial_delays)
    push!(base2s, later_delays)
    push!(RTs_play, means2)

    push!(rews, [nansum(rew)/size(rew, 1) for rew = all_rews_p][keep])

    all_TTs = []
    for u = 1:length(initial_delays)
        new_TTs = []
        rts, tnums = all_RTs_p[keep[u]], all_trial_nums_p[keep[u]]
        initial, later = params["initial"][u, :], params["later"][u, :]
        initial_post_mean(r) = calc_post_mean(r, muhat=initial[1], sighat=initial[2], deltahat=initial[3], mode = false)
        later_post_mean(r) = calc_post_mean(r, muhat=later[1], sighat=later[2], deltahat=later[3], mode = false)
        tnum = 1
        for ep = 1:size(rts, 1)
            for b = 1:sum(tnums[ep, :] .> 0.5)
                t, rt = tnums[ep, b], rts[ep, b]
                if t > 1.5
                    if t == tnum # same trial
                        push!(new_TTs, later_post_mean(rt))
                        #println(t, " ", rt, " same trial ", new_TTs[end])
                    else # first action of new trial
                        push!(new_TTs, initial_post_mean(rt))
                        #println(t, " ", rt, " new trial ", new_TTs[end])
                    end
                end
                tnum = t
            end
        end
        push!(all_TTs, nanmean(Float64.(new_TTs)))
    end
    push!(TTs_play, all_TTs)

end


#### plot some results ####

titles = ["reaction"; "thinking"; "initial"; "later"; "rewards"]
ylabs = ["time (ms)"; "thinking time (ms)"; "time (ms)"; "time (ms)"; "avg. reward"]
datas = [RTs_play, TTs_play, base1s, base2s, rews]
inds = [2;5]
titles, datas, ylabs = titles[inds], datas[inds], ylabs[inds]

fig = figure(figsize = (15*cm, 3*cm))
grids = fig.add_gridspec(nrows=1, ncols=length(datas), left=0.00, right=0.36, bottom = 0, top = 1.0, wspace=0.6)

for (idat, data) = enumerate(datas)
    torus, euclid = data
    NT, NE = length(torus), length(euclid)
    mus = [mean(torus); mean(euclid)] # mean across users
    diff = mus[1] - mus[2]
    comb = [torus; euclid]
    ctrls = zeros(10000)
    for i = 1:10000
        newcomb = comb[randperm(length(comb))]
        ctrls[i] = mean(newcomb[1:NT]) - mean(newcomb[NT+1:end])
    end
    println(titles[idat], " means: ", mus, " p = ", mean(ctrls .> diff))

    ax = fig.add_subplot(grids[1,idat])
    ax.bar(1:2, mus, color = col_c) # bar plot
    # plot individual data points
    ax.scatter(ones(NT)+randn(NT)*0.1, torus, marker = ".", s = 6, color = "k")
    ax.scatter(ones(NE)*2+randn(NE)*0.1, euclid, marker = ".", s = 6, color = "k")
    ax.set_xticks(1:2, ["wrap"; "no-wrap"], rotation = 45, ha = "right")
    ax.set_ylabel(ylabs[idat])
    #ax.set_title(titles[idat])
end

### plot rew vs RT for both conditions ###
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.47, right=1.0, bottom = 0, top = 1.0, wspace=0.6)
ax = fig.add_subplot(grids[1,1])
for i = 1:2
    ax.scatter(RTs_play[i], rews[i], color = ["k", col_c][i], marker = ".", s = 60)
end
ax.legend(["wrap"; "no-wrap"], loc = (0.4, 0.65), handletextpad=0.4, borderaxespad = 0.3, handlelength = 1.0)
ax.set_xlabel("mean RT (ms)")
ax.set_ylabel("mean reward")
ax.set_yticks(4:2:16)

### distribution of path lengths ###

@load "$(datadir)/wrap_and_nowrap_pairwise_dists.bson" dists

ds = 1:12
hist_wraps = [sum(dists[1] .== d) for d = ds]
hist_nowraps = [sum(dists[2] .== d) for d = ds]
xs = reduce(vcat, [[d-0.5; d+0.5] for d = ds])
hwraps = reduce(vcat, [[h;h] for h = hist_wraps/sum(hist_wraps)])
hnowraps = reduce(vcat, [[h;h] for h = hist_nowraps/sum(hist_wraps)])
ax = fig.add_subplot(grids[1,2])
plot(xs, hwraps, color = "k")
plot(xs, hnowraps, color = col_c)
axvline(mean(dists[1]), color = "k", lw = 1.5)
axvline(mean(dists[2]), color = col_c, lw = 1.5)
ax.set_xlabel("distance to goal")
ax.set_ylabel("frequency")

# add labels and save
y1 = 1.16
x1, x2, x3, x4 = -0.05, 0.15, 0.40, 0.70
plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label, )
plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x3,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x4,y1,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
savefig("./figs/supp_human_euclidean_comparison.pdf", bbox_inches = "tight")
savefig("./figs/supp_human_euclidean_comparison.png", bbox_inches = "tight")
close()


