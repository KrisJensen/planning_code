## This script plots Figure S3 of Jensen et al.

include("plot_utils.jl")
using NaNStatistics

## instantiate figure
fig = figure(figsize = (10*cm, 7*cm))

# load data
@load "$datadir/rew_and_plan_by_n.bson" res_dict
meanrews, pfracs, seeds, Nhiddens, epochs = [res_dict[k] for k = ["meanrews", "planfracs", "seeds", "Nhiddens", "epochs"]]

# extract reward and plan data
mms = mean(meanrews, dims = 2)[:, 1, :] # mean across agents
sms = std(meanrews, dims = 2)[:, 1, :] / sqrt(length(seeds)) # standard error
mps = mean(pfracs, dims = 2)[:, 1, :] # mean
sps = std(pfracs, dims = 2)[:, 1, :] / sqrt(length(seeds)) # standard error

## convert from epochs to episodes
xs = epochs*40*200 / 1000000

# for both reward and planning fraction
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.00, right=1.00, bottom = 0.0, top = 0.4, wspace=0.45, hspace=0.60)
for (idat, dat) = enumerate([(mms, sms), (mps, sps)])
    ax = fig.add_subplot(grids[1,idat]) # new subplot
    m, s = dat # mean and standard error
    for (ihid, Nhidden) = enumerate(Nhiddens) # for each network size
        frac = (Nhidden - minimum(Nhiddens))/(maximum(Nhiddens) - minimum(Nhiddens))
        frac = (0.45*frac .+ 0.76)
        col = col_p * frac
        # plot mean and sem
        ax.plot(xs, m[ihid, :], ls = "-", color = col, label = Nhidden)
        ax.fill_between(xs, m[ihid, :]-s[ihid, :], m[ihid, :]+s[ihid, :], color = col, alpha = 0.2)
    end
    
    # set some axis labels etc.
    ax.set_xlabel("training episodes (x"*L"$10^6$"*")")
    if idat == 1
        ax.legend(frameon = false, fontsize = fsize_leg, handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.05)
        ax.set_ylabel("mean reward")
        ax.set_ylim(0, 9)
    else
        ax.set_ylabel(L"$p$"*"(rollout)")
        ax.set_ylim(0, 0.65)
        ax.axhline(0.2, ls = "--", color = "k")
    end
    ax.set_xticks(0:2:8)
    ax.set_xlim(0,8)
end


## add human data

@load "$(datadir)/human_RT_and_rews_follow.bson" data; data_follow = data # guided episodes
means = [nanmean(RTs) for RTs = data_follow["all_RTs"]]
keep = findall(means .< 690) # non-outlier users
Nkeep = length(keep)

@load "$(datadir)/human_all_data_play.bson" data;
_, _, _, _, all_rews_p, all_RTs_p, all_trial_nums_p, _ = data;
@load "$datadir/guided_lognormal_params_delta.bson" params # parameters of prior distributions
all_TTs, all_DTs = [], []
for u = keep # for each participant
    rts, tnums = all_RTs_p[u], all_trial_nums_p[u] # RTs and trial numbers
    new_TTs, new_DTs = [zeros(size(rts)) .+ NaN for _ = 1:2] # thinking time and delay times
    initial, later = params["initial"][u, :], params["later"][u, :]
    # functions for computing posterior means
    initial_post_mean(r) = calc_post_mean(r, muhat=initial[1], sighat=initial[2], deltahat=initial[3], mode = false)
    later_post_mean(r) = calc_post_mean(r, muhat=later[1], sighat=later[2], deltahat=later[3], mode = false)
    tnum = 1
    for ep = 1:size(rts, 1) # for each episode
        for b = 1:sum(tnums[ep, :] .> 0.5) # for each action
            t, rt = tnums[ep, b], rts[ep, b] # trial number and response time
            if b > 1.5 # discard very first action
                if t == tnum # same trial as before
                    new_TTs[ep, b] = later_post_mean(rt) 
                else # first action of new trial
                    new_TTs[ep, b] = initial_post_mean(rt)
                end
                new_DTs[ep, b] = rt - new_TTs[ep, b] # delays is response time minus thinking time
            end
            tnum = t
        end
    end
    push!(all_TTs, Float64.(new_TTs))
    push!(all_DTs, Float64.(new_DTs))
end
# store data
all_RTs = [all_TTs[i] + all_DTs[i] for i = 1:length(all_TTs)]

##
# combine data
rews_by_episode = reduce(hcat, [nansum(rews, dims = 2) for rews = all_rews_p])
RTs_by_episode = reduce(hcat, [nanmedian(RTs, dims = 2) for RTs = all_RTs])
TTs_by_episode = reduce(hcat, [nanmedian(TTs, dims = 2) for TTs = all_TTs])
DTs_by_episode = reduce(hcat, [nanmedian(DTs, dims = 2) for DTs = all_DTs])
function permtest(m, label)
    # run permutation test for significance
    cval = cor(1:length(m), m) # correlation
    Niter = 10000; ctrl = zeros(Niter)
    for n = 1:Niter
        ctrl[n] = cor(randperm(length(m)), m)
    end
    println("$(label) correlation of mean: $cval, permutation p = $(mean(ctrl .> cval))")
end

## first plot rews
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.00, right=0.175, bottom = 0.63, top = 1.0, wspace=0.45, hspace=0.60)

dat = rews_by_episode
mr = mean(dat, dims = 2)[:] # mean
sr = std(dat, dims = 2)[:] / sqrt(size(dat, 2)) #standard error
xs = 1:length(mr)
ax = fig.add_subplot(grids[1,1]) # new subplot
permtest(mr, "reward")
cvals = [cor(xs, dat[:, i]) for i = 1:size(dat, 2)]
println("reward mean & sem across people: $(mean(cvals)) & $(std(cvals)/sqrt(length(cvals)))")

ax.plot(xs, mr, ls = "-", color = col_h)
ax.fill_between(xs, mr-sr, mr+sr, color = col_h, alpha = 0.2)
# set some axis labels etc.
ax.set_xlabel("episode")
ax.set_ylabel("mean reward")
ax.set_ylim(6.5, 9.5)
ax.set_xticks(0:12:38)
ax.set_xlim(0,38)

## now plot RTs
grids = fig.add_gridspec(nrows=1, ncols=3, left=0.330, right=1.00, bottom = 0.63, top = 1.0, wspace=0.55, hspace=0.60)

labels = ["response"; "thinking"; "delay"]
lss = ["-", "-", "-"]
all_cvals = []
for (idat, dat) = enumerate([RTs_by_episode, TTs_by_episode, DTs_by_episode])
    ax = fig.add_subplot(grids[1,idat]) # new subplot
    m = mean(dat, dims = 2)[:] # mean
    s = std(dat, dims = 2)[:] / sqrt(size(dat, 2)) #standard error
    xs = 1:length(m)
    
    # run permutation test for significance
    permtest(m, labels[idat])
    cvals = [cor(xs, dat[:, i]) for i = 1:size(dat, 2)]
    push!(all_cvals, cvals)
    println("$(labels[idat]) mean & sem across people: $(mean(cvals)) & $(std(cvals)/sqrt(length(cvals)))")

    ax.plot(xs, m, ls = lss[idat], color = col_h)
    ax.fill_between(xs, m-s, m+s, color = col_h, alpha = 0.2)
    ax.set_xlabel("episode")
    if idat == 1
        ax.set_ylabel("time (ms)")
    end
    ax.set_xticks(0:12:38)
    ax.set_xlim(0,38)
    ax.set_title(labels[idat], fontsize = fsize)
end
diffs = all_cvals[3] - all_cvals[2]
println("delay - thinking cvals: m = $(mean(diffs)), sem = $(std(diffs)/sqrt(length(diffs)))")

## add panel labels and save
y1, y2 = 1.08, 0.45
x1, x2 = -0.09, 0.45
fsize = fsize_label
plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
plt.text(x2-0.25,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
plt.text(x1,y2,"C"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
plt.text(x2,y2,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)


savefig("./figs/supp_fig_by_size.pdf", bbox_inches = "tight")
savefig("./figs/supp_fig_by_size.png", bbox_inches = "tight")
close()


### print avg rew and RT for first/last 5 episodes ###

mr1, mr2 = (mean(mr[1:5])), (mean(mr[34:38]))
println("mean first five rew: $mr1")
println("mean last five rew: $mr2")

mt = mean(RTs_by_episode, dims = 2)[:] # mean
mt1, mt2 = (mean(mt[1:5])), (mean(mt[34:38]))
println("mean first five RT: $mt1")
println("mean last five RT: $mt2")

println("scale up: ", mr1*mt1/mt2)

println("rel time: ", abs(mt2-mt1)/mt1*100)
println("rel rew: ", abs(mr2-mr1)/mr1*100)
