#This script plots Figure S9 of Jensen et al.

include("plot_utils.jl")

# instantiate figure
fig = figure(figsize = (10*cm, 3*cm))
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.00, right=1.00, bottom = 0.0, top = 1.0, wspace=0.35)

# load data
@load "$datadir/rew_and_plan_by_n.bson" res_dict
meanrews, pfracs, seeds, Nhiddens, epochs = [res_dict[k] for k = ["meanrews", "planfracs", "seeds", "Nhiddens", "epochs"]]

# extract reward and plan data
mms = mean(meanrews, dims = 2)[:, 1, :] # mean across agents
sms = std(meanrews, dims = 2)[:, 1, :] / sqrt(length(seeds)) # standard error
mps = mean(pfracs, dims = 2)[:, 1, :] # mean
sps = std(pfracs, dims = 2)[:, 1, :] / sqrt(length(seeds)) # standard error

# convert from epochs to episodes
xs = epochs*40*200 / 1000000

# for both reward and planning fraction
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

# add panel labels and save
y1 = 1.16
x1, x2 = -0.09, 0.45
fsize = fsize_label
plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)

savefig("./figs/supp_fig_by_size.pdf", bbox_inches = "tight")
savefig("./figs/supp_fig_by_size.png", bbox_inches = "tight")
close()
