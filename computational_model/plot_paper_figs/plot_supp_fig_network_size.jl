include("plot_utils.jl")
using BSON: @load
using Random, NaNStatistics, Statistics
cm = 1/2.54

fig = figure(figsize = (10*cm, 3*cm))
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.00, right=1.00, bottom = 0.0, top = 1.0, wspace=0.35)

@load "$datadir/rew_and_plan_by_n_model.bson" res_dict
meanrews, pfracs, seeds, Nhiddens, epochs = [res_dict[k] for k = ["meanrews", "planfracs", "seeds", "Nhiddens", "epochs"]]
epochs = epochs[epochs .<= plan_epoch]
hinds = [2;3;4]

i1 = 1
mms = mean(meanrews, dims = 2)[hinds, 1, i1:length(epochs)]
sms = std(meanrews, dims = 2)[hinds, 1, i1:length(epochs)] / sqrt(length(seeds))
mps = mean(pfracs, dims = 2)[hinds, 1, i1:length(epochs)]
sps = std(pfracs, dims = 2)[hinds, 1, i1:length(epochs)] / sqrt(length(seeds))
Nhiddens = Nhiddens[hinds]
xs = epochs[i1:end]*40*200 / 1000000
Nhid = length(Nhiddens)
cols = [[0, i/(Nhid+1), 1-i/(Nhid+1)] for i = 1:Nhid]

for (idat, dat) = enumerate([(mms, sms), (mps, sps)])
    ax = fig.add_subplot(grids[1,idat])
    m, s = dat
    for (ihid, Nhidden) = enumerate(Nhiddens)
        frac = (Nhidden - minimum(Nhiddens))/(maximum(Nhiddens) - minimum(Nhiddens))
        frac = (0.45*frac .+ 0.76)
        col = col_p * frac
        ax.plot(xs, m[ihid, :], ls = "-", color = col, label = Nhidden)
        ax.fill_between(xs, m[ihid, :]-s[ihid, :], m[ihid, :]+s[ihid, :], color = col, alpha = 0.2)
    end
    
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

add_labels = true
if add_labels
    y1 = 1.16
    x1, x2 = -0.09, 0.45
    fsize = fsize_label
    plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
    plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
end

savefig("./figs/supp_fig_by_size.pdf", bbox_inches = "tight")
savefig("./figs/supp_fig_by_size.png", bbox_inches = "tight")
close()
