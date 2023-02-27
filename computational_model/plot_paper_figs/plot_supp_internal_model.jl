include("plot_utils.jl")
using BSON: @load
using Random, NaNStatistics, Statistics
cm = 1/2.54

fig = figure(figsize = (10*cm, 6*cm))
grids = fig.add_gridspec(nrows=2, ncols=2, left=0.00, right=1.00, bottom = 0.0, top = 1.0, wspace=0.5, hspace = 0.30)

@load "$(datadir)/internal_model_accuracy.bson" results

global epochs = []
rews, states = [], []
for seed = 61:65
    epochs = sort([k for k = keys(results[seed])])
    global epochs = epochs[epochs .<= plan_epoch]
    push!(rews, [results[seed][e]["rew"] for e = epochs])
    push!(states, [results[seed][e]["state"] for e = epochs])
end
rews, states = [reduce(hcat, arr) for arr = [rews, states]]

i1 = 1
mrs, mss = mean(rews, dims = 2)[i1:end], mean(states, dims = 2)[i1:end]
srs, sss = std(rews, dims = 2)[i1:end]/sqrt(length(seeds)), std(states, dims = 2)[i1:end]/sqrt(length(seeds))
xs = epochs[i1:end]*40*200 / 1000000

for (idat, dat) = enumerate([(mss, sss), (mrs, srs)])
    for irange = 1:2
        ax = fig.add_subplot(grids[irange,idat])
        m, s = dat
        ax.plot(xs, m, ls = "-", color = col_p)
        ax.fill_between(xs, m-s, m+s, color = col_p, alpha = 0.2)
        if irange == 2
            ax.set_xlabel("training episodes (x"*L"$10^6$"*")")
            ax.set_xticks(0:2:8)
            ax.set_ylim(0.99, 1.0002)
            ax.set_yticks([0.99, 1.0])
            ax.set_yticklabels(["0.99"; "1.0"])
        else
            ax.set_xticks([])
            ax.set_ylim(0.0, 1.02)
            ax.set_yticks([0.0; 0.5; 1.0])
            ax.set_yticklabels(["0.00"; "0.50"; "1.00"])
        end
        if idat == 1
            ax.set_ylabel("state prediction")
        else
            ax.set_ylabel("reward prediction")
        end
        ax.set_xlim(0,8)
    end
end

add_labels = true
if add_labels
    y1 = 1.12
    x1, x2 = -0.13, 0.46
    fsize = fsize_label
    plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
    plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
end

savefig("./figs/supp_internal_model.pdf", bbox_inches = "tight")
savefig("./figs/supp_internal_model.png", bbox_inches = "tight")
close()
