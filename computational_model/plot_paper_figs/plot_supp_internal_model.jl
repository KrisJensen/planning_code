#This script plots Figure S7 of Jensen et al.

include("plot_utils.jl")

fig = figure(figsize = (10*cm, 6*cm))
grids = fig.add_gridspec(nrows=2, ncols=2, left=0.00, right=1.00, bottom = 0.0, top = 1.0, wspace=0.5, hspace = 0.30)

# start by loading some data to plot
@load "$(datadir)/internal_model_accuracy.bson" results

global epochs = [] # training epochs
rews, states = [], [] # reward prediction accuracy and state prediction accuracy
for seed = seeds # for each trained model
    epochs = sort([k for k = keys(results[seed])]) # checkpoint epochs
    global epochs = epochs[epochs .<= plan_epoch] # training epochs
    push!(rews, [results[seed][e]["rew"] for e = epochs]) # reward prediction accuracy
    push!(states, [results[seed][e]["state"] for e = epochs]) # state prediction accuracy
end
rews, states = [reduce(hcat, arr) for arr = [rews, states]] # concatenate across models

mrs, mss = mean(rews, dims = 2)[:], mean(states, dims = 2)[:] # mean accuracies
srs, sss = std(rews, dims = 2)[:]/sqrt(length(seeds)), std(states, dims = 2)[:]/sqrt(length(seeds)) # standard errors
xs = epochs*40*200 / 1000000 # convert to number of episodes seen

# for the state prediction data and reward prediction data
for (idat, dat) = enumerate([(mss, sss), (mrs, srs)])
    for irange = 1:2 # for the full and zoomed in y ranges
        ax = fig.add_subplot(grids[irange,idat])
        m, s = dat # extract mean and sem for this data
        ax.plot(xs, m, ls = "-", color = col_p) # plot mean
        ax.fill_between(xs, m-s, m+s, color = col_p, alpha = 0.2) # standard error
        if irange == 2 # zoomed in
            ax.set_xlabel("training episodes (x"*L"$10^6$"*")")
            ax.set_xticks(0:2:8)
            ax.set_ylim(0.99, 1.0002)
            ax.set_yticks([0.99, 1.0])
            ax.set_yticklabels(["0.99"; "1.0"])
        else # full range
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

# add labels and save
y1 = 1.12
x1, x2 = -0.13, 0.46
fsize = fsize_label
plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)

savefig("./figs/supp_internal_model.pdf", bbox_inches = "tight")
savefig("./figs/supp_internal_model.png", bbox_inches = "tight")
close()
