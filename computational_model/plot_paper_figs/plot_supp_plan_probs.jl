#This script plots Figure S13 of Jensen et al.

include("plot_utils.jl")

fig = figure(figsize = (5*cm, 3*cm))
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.00, right=1.00, bottom = 0.0, top = 1.0, wspace=0.35)

all_ms = [] # store list of pre/post values for both successful and unsuccessful rollouts
for i = 1:2 # for successful and unsuccessful rollouts
    ms = []
    for seed = seeds # for each trained model
        @load "$(datadir)/causal_N100_Lplan8_$(seed)_$(plan_epoch).bson" data # load data
        # extract pre- and post-rollout rollout probabilities
        p_initial_sim, p_continue_sim = data["p_initial_sim"], data["p_continue_sim"]
        inds = findall(.~isnan.(sum(p_continue_sim, dims = 1)[:])) #data for both successful and unsuccesful rollout
        # store data
        push!(ms, [mean(p_initial_sim[i, inds]); mean(p_continue_sim[i, inds])])
    end
    ms = reduce(hcat, ms) # concatenate across models
    push!(all_ms, ms[2, :]) # store data for later stats
    # mean and two standard errors across seeds
    m3, s3 = mean(ms, dims = 2)[1:2], 2*std(ms, dims = 2)[1:2] / sqrt(length(seeds))

    # plot our newly loaded data
    ax = fig.add_subplot(grids[1,i])
    ax.bar(1:2, m3, yerr = s3, color = [col_p1, col_p2][i], capsize = capsize) # bar plot
    # plot individual data points
    shifts = 1:size(ms, 2); shifts = (shifts .- mean(shifts))/std(shifts)*0.2 # add some jitter
    ax.scatter([1 .+ shifts; 2 .+ shifts], [ms[1, :]; ms[2, :]], color = col_point, marker = ".", s = 15, zorder = 100)
    ax.set_xticks(1:2)
    ax.set_xticklabels(["pre"; "post"])
    if i == 1 # successful
        ax.set_ylabel(L"$\pi($"*"rollout"*L"$)$", labelpad = -1.5)
        ax.set_title("succ.", fontsize = fsize)
        ax.set_yticks(0.0:0.2:0.8)
    else # unsuccessful
        ax.set_title("unsucc.", fontsize = fsize)
        ax.set_yticks([])
    end
    ax.set_ylim(0.0, 0.8)
    ax.set_xlim(0.5, 2.5)
end

# add labels and save figure
y1 = 1.16
x1, x2 = -0.25, 0.45
fsize = fsize_label
plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)

savefig("./figs/supp_plan_probs.pdf", bbox_inches = "tight")
savefig("./figs/supp_plan_probs.png", bbox_inches = "tight")
close()

# print the difference between successful and unsuccessful rollouts
delta = all_ms[2] - all_ms[1]
println("post delta: ", mean(delta), " ", std(delta)/sqrt(length(delta)))
println("Gaussian p = ", cdf(Normal(mean(delta), std(delta)/sqrt(length(delta))), 0))

