#This script plots panels C-E of Figure 5 from Jensen et al.

include("plot_utils.jl")
using MultivariateStats

fig = figure(figsize = (17*cm, 3.0*cm))

# load data
@load "$(datadir)planning_as_pg.bson" res_dict
seeds = sort([k for k = keys(res_dict)])

# PCA plot of mean hidden state updates
seed = 62 # example seed to plot
alphas = res_dict[seed]["jacs"] # true hidden state updates
actions = Int.(res_dict[seed]["sim_as"]) # rollout actions
betas = res_dict[seed]["sim_gs"] # predicted hidden state updates
betas = reduce(vcat, [betas[i:i, :, actions[i]] for i = 1:length(actions)]) # concatenate
pca = MultivariateStats.fit(PCA, (betas .- mean(betas, dims = 1))'; maxoutdim=3) # perform PCA on the predicted changes
Zb = predict(pca, (betas .- mean(betas, dims = 1))') # project into PC space
Za = predict(pca, (alphas .- mean(alphas, dims = 1))') # project into PC space

# plot result
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.19, right=0.42, bottom = -0.24, top = 0.98, wspace=0.05)
ax = fig.add_subplot(grids[1,1], projection="3d")

cols = [col_c, col_p, "g", "c"] # colours to use
for a = 1:4 # for each action
    meanb = mean(Zb[:, actions .== a], dims = 2)[:] # mean predicted
    meanb = meanb / sqrt(sum(meanb.^2)) # normalize vector
    ax.plot3D([0; meanb[1]], [0; meanb[2]], [0; meanb[3]], ls = "-", color = cols[a], lw = 2) # plot predicted
    ax.scatter3D([meanb[1]], [meanb[2]], [meanb[3]], color = cols[a], s = 50) # plot end points
    meana = mean(Za[:, actions .== a], dims = 2)[:] # mean empirical
    meana = meana / sqrt(sum(meana.^2)) # normalize vector
    ax.plot3D([0; meana[1]], [0; meana[2]], [0; meana[3]], ls = ":", color = cols[a], lw = 3) # plot empirical
end
# add some labels
ax.plot3D(zeros(2), zeros(2), zeros(2), ls = "-", color = "k", lw = 2, label = L"{\bf \alpha}^\mathrm{PG}_{1}")
ax.plot3D(zeros(2), zeros(2), zeros(2), ls = ":", color = "k", lw = 3, label = L"{\bf \alpha}^\mathrm{RNN}")
ax.set_xlabel("PC 1", labelpad = -16, rotation = 9);
ax.set_ylabel("PC 2", labelpad = -17, rotation = 107);
ax.set_zlabel("PC 3", labelpad = -17, rotation = 92);

# set some plot parameters
ax.view_init(elev=35., azim=75.)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.legend(frameon = false, ncol = 2, bbox_to_anchor = (0.52, 1.05), loc = "upper center",columnspacing=1,
        fontsize = fsize_leg, borderpad = 0.0, labelspacing = 0.2, handlelength = 1.3,handletextpad=0.4, handleheight=1)

# plot discrepancy between expected and empirical
meanspca, meanspca2 = [[[] for j = 1:3] for _ = 1:2]
# iterate through alpha_RNN, alpha_RNN_ctrl, and alpha_RNN_ctrl2 [second action switched]
for (ij, jkey) = enumerate(["jacs"; "jacs_shift"; "jacs_shift2"])
    for seed = seeds # iterate through models
        sim_as, sim_a2s = [res_dict[seed][k] for k = ["sim_as", "sim_a2s"]] # rollout actions
        jacs, gs, gs2 = [copy(res_dict[seed][k]) for k = [jkey, "sim_gs", "sim_gs2"]] # alpha_RNN, alpha_PG1, alpha_PG2
        inds, inds2 = 1:length(sim_as), findall(.~isnan.(sim_a2s)) # actions to consider (min rollout length of 2 for alpha_PG2)
        betas = reduce(vcat,[gs[i:i,:,Int(sim_as[i])] for i=inds]) # alpha_PG1
        betas2 = reduce(vcat,[gs2[i:i,:,Int(sim_as[i])] for i=inds2]) # alpha_PG2
        jacs, betas, betas2 = [arr .- mean(arr, dims = 1) for arr = [jacs, betas, betas2]] # mean-subtract
        jacs, betas, betas2 = [arr ./ sqrt.(sum(arr.^2, dims = 2)) for arr = [jacs, betas, betas2]] # normalize
        # compute angles in PC space
        pca, pca2 = [MultivariateStats.fit(PCA, (beta .- mean(beta, dims = 1))'; maxoutdim=3) for beta = [betas, betas2]]
        Za,Zb = [predict(pca, (vals .- mean(vals, dims = 1))') for vals = [jacs, betas]] # project into PC space
        Za2,Zb2 = [predict(pca2, (vals .- mean(vals, dims = 1))') for vals = [jacs[inds2,:], betas2]] # project into PC space
        Za,Za2,Zb,Zb2 = [arr ./ sqrt.(sum(arr.^2, dims = 2)) for arr = [Za, Za2, Zb, Zb2]] # normalize
        push!(meanspca[ij], mean(sum(Za .* Zb, dims = 2)[:, 1, :], dims = 1)[1]) # cos \theta for alpha_PG1
        push!(meanspca2[ij], mean(sum(Za2 .* Zb2, dims = 2)[:, 1, :], dims = 1)[1]) # cos \theta for alpha_PG2
    end
end
meanspca, meanspca2 = [reduce(hcat, ms) for ms = [meanspca, meanspca2]] # concatenate results

# plot results
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.49, right=0.76, bottom = 0.0, top = 1.0, wspace=0.25)
labs = repeat([L"$1^\mathrm{st}$ action", L"$2^\mathrm{nd}$ action"], outer = 2)
global limy = nothing # instantiate ylim parameter
for (ires, res) = enumerate([meanspca, meanspca2]) # first action and seconnd action
    ax = fig.add_subplot(grids[1,ires])
    res = res[:, [1; ires+1]] # data for this action
    mus, ss = mean(res, dims = 1)[:], std(res, dims = 1)[:]/sqrt(length(seeds)) # summary statistics
    println("action $ires: mu = ", mus, ", sem = ", ss) # print result
    ax.bar([1;2], mus, yerr = ss, color = col_p, capsize = capsize) # bar plot
    # plot individual data points
    shifts = 1:size(res, 1); shifts = (shifts .- mean(shifts))/std(shifts)*0.2 # add some jitter
    ax.scatter([1 .+ shifts; 2 .+ shifts], [res[:, 1]; res[:, 2]], color = col_point, marker = ".", s = 15)
    ax.axhline(0, color = col_c, lw = 2) # baseline
    if ires == 1 # set some plotting parameters
        vmin, vmax = minimum(mus-ss), maximum(mus+ss)
        global limy = [vmin - 0.05*(vmax-vmin); vmax+0.05*(vmax-vmin)]
        ax.set_ylabel(L"$\cos \theta$", labelpad = -0.07)
    else
        global limy = limy
        ax.set_yticks([])
    end
    # set some labels etc.
    ax.set_xticks([1;2], [L"${\bf \alpha}^\mathrm{RNN}$", L"${\bf \alpha}^\mathrm{RNN}_\mathrm{ctrl}$"])
    ax.set_ylim(limy)
    ax.set_title(labs[ires], fontsize = fsize, pad = 0.04)
    ax.set_xlim(0.5, 2.5)
end

# Plot rollout frequency by network size

# load our stored data
@load "$datadir/rew_and_plan_by_n.bson" res_dict
# extract the relevant data
meanrews, pfracs, seeds, Nhiddens, epochs = [res_dict[k] for k = ["meanrews", "planfracs", "seeds", "Nhiddens", "epochs"]]

# only consider second epoch onwards
mms = mean(meanrews, dims = 2)[:, 1, 3:length(epochs)] # mean reward across seeds
sms = std(meanrews, dims = 2)[:, 1, 3:length(epochs)] / sqrt(length(seeds)) # standard error
mps = mean(pfracs, dims = 2)[:, 1, 3:length(epochs)] # mean rollout frequency across seeds
sps = std(pfracs, dims = 2)[:, 1, 3:length(epochs)] / sqrt(length(seeds)) # standard error

# plot the data
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.87, right=1.0, bottom = 0, top = 1.0, wspace=0.15)
ax = fig.add_subplot(grids[1,1])
ax.axhline(0.2, ls = "-", color = col_c) # baseline
for (ihid, Nhidden) = enumerate(Nhiddens) # for each network sizes
    frac = 0.45*(Nhidden - minimum(Nhiddens))/(maximum(Nhiddens) - minimum(Nhiddens)) .+ 0.76
    col = col_p * frac # colour
    # plot mean and standard error
    ax.plot(mms[ihid, :], mps[ihid, :], ls = "-", color = col, label = Nhidden)
    ax.fill_between(mms[ihid, :], mps[ihid, :]-sps[ihid, :], mps[ihid, :]+sps[ihid, :], color = col, alpha = 0.2)
end
# set some labels and other plotting parameters
ax.set_xlabel("mean reward")
ax.set_ylabel(L"$p$"*"(rollout)")
ax.set_ylim(0, 0.65)
ax.set_xticks([0;4;8])
ax.legend(frameon = false, fontsize = fsize_leg, handlelength=1.5, handletextpad=0.5, borderpad = 0.0,
        labelspacing = 0.05, loc = "lower center", bbox_to_anchor = (0.75, -0.035))

# save figure
savefig("./figs/fig_mechanism_neural.pdf", bbox_inches = "tight")
savefig("./figs/fig_mechanism_neural.png", bbox_inches = "tight")
close()

