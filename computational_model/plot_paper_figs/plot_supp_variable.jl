#This script plots Figure S8 of Jensen et al.

include("plot_utils.jl") #various global settings
using Flux

fig = figure(figsize = (12*cm, 3.0*cm))

#plot thinking times against the probability of performing a rollout
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.0, right=0.3, bottom = 0.0, top = 1.0, wspace=0.05)
ax = fig.add_subplot(grids[0,0])
#load results
@load "$(datadir)RT_predictions_variable_N100_Lplan8_$(plan_epoch).bson" data
allsims, RTs, pplans, dists = [data[k] for k = ["correlations"; "RTs_by_u"; "pplans_by_u"; "dists_by_u"]];
RTs, pplans, dists = [reduce(vcat, arr) for arr = [RTs, pplans, dists]]

bins = 0.1:0.05:0.8 #bin edges for histogram
xs = 0.5*(bins[1:length(bins)-1] + bins[2:end]) #bin centers
RTs_shuff = RTs[randperm(length(RTs))] #perform a shuffle
dat = [RTs[(pplans .>= bins[i]) .& (pplans .< bins[i+1])] for i = 1:length(bins)-1] #real data
#create shuffle
dat_shuff = [RTs_shuff[(pplans .>= bins[i]) .& (pplans .< bins[i+1])] for i = 1:length(bins)-1]
#data to plot
m, m_c = [[mean(d) for d = dat] for dat = [dat, dat_shuff]]
s, s_c = [[std(d)/sqrt(length(d)) for d = dat] for dat = [dat, dat_shuff]]

#plot the data
ax.bar(xs, m, color = col_p, width = 0.04, linewidth = 0, label = "data")
ax.errorbar(xs, m, yerr = s, fmt = "none", color = "k", capsize = 2, lw = 1.5)
ax.errorbar(xs, m_c, yerr = s_c, fmt = "-", color = col_c, capsize = 2, lw = 1.5, label = "shuffle")
ax.set_xlabel(L"$\pi$"*"(rollout)")
ax.set_ylabel("thinking time (ms)")
ax.set_yticks([0;50;100;150;200;250])
ax.set_ylim(0, 250)
ax.legend(frameon = false, fontsize = fsize_leg)

m = mean(allsims, dims = 1)[1:2]
s = std(allsims, dims = 1)[1:2] / sqrt(size(allsims, 1))
println("mean and sem of correlations: ", m, " ", s) #print results


# plot performance vs nplan

#start by loading and extracting data
@load "$datadir/perf_by_n_variable_N100_Lplan8.bson" res_dict
seeds = sort([k for k = keys(res_dict)])
Nseed = length(seeds)
ms1, ms2, bs, es1, es2 = [], [], [], [], []
for (is, seed) = enumerate(seeds) #for each model
    #time within trial, distance to goal, and policy
    dts, mindists, policies = [res_dict[seed][k] for k = ["dts"; "mindists"; "policies"]]
    #select episodes where the trial finished for all rollout numbers
    keepinds = findall((.~isnan.(sum(dts, dims = (1,3))[:])) .& (mindists[:, 2] .>= 0))
    new_dts = dts[:, keepinds, :]
    new_mindists = mindists[keepinds, 2]
    policies = policies[:, keepinds, :, :, :]
    #mean performance across episodes with (m1) and without (m2) rollout feedback
    m1, m2 = mean(new_dts[1,:,:], dims = 1)[:], mean(new_dts[2,:,:], dims = 1)[:]
    push!(ms1, m1); push!(ms2, m2); push!(bs, mean(new_mindists)) #also store optimal (bs)
    p1, p2 = policies[1, :, :, :, :], policies[2, :, :, :, :] #extract log policies
    p1, p2 = [p .- Flux.logsumexp(p, dims = 4) for p = [p1, p2]] #normalize
    e1, e2 = [-sum(exp.(p) .* p, dims = 4)[:, :, :, 1] for p = [p1, p2]] #entropy
    m1, m2 = [mean(e[:,:,1], dims = 1)[:] for e = [e1,e2]] #only consider entropy of first action
    push!(es1, m1); push!(es2, m2) #store entropies
end
#concatenate across seeds
ms1, ms2, es1, es2 = [reduce(hcat, arr) for arr = [ms1, ms2, es1, es2]]
# compute mean and std across seeds
m1, s1 = mean(ms1, dims = 2)[:], std(ms1, dims = 2)[:]/sqrt(Nseed)
m2, s2 = mean(ms2, dims = 2)[:], std(ms2, dims = 2)[:]/sqrt(Nseed)
me1, se1 = mean(es1, dims = 2)[:], std(es1, dims = 2)[:]/sqrt(Nseed)
me2, se2 = mean(es2, dims = 2)[:], std(es2, dims = 2)[:]/sqrt(Nseed)
nplans = (1:length(m1)) .- 1 #

# plot performance vs number of rollouts
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.41, right=0.62, bottom = 0.0, top = 1.0, wspace=0.50)
ax = fig.add_subplot(grids[1,1])
ax.plot(nplans,m1, ls = "-", color = col_p, label = "agent") #mean
ax.fill_between(nplans,m1-s1,m1+s1, color = col_p, alpha = 0.2) #standard error
plot([nplans[1]; nplans[end]], ones(2)*mean(bs), color = col_c, ls = "-", label = "optimal") #optimal baseline
legend(frameon = false, loc = "upper right", fontsize = fsize_leg, handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.05)
xlabel("# rollouts")
ylabel("steps to goal")
ylim(0.9*mean(bs), maximum(m1+s1)+0.1*mean(bs))
xticks([0;5;10;15])

# plot change in policy with successful and unsuccessful rollouts
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.75, right=1.00, bottom = 0, top = 1.0, wspace=0.10)
for i = 1:2 #rewarded and non-rewarded rollout
    ms = []
    for seed = seeds #iterate through random seeds
        @load "$(datadir)/variable_causal_N100_Lplan8_$(seed)_$(plan_epoch).bson" data
        #rollouts action probability under new and old policy
        p_simulated_actions, p_simulated_actions_old = data["p_simulated_actions"], data["p_simulated_actions_old"]
        #rollout probabilities
        p_initial_sim, p_continue_sim = data["p_initial_sim"], data["p_continue_sim"]
        p_simulated_actions ./= (1 .- p_continue_sim) #renormalize over actions
        p_simulated_actions_old ./= (1 .- p_initial_sim) #renormalize over actions
        inds = findall(.~isnan.(sum(p_simulated_actions, dims = 1)[:])) #make sure we have data for both scenarios
        push!(ms, [mean(p_simulated_actions_old[i, inds]); mean(p_simulated_actions[i, inds])]) #mean for new and old
    end
    ms = reduce(hcat, ms) #concatenate across seeds
    m3, s3 = mean(ms, dims = 2)[1:2], std(ms, dims = 2)[1:2] / sqrt(length(seeds)) #mean and sem across seeds

    # plot results
    ax = fig.add_subplot(grids[1,i])
    ax.bar(1:2, m3, yerr = s3, color = [col_p1, col_p2][i], capsize = capsize)
    # plot individual data points
    shifts = 1:size(ms, 2); shifts = (shifts .- mean(shifts))/std(shifts)*0.2
    ax.scatter([1 .+ shifts; 2 .+ shifts], [ms[1, :]; ms[2, :]], color = col_point, marker = ".", s = 15)
    ax.set_xticks(1:2, ["pre"; "post"])
    if i == 1 #successful rollout
        ax.set_ylabel(L"$\pi(\hat{a}_1)$", labelpad = 0)
        ax.set_title("succ.", fontsize = fsize)
        ax.set_yticks([0.1;0.3;0.5;0.7])
    else #unsuccessful rollout
        ax.set_title("unsucc.", fontsize = fsize)
        ax.set_yticks([])
    end
    #set some parameters
    ax.set_ylim(0.0, 0.8)
    ax.set_xlim([0.4; 2.6])
    ax.axhline(0.25, color = color = col_c, ls = "-")
end

# add labels and save
y1, y2 = 1.16, 0.46
x1, x2, x3 = -0.105, 0.33, 0.675
plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label, )
plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x3,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)

savefig("./figs/supp_variable_time.pdf", bbox_inches = "tight")
savefig("./figs/supp_variable_time.png", bbox_inches = "tight")
close()
