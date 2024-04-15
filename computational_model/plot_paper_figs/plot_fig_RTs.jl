#This script plots Figure 2 of Jensen et al.

include("plot_utils.jl") #various global settings

#get indices of participants to use
keep = get_human_inds(); Nkeep = length(keep)

#load human prior parameters
@load "$datadir/guided_lognormal_params_delta.bson" params
lognormal_params = params

bot, top = 0.62, 0.38 #some figure settings
fig = figure(figsize = (15*cm, 7.5*cm)) #create figure

#plot performance vs trial number
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.00, right=0.20, bottom = bot, top = 1.0, wspace=0.05)

#load human data
@load "$datadir/human_by_trial_play.bson" data
Rmin_h, μs_h, ss_h = data
μ_h = mean(μs_h[:, keep], dims = 2)[:] #mean across people
s_h = 2*std(μs_h[:, keep], dims = 2)[:] / sqrt(Nkeep) #95% CI across people

#load model data
μs_m = []
for seed = seeds #iterate over our models
    @load "$(datadir)/model_by_trial$(seed).bson" data
    global Rmin_m, μ_m, s_m, minval = data #data for this model
    μ_m[1] += 1 #starts from '0' by default
    push!(μs_m, μ_m)
end
μs_m = reduce(hcat, μs_m)
μ_m = mean(μs_m, dims = 2)[:] #mean across models
s_m = 2*std(μs_m, dims = 2)[:] / sqrt(length(seeds)) #95% CI

#plot this data
ax = fig.add_subplot(grids[0,0])
ax.plot(1:Rmin_h, μ_h, ls = "-", color = col_h, label = "human")
ax.fill_between(1:Rmin_h, μ_h-s_h, μ_h+s_h, color = col_h, alpha = 0.2)
ax.plot(1:Rmin_m, μ_m, ls = "-", color = col_p, label = "model")
ax.fill_between(1:Rmin_m, μ_m-s_m, μ_m+s_m, color = col_p, alpha = 0.2)
opt = ones(Rmin_h)*minval #optimal exploitation
opt[1] = 8.223333333333333 #optimal exploration
ax.plot(1:Rmin_h, opt, ls = "-", color = col_c, label = "optimal") #plot optimal
ax.set_xlabel("trial number")
ax.set_ylabel("steps to goal")
ax.legend(frameon = false, fontsize = fsize_leg, handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.3)
ax.set_xticks(1:Rmin_h)
ax.set_yticks(3:3:12)

#now plot distribution of response times

@load "$(datadir)RT_by_complexity_by_user_play.bson" data; #load human participant data
RTs_p, dists_p, all_trial_nums_p, all_trial_time_p = data;

TTs = [] #thinking times
for u = 1:length(RTs_p) #for each user

    #compute posterior mean thinking times
    initial, later = [lognormal_params[key][u, :] for key = ["initial"; "later"]] #prior parameters
    #posterior mean for initial action
    initial_post_mean(r) = calc_post_mean(r, muhat=initial[1], sighat=initial[2], deltahat=initial[3], mode = false)
    #posterior mean for later actions
    later_post_mean(r) = calc_post_mean(r, muhat=later[1], sighat=later[2], deltahat=later[3], mode = false)

    newRT = later_post_mean.(RTs_p[u]) #posterior mean for later actions
    newRT[:, :, 1] = initial_post_mean.(RTs_p[u][:, :, 1]) #use different parameters for first action
    push!(TTs, newRT) #store thinking times
end
#concatenate thinking and response times across users
cat_TTs = reduce(vcat, TTs[keep])[:]; cat_TTs = cat_TTs[.~isnan.(cat_TTs)];
cat_RTs = reduce(vcat, RTs_p[keep])[:]; cat_RTs = cat_RTs[.~isnan.(cat_RTs)];

#plot these distributions
grids = fig.add_gridspec(nrows=2, ncols=1, left=0.0, right=0.24, bottom = 0-0.01, top = top+0.03, wspace=0.05, hspace = 0.3)

bins = 0:40:800 #bins for histograms
for (idat, RT_dat) = enumerate([cat_TTs, cat_RTs]) #plot both thinking and response times
    ax = fig.add_subplot(grids[idat-1,0])
    ax.hist(RT_dat, bins = bins, color = "k")

    if idat == 1 #thinking times
        ax.set_xlabel("thinking time (ms)")
        ax.set_ylabel("              # actions (x1000)")
    else #response times
        ax.set_xlabel("response time (ms)", labelpad = 1)
        ax.set_xticks([])
    end
    #set some plot parameters
    ax.set_ylim(0, 40000)
    ax.set_yticks([0;20000;40000])
    ax.set_yticklabels([0;20;40])
    if idat == 1 ax.set_xticks([0;400;800]) end
    ax.set_xlim(bins[1], bins[end])
end

#plot thinking time vs. distance to goal and time within trial
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.36, right=0.60, bottom = bot, top = 1.0, wspace=0.05)
ax = fig.add_subplot(grids[0,0])

dists = 1:6 #distances to goal to plot
thinks = []
for u = keep #for each user
    dats_p = [TTs[u][(dists_p[u].==dist), :] for dist in dists] #thinking times for each distance
    mu = [] #list of means across actions for each distance
    for (idat, dat) = enumerate(dats_p) #for each distance
        if (size(dat, 1) > 0.5) #if we have some data
            push!(mu, nanmean(dat, dims = 1)[:]) #mean for this user
        else
            push!(mu, ones(size(dat, 2))*NaN) #no data for this user
        end
    end
    push!(thinks, reduce(hcat, mu)') #store results for this user
end
#compute summary statistics
cat3(a, b) = cat(a, b, dims = 3)
thinks = reduce(cat3, thinks)[:, 1:10, :] #thinking time
m = nanmean(thinks, dims = 3)[:, :, 1] #mean across participants
Ns = sum(.~isnan.(thinks), dims = 3)[:, :, 1] #number of participants for each data point
s = nanstd(thinks, dims = 3)[:, :, 1] ./ sqrt.(Ns) #standard error
plotdists = 2:5 #distances to go to plot

#plot results for human participants
cols = [[0;0;0], [0.35;0.35;0.35], [0.54;0.54;0.54], [0.7;0.7;0.7]]
for (idist, dist) = enumerate(plotdists) #for each distance to goal
    ts = 1:dist #times within trial to plot
    label = "goal dist = $dist" #label for legend
    ax.plot(ts, m[dist,ts], ls = "-", color = cols[idist], label = label)
    ax.fill_between(
        ts, m[dist,ts] - s[dist,ts],
        m[dist,ts] + s[dist,ts], color = cols[idist],
        alpha = 0.2,)
end
ax.legend(frameon=false, fontsize = fsize_leg, loc = "upper right", handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.3)
#set some labels etc.
ax.set_xlabel("step within trial")
ax.set_ylabel("thinking time (ms)")
ax.set_xticks([1;3;5])
ax.set_title("human", fontsize = fsize)
ax.set_ylim([50; 250])
ax.set_yticks([50;150;250])


# load results for the RL agent
t_per_rollout = 120 #each rollout is 120 ms
μs = [] #mean for each situation
for seed = seeds #load data for each RL agent
    @load "$(datadir)model_RT_by_complexity$(seed)_$plan_epoch.bson" data
    global dists, dats = data
    μ = [nanmean(dat, dims = 1) for dat in dats] #mean across each situation
    push!(μs, μ)
end
μs = [reduce(vcat, [μs[s][i] for s = 1:length(seeds)]) for i = 1:length(dists)] #concatenate across agents
μm = [(mean(μ, dims = 1)[:] .- 1)*t_per_rollout for μ = μs] #subtract 1 since 1 iteration corresponds to zero thinking
ssm = [std(μ, dims = 1)[:]/sqrt(length(seeds))*t_per_rollout for μ = μs] #standard error

# plot results for our RL agent
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.76, right=1.0, bottom = bot, top = 1.0, wspace=0.05)
ax = fig.add_subplot(grids[0,0])
plotdists = 2:5 #distances to goal to plot
cols = [[0.00;0.19;0.52], [0.19;0.39;0.72], [0.34;0.54;0.87], [0.49;0.69;1.0]]
for (idist, dist) = enumerate(plotdists)
    label = "goal dist = $dist"
    ts = 1:dist
    ax.plot(ts, μm[dist][ts], ls = "-", color = cols[idist], label = label)
    ax.fill_between(
        ts,
        μm[dist][ts] - ssm[dist][ts],
        μm[dist][ts] + ssm[dist][ts],
        color = cols[idist],
        alpha = 0.2,
    )
end
#set some labels
ax.legend(frameon=false, fontsize = fsize_leg, loc = "upper right", handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.3)
ax.set_xlabel("step within trial")
ax.set_ylabel("thinking time (ms)")
ax.set_xticks([1;3;5])
ax.set_title("model", fontsize = fsize)
ax.set_ylim([0; 200])
ax.set_yticks([0;100;200])

#plot thinking times against the probability of performing a rollout
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.38, right=0.65, bottom = 0.0, top = top, wspace=0.05)
ax = fig.add_subplot(grids[0,0])
#load results
@load "$(datadir)RT_predictions_N100_Lplan8_$(plan_epoch).bson" data
allsims, RTs, pplans, dists = [data[k] for k = ["correlations"; "RTs_by_u"; "pplans_by_u"; "dists_by_u"]];
RTs, pplans, dists = [reduce(vcat, arr) for arr = [RTs, pplans, dists]]

bins = 0.05:0.05:0.8 #bin edges for histogram
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

#compute and print residual correlations
RTs_by_u, pplans_by_u, dists_by_u = [data[k] for k = ["RTs_by_u", "pplans_by_u", "dists_by_u"]];
allcors = []
for u = 1:length(RTs_by_u) #for each user
    mean_sub_RTs, mean_sub_pplans = [], [] #residual arrays
    for dist = 1:20 #for each distance-to-goal
        inds = (dists_by_u[u] .== dist) #all data points at this distance
        if sum(inds) >= 2 #require at least 2 data points at this distance
            new_RTs, new_pplans = RTs_by_u[u][inds], pplans_by_u[u][inds] #extract RTs and pi(rollout)
            mean_sub_RTs = [mean_sub_RTs; new_RTs .- mean(new_RTs)] #subtract mean of RTs and append
            mean_sub_pplans = [mean_sub_pplans; new_pplans .- mean(new_pplans)] #subtract mean of pi(rollout) and append
        end
    end
    push!(allcors, cor(mean_sub_RTs, mean_sub_pplans)) #append the residual correlation for this user
end
println("residual correlations: ", mean(allcors), " ", std(allcors)/sqrt(length(allcors)))
allsims[:, 3] = allcors; #add residual correlation to list of stuff to be plotted

#plot bar plot of correlations
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.80, right=1.0, bottom = 0.0, top = top, wspace=0.05)
ax = fig.add_subplot(grids[0,0])
m = mean(allsims, dims = 1)[:]
s = std(allsims, dims = 1)[:] / sqrt(size(allsims, 1))
ax.bar(1:3, m, yerr = s, color = [col_p, col_c, col_c], capsize = capsize)
ax.set_yticks([0;0.1;0.2;0.3])

#plot individual data points
for i_n = 1:3 #for each bar
    corrs = allsims[:, i_n] #individual correlations
    shifts = 1:length(corrs); shifts = (shifts .- mean(shifts))/std(shifts)*0.15
    ax.scatter(i_n .+ shifts, corrs, color = col_point, marker = ".", s = 3, alpha = 0.5, zorder = 100)
end
ax.set_yticks([-0.1;0;0.1;0.2;0.3;0.4])
ax.set_xticks(1:3)
ax.set_xticklabels([L"$\pi$"*"(rollout)"; "goal dist"; "residual"], rotation = 45, ha = "right", rotation_mode = "anchor")
ax.set_ylabel("correlation with\nthinking time")
ax.set_xlim(0.25, 3.75)
println("mean and sem of correlations: ", m, " ", s) #print results

#perform a shuffle test for reporting chance level correlations
N = 100; shuffs = zeros(N)
for n = 1:N #for each shuffle
    new = zeros(Nkeep) #array for randomized data
    for u = 1:Nkeep new[u] = cor(pplans_by_u[u], shuffle(RTs_by_u[u])) end
    shuffs[n] = mean(new) #mean randomized correlation
end
println("shuffled correlation: ", mean(shuffs), " ", std(shuffs))
#also print true correlation
real = zeros(Nkeep)
for u = 1:Nkeep real[u] = cor(pplans_by_u[u], RTs_by_u[u]) end
println("real correlation: ", mean(real), " ", std(real)/sqrt(length(real)))

### add labels and save ###
y1, y2 = 1.07, 0.46
x1, x2, x3 = -0.07, 0.28, 0.65
plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label, )
plt.text(x2-0.04,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x3-0.02,y1,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x1-0.01,y2,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x2-0.01,y2,"E";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x3+0.02,y2,"F";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)

savefig("./figs/fig_RTs.pdf", bbox_inches = "tight")
savefig("./figs/fig_RTs.png", bbox_inches = "tight")
close()


