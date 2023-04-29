#This script plots Figure 2 of Jensen et al.

include("plot_utils.jl") #various global settings

#get indices of participants to use
keep = get_human_inds(); Nkeep = length(keep)

#load human prior parameters
@load "$datadir/guided_lognormal_params_delta.bson" params
lognormal_params = params

bot, top = 0.62, 0.38 #some figure settings
fig = figure(figsize = (15*cm, 7.5*cm)) #create figure

### performance vs trial number ###

grids = fig.add_gridspec(nrows=1, ncols=1, left=0.00, right=0.20, bottom = bot, top = 1.0, wspace=0.05)
@load "$(datadir)/human_by_trial_play.bson" data
Rmin_h, μs_h, ss_h = data
μ_h = mean(μs_h[:, keep], dims = 2)[:]
s_h = 2*std(μs_h[:, keep], dims = 2)[:] / sqrt(Nkeep)

μs_m = []
for seed = seeds
    @load "$(datadir)/model_by_trial$(prior)$(seed).bson" data
    global Rmin_m, μ_m, s_m, minval = data
    μ_m[1] += 1 #starting from zero
    push!(μs_m, μ_m)
end
μs_m = reduce(hcat, μs_m)
μ_m = mean(μs_m, dims = 2)[:]
s_m = 2*std(μs_m, dims = 2)[:] / sqrt(length(seeds))

ax = fig.add_subplot(grids[0,0])
ax.plot(1:Rmin_h, μ_h, ls = "-", color = col_h, label = "human")
ax.fill_between(1:Rmin_h, μ_h-s_h, μ_h+s_h, color = col_h, alpha = 0.2)
ax.plot(1:Rmin_m, μ_m, ls = "-", color = col_p, label = "model")
ax.fill_between(1:Rmin_m, μ_m-s_m, μ_m+s_m, color = col_p, alpha = 0.2)
opt = ones(Rmin_h)*minval
opt[1] = 8.223333333333333 #optimal exploration
ax.plot(1:Rmin_h, opt, ls = "-", color = col_c, label = "optimal")
ax.set_xlabel("trial number")
ax.set_ylabel("steps to goal")
ax.legend(frameon = false, fontsize = fsize_leg, handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.3)
ax.set_xticks(1:Rmin_h)
ax.set_yticks(3:3:12)

### distribution of response times ###

#@load "$(datadir)/human_RT_and_rews_play.bson" data
#all_RTs = data["all_RTs"]
#RTs = reduce(vcat, data["all_RTs"][keep])[:]

@load "$(datadir)RT_by_complexity_by_user_play.bson" data;
RTs_p, dists_p, all_trial_nums_p, all_trial_time_p = data;

TTs = []
for u = 1:length(RTs_p)
    newRT = copy(RTs_p[u])
    newRT .-= action_times[u]
    newRT[:, :, 1] .-= process_times[u]

    if weiji
        initial, later = [lognormal_params[key][u, :] for key = ["initial"; "later"]]
        #posterior mean for initial action
        initial_post_mean(r) = calc_post_mean(r, muhat=initial[1], sighat=initial[2], deltahat=initial[3], mode = false)
        #posterior mean for later actions
        later_post_mean(r) = calc_post_mean(r, muhat=later[1], sighat=later[2], deltahat=later[3], mode = false)

        newRT = later_post_mean.(RTs_p[u]) #posterior mean
        newRT[:, :, 1] = initial_post_mean.(RTs_p[u][:, :, 1]) #use different parameters for first action
    end
    @assert sum(isnan.(newRT)) == sum(isnan.(RTs_p[u]))

    push!(TTs, newRT)
end
cat_TTs = reduce(vcat, TTs[keep])[:]; cat_TTs = cat_TTs[.~isnan.(cat_TTs)];
cat_RTs = reduce(vcat, RTs_p[keep])[:]; cat_RTs = cat_RTs[.~isnan.(cat_RTs)];

#grids = fig.add_gridspec(nrows=2, ncols=1, left=0.36, right=0.60, bottom = bot-0.01, top = 1.03, wspace=0.05, hspace = 0.3)
grids = fig.add_gridspec(nrows=2, ncols=1, left=0.0, right=0.24, bottom = 0-0.01, top = top+0.03, wspace=0.05, hspace = 0.3)

bins = -180:40:1000
if weiji bins = 0:40:800 end
for (idat, RT_dat) = enumerate([cat_TTs, cat_RTs])
    ax = fig.add_subplot(grids[idat-1,0])
    ax.hist(RT_dat, bins = bins, color = "k")
    if idat == 1 y0 = 23000 else y0 = 13500 end
    if ~weiji
        plt.arrow(mean(RT_dat), y0, 0, -6000, color = col_c, head_length = 2000, head_width = 40, lw = 3, length_includes_head = true)
    end
    #ax.axvline(mean(RT_dat), color = col_c, ls = "--")
    if idat == 1
        ax.set_xlabel("thinking time (ms)")
        ax.set_ylabel("              # actions (x1000)")
    else
        ax.set_xlabel("response time (ms)", labelpad = 1)
        ax.set_xticks([])
    end
    ax.set_yticks([0;10000;20000], [0;10;20])
    ax.set_ylim(0, 25000)
    if weiji
        ax.set_ylim(0, 40000)
        ax.set_yticks([0;20000;40000], [0;20;40])
        if idat == 1 ax.set_xticks([0;400;800]) end
    end
    ax.set_xlim(bins[1], bins[end])
end

### TT vs dist to goal (human) ###

#grids = fig.add_gridspec(nrows=1, ncols=1, left=0.76, right=1.0, bottom = bot, top = 1.0, wspace=0.05)
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.36, right=0.60, bottom = bot, top = 1.0, wspace=0.05)

ax = fig.add_subplot(grids[0,0])
dists = 1:6

thinks = []
for u = keep
    #println(u)
    dats_p = [TTs[u][(dists_p[u].==dist), :] for dist in dists]
    mu = []
    for (idat, dat) = enumerate(dats_p)
        if (size(dat, 1) > 0.5)
            push!(mu, nanmean(dat, dims = 1)[:])
        else
            push!(mu, ones(size(dat, 2))*NaN)
        end
    end
    push!(thinks, reduce(hcat, mu)')
end
cat3(a, b) = cat(a, b, dims = 3)
thinks = reduce(cat3, thinks)[:, 1:10, :]
m = nanmean(thinks, dims = 3)[:, :, 1]
Ns = sum(.~isnan.(thinks), dims = 3)[:, :, 1]
s = nanstd(thinks, dims = 3)[:, :, 1] ./ sqrt.(Ns)
plotdists = 2:5
global all_vals_h = []

cols = [[0;0;0], [0.35;0.35;0.35], [0.54;0.54;0.54], [0.7;0.7;0.7]]
for (idist, dist) = enumerate(plotdists)
    ts = 1:dist
    if dist % 1 == 0 label = "goal dist = $dist" else label = nothing end
    col = ones(3) * (dist - minimum(plotdists)) / maximum(plotdists .- 1) * 0.7
    ax.plot(ts, m[dist,ts], ls = "-", color = cols[idist], label = label)
    ax.fill_between(
        ts, m[dist,ts] - s[dist,ts],
        m[dist,ts] + s[dist,ts], color = cols[idist],
        alpha = 0.2,)
    global all_vals_h = [all_vals_h; m[dist, ts]]
end
ax.legend(frameon=false, fontsize = fsize_leg, loc = "upper right", handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.3)
ax.set_xlabel("step within trial")
ax.set_ylabel("thinking time (ms)")
ax.set_xticks([1;3;5])
ax.set_title("human", fontsize = fsize)

tt_ylims = [0; 230]
tt_yticks = [0;100;200]
if weiji
    tt_ylims = [0;200]
    ax.set_ylim(tt_ylims .+ 50)
    ax.set_yticks(tt_yticks .+ 50)
else
    ax.set_ylim(tt_ylims)
    ax.set_yticks(tt_yticks)
end

### RT vs dist to goal (model) ###

t_per_rollout, baseline = 120, 1 #ms
by_step = false#true
if by_step t_per_rollout, baseline = 120/6, 0 end

μs = []
for seed = seeds
    if by_step
        @load "$(datadir)model_RT_by_complexity_bystep$(prior)$(seed)_$plan_epoch.bson" data
    else
        @load "$(datadir)model_RT_by_complexity$(prior)$(seed)_$plan_epoch.bson" data
    end
    global dists, dats = data

    μ = [nanmean(dat, dims = 1) for dat in dats]
    #s = [nanstd(dat, dims = 1) ./ sqrt.(sum(1 .- isnan.(dat), dims = 1)) for dat in dats]
    push!(μs, μ)
end
μs = [reduce(vcat, [μs[s][i] for s = 1:length(seeds)]) for i = 1:length(dists)]
μm = [(mean(μ, dims = 1)[:] .- baseline )*t_per_rollout for μ = μs] #baseline is how many steps correspond to zero thinking
ssm = [std(μ, dims = 1)[:]/sqrt(length(seeds))*t_per_rollout for μ = μs]

#grids = fig.add_gridspec(nrows=1, ncols=1, left=0.0, right=0.24, bottom = 0.0, top = top, wspace=0.05)
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.76, right=1.0, bottom = bot, top = 1.0, wspace=0.05)

ax = fig.add_subplot(grids[0,0])
plotdists = 2:5
global all_vals_m = []
cols = [[0.00;0.19;0.52], [0.19;0.39;0.72], [0.34;0.54;0.87], [0.49;0.69;1.0]]
for (idist, dist) = enumerate(plotdists)
    if dist % 1 == 0 label = "goal dist = $dist" else label = nothing end
    ts = 1:dist
    frac = (dist - minimum(plotdists))/(maximum(plotdists) - minimum(plotdists))
    frac = (0.45*frac .+ 0.76)
    col = col_p * frac
    ax.plot(ts, μm[dist][ts], ls = "-", color = cols[idist], label = label)
    ax.fill_between(
        ts,
        μm[dist][ts] - ssm[dist][ts],
        μm[dist][ts] + ssm[dist][ts],
        color = cols[idist],
        alpha = 0.2,
    )
    global all_vals_m = [all_vals_m; μm[dist][ts]]
end
ax.legend(frameon=false, fontsize = fsize_leg, loc = "upper right", handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.3)
ax.set_xlabel("step within trial")
ax.set_ylabel("thinking time (ms)")
ax.set_xticks([1;3;5])
#if ~by_step ax.set_ylim(0, 200) end
ax.set_title("model", fontsize = fsize)
ax.set_ylim(tt_ylims)
ax.set_yticks(tt_yticks)

println("correlation of means: ", cor(all_vals_h, all_vals_m))

### RT vs p(roll) ###

grids = fig.add_gridspec(nrows=1, ncols=1, left=0.38, right=0.65, bottom = 0.0, top = top, wspace=0.05)
ax = fig.add_subplot(grids[0,0])

#@load "$(datadir)RT_predictions.bson" data
if weiji
    @load "$(datadir)RT_predictions_N100_Lplan8$(prior)_$(plan_epoch)_weiji.bson" data
else
    @load "$(datadir)RT_predictions_new$(prior)_$plan_epoch.bson" data
end
allsims, RTs, pplans, dists, steps = [data[k] for k = ["correlations"; "RTs"; "pplans"; "dists"; "steps"]];
bins = 0.05:0.05:0.8
xs = 0.5*(bins[1:length(bins)-1] + bins[2:end])

RTs_shuff = RTs[randperm(length(RTs))]
dat = [RTs[(pplans .>= bins[i]) .& (pplans .< bins[i+1])] for i = 1:length(bins)-1]
dat_shuff = [RTs_shuff[(pplans .>= bins[i]) .& (pplans .< bins[i+1])] for i = 1:length(bins)-1]

m, m_c = [[mean(d) for d = dat] for dat = [dat, dat_shuff]]
s, s_c = [[std(d)/sqrt(length(d)) for d = dat] for dat = [dat, dat_shuff]]

N = [length(d) for d = dat]
#ax.errorbar(xs, m, yerr = s, fmt = "", color = col_p, capsize = 4)
ax.bar(xs, m, color = col_p, width = 0.04, linewidth = 0, label = "data")
#ax.errorbar(xs, m, yerr = s, fmt = "none", color = col_p, capsize = 0, lw = 0, label = "data")
ax.errorbar(xs, m, yerr = s, fmt = "none", color = "k", capsize = 2, lw = 1.5)
ax.errorbar(xs, m_c, yerr = s_c, fmt = "-", color = col_c, capsize = 2, lw = 1.5, label = "shuffle")
ax.set_xlabel(L"$\pi$"*"(rollout)")
ax.set_ylabel("thinking time (ms)")
if weiji
    ax.set_yticks([0;50;100;150;200;250])
    ax.set_ylim(0, 250)
else
    ax.set_yticks([0;50;100;150;200])
    ax.set_ylim(0, 200)
end
ax.legend(frameon = false, fontsize = fsize_leg)

### correlations ###

#compute residual
RTs_by_u, pplans_by_u, dists_by_u, steps_by_u = [data[k] for k = ["RTs_by_u", "pplans_by_u", "dists_by_u", "steps_by_u"]];
allcors = []
for u = 1:length(RTs_by_u)
    mean_sub_RTs, mean_sub_pplans = [], []
    for dist = 1:20 #for each distance-to-goal
        inds = (dists_by_u[u] .== dist) #all data points at this distance
        #for xstep = 1:100
        #inds = (dists_by_u[u] .== dist) .& (steps_by_u[u] .== -xstep)
        if sum(inds) >= 2 #require at least 2 data points
            new_RTs, new_pplans = RTs_by_u[u][inds], pplans_by_u[u][inds] #find the corresponding RTs and pi(rollout)
            mean_sub_RTs = [mean_sub_RTs; new_RTs .- mean(new_RTs)] #subtract mean of RTs and append
            mean_sub_pplans = [mean_sub_pplans; new_pplans .- mean(new_pplans)] #subtract mean of pi(rollout) and append
        #end
        end
    end
    push!(allcors, cor(mean_sub_RTs, mean_sub_pplans))
    #println(allcors[end]) #compute residual correlation: 0.043
end
println("residual correlations: ", mean(allcors), " ", std(allcors)/sqrt(length(allcors)))
allsims[:, 3] = allcors; #residual instead of step


grids = fig.add_gridspec(nrows=1, ncols=1, left=0.80, right=1.0, bottom = 0.0, top = top, wspace=0.05)
ax = fig.add_subplot(grids[0,0])
n = 3
m = mean(allsims, dims = 1)[1:n]
s = std(allsims, dims = 1)[1:n] / sqrt(size(allsims, 1))
ax.bar(1:n, m, yerr = s, color = [col_p, col_c, col_c][1:n], capsize = capsize)
ax.set_yticks([0;0.1;0.2;0.3])

if plot_points
    for i_n = 1:n
        corrs = allsims[:, i_n]; col = col_point
        shifts = 1:length(corrs); shifts = (shifts .- mean(shifts))/std(shifts)*0.15
        ax.scatter(i_n .+ shifts, corrs, color = col, marker = ".", s = 3, alpha = 0.5)
    end
    ax.set_yticks([-0.1;0;0.1;0.2;0.3;0.4])
end
#ax.set_xticks(1:n, [L"$\pi$"*"(rollout)"; "goal dist"; "step number"][1:n], rotation = 45, ha = "right", rotation_mode = "anchor")
ax.set_xticks(1:n, [L"$\pi$"*"(rollout)"; "goal dist"; "residual"][1:n], rotation = 45, ha = "right", rotation_mode = "anchor")
ax.set_ylabel("correlation with\nthinking time")
ax.set_xlim(0.25, n + 0.75)
println("mean and sem of correlations: ", m, " ", s)


N = 100; shuffs = zeros(N); pplans_by_u = data["pplans_by_u"]; RTs_by_u = data["RTs_by_u"];
for n = 1:N
    new = zeros(Nkeep)
    for u = 1:Nkeep new[u] = cor(pplans_by_u[u], shuffle(RTs_by_u[u])) end
    shuffs[n] = mean(new)
end
println("shuffled correlation: ", mean(shuffs), " ", std(shuffs))
real = zeros(Nkeep)
for u = 1:Nkeep real[u] = cor(pplans_by_u[u], RTs_by_u[u]) end
println("real correlation: ", mean(real), " ", std(real)/sqrt(length(real)))

### add labels and save ###

add_labels = true
if add_labels
    y1 = 1.07
    y2 = 0.46
    x1, x2, x3 = -0.07, 0.28, 0.65
    fsize = fsize_label
    plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
    #plt.text(x2-0.02,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    #plt.text(x3-0.01,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    #plt.text(x1,y2,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x2-0.04,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x3-0.02,y1,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x1-0.01,y2,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x2-0.01,y2,"E";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x3+0.02,y2,"F";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
end

savefig("./figs/fig_RTs$(prior).pdf", bbox_inches = "tight")
savefig("./figs/fig_RTs$(prior).png", bbox_inches = "tight")
close()


