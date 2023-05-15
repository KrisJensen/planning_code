# This script plots Figure 4 of Jensen et al.
# By default, only model results are plotted since the experimental results require data from Widloski et al.
include("plot_utils.jl")

# load experimental replay data
if plot_experimental_replays rnames, exp_dict = load_exp_data() end

# load model data
@load datadir * "model_replay_analyses.bson" res_dict;
mod_dict = res_dict
seeds = sort([k for k = keys(mod_dict)]) # seeds used for our model analyses
Nseed = length(seeds) # number of models to consider

bot, top = 0.62, 0.38 # bottom of top row and top of bottom row
fig = figure(figsize = (15*cm, 7.5*cm)) # create figure

if plot_experimental_replays
    # plot example replays
    example_rep = load_pickle(replaydir*"figs/replay/widloski_Billy3_20181207_s1_examples.p") # load a session
    reps = example_rep["reps"]; walls = example_rep["walls"]; home = example_rep["home"] # arena properties

    grids = fig.add_gridspec(nrows=1, ncols=1, left=-0.04, right=0.22, bottom = bot-0.08, top = 1.0, wspace=0.15)
    ax = fig.add_subplot(grids[1,1])
    #plot arena
    for x = (0:5).-0.5 # basic grid
        ax.plot([x;x], [-0.5;4.5], color = "k", lw = lw_arena*0.8)
        ax.plot([-0.5;4.5], [x;x], color = "k", lw = lw_arena*0.8)
    end
    for x = 1:3
        for y = 1:3
            ax.scatter([x], [y], s = 200, marker = ".", color = "k") # wells
        end
    end
    ax.axis("off")
    for w = walls #plot the walls
        plot(w[:,1], w[:,2], color = "k", lw = lw_wall*4/5)
    end
    cols = [col_p1, col_p1, col_p2, col_p2] # line colours
    labels = ["successful"; nothing; "unsuccessful"; nothing] # legend labels
    lw_rep = 2
    for (irep, rep) = enumerate(reps) # for each example replay
        rep = rep .+ 0.7*(irep/(length(reps)+1) .- 0.5) # shift for visibility
        plot(rep[:,1], rep[:,2], color = cols[irep], label = labels[irep], lw = lw_rep, zorder = 800) # plot replay
        ax.scatter([rep[1,1]], [rep[1,2]], color = cols[irep], s = 150, zorder = 1000, marker = ".") # plot initial location
    end
    ax.plot([home[1]], [home[2]], color = "k", marker="x", markersize=12*0.8, ls="", mew=3) # plot home well
    ax.legend(frameon = false, fontsize = fsize_leg, loc = "upper center", bbox_to_anchor = (0.5, 1.20), handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.05)
end


# Plot frequency of replays crossing walls

grids = fig.add_gridspec(nrows=1, ncols=2, left=0.35, right=0.60, bottom = bot, top = 1.0, wspace=0.15)
ylims = [0;0.45] # y axis limits

if plot_experimental_replays
    batch_wall_probs = [exp_dict[n]["wall_crossings"][1] for n = rnames] # wall crossings for each session
    batch_rand_wall_probs = [mean(exp_dict[n]["wall_crossings"][2:end]) for n = rnames] # now for control wall locations
    inds = findall((batch_wall_probs .> 0) .|| (batch_rand_wall_probs .> 0)) # filter out sessions with no walls for this analysis
    batch_wall_probs, batch_rand_wall_probs = batch_wall_probs[inds], batch_rand_wall_probs[inds]
    μ, s = mean(batch_wall_probs), std(batch_wall_probs)/sqrt(length(batch_wall_probs)) # summary statistics
    μr, sr = mean(batch_rand_wall_probs), std(batch_rand_wall_probs)/sqrt(length(batch_rand_wall_probs)) # summary statistics

    # plot experimental results
    ax = fig.add_subplot(grids[1,1])
    plot_comparison(ax, [batch_wall_probs batch_rand_wall_probs]; xticklabs=["true", "ctrl"], ylab = L"$p$"*"(cross wall)", col2 = col_c, ylims = ylims, plot_title = "experiment")

    # report results and significance
    rands, trueval = permutation_test(batch_rand_wall_probs, batch_wall_probs)
    println("walls: d = ", round(trueval, digits = 2) ,", p = ", mean(rands .> trueval))
end

# repeat analysis for computational model
batch_wall_probs = [mean(mod_dict[seed]["batch_wall_probs"]) for seed = seeds] # fraction crossing walls
batch_rand_wall_probs = [mean(mod_dict[seed]["batch_rand_wall_probs"]) for seed = seeds] # wall crossings in resampled environments

# compute summary statistics and plot
μ, s = mean(batch_wall_probs), std(batch_wall_probs)/sqrt(Nseed) # true values
μr, sr = mean(batch_rand_wall_probs), std(batch_rand_wall_probs)/sqrt(Nseed) # control values
ax = fig.add_subplot(grids[1,2])
plot_comparison(ax, [batch_wall_probs batch_rand_wall_probs]; xticklabs=["true", "ctrl"], ylab = nothing, col = col_p, col2 = 1.2*col_p, ylims = ylims, plot_title = "model", yticks = [])


# plot frequency of replays reaching the goal location

grids = fig.add_gridspec(nrows=1, ncols=2, left=0.75, right=1.00, bottom = bot, top = 1.0, wspace=0.15)
ylims = [0;0.8]

if plot_experimental_replays
    true_succs = [exp_dict[n]["goaldir_freq"][1] for n = rnames] # frequency of reaching true goal
    false_succs = [nanmean(exp_dict[n]["goaldir_freq"][2]) for n = rnames] # frequency of reaching ctrl goals
    μ, s = mean(true_succs), std(true_succs)/sqrt(length(true_succs)) # mean and sem
    μr, sr = mean(false_succs), std(false_succs)/sqrt(length(false_succs)) # ctrl mean and sem
    ax = fig.add_subplot(grids[1,1])
    plot_comparison(ax, [true_succs false_succs]; xticklabs=["true", "ctrl"], ylab = L"$p$"*"(goal)", col2 = col_c, ylims = ylims, plot_title = "experiment")

    # print result and stats
    rands, trueval = permutation_test(true_succs, false_succs)
    println("goal: d = ", round(trueval, digits = 2) ,", p = ", mean(rands .> trueval))
end

# repeat analyses for computational model
true_succs = [mean(mod_dict[seed]["true_succs"]) for seed = seeds] # frequency of replays reaching goal
false_succs = [mean(mod_dict[seed]["false_succs"]) for seed = seeds] # frequency of reaching ctrl goal location
μ, s = mean(true_succs), std(true_succs)/sqrt(Nseed) # true summary statistics
μr, sr = mean(false_succs), std(false_succs)/sqrt(Nseed) # ctrl summary statistics
# plot result
ax = fig.add_subplot(grids[1,2])
plot_comparison(ax, [true_succs false_succs]; xticklabs=["true", "ctrl"], ylab = nothing, col = col_p, col2 = 1.2*col_p, ylims = ylims, plot_title = "model", yticks = [])

# plot frequency of 'following' the replayed action

grids = fig.add_gridspec(nrows=1, ncols=3, left=0.0, right=0.40, bottom = 0.0, top = top, wspace=0.15)
ylims = [0; 1.02]; ticks = 0:0.5:1.0

if plot_experimental_replays
    # instantiate some arrays
    all_next_as, all_sim_as, all_succs, all_trialnums = [], [], [], []
    all_succsc = [[] for _ = 1:7] # 'success' for control locations
    all_ns = [] # session number
    for (i_n, name) = enumerate(rnames) #for each session
        # load some data
        trialnums, succs, next_as, sim_as = [exp_dict[name]["perf_by_rep"][k] for k = ["trialnums", "success", "a0s", "rep_a0s"]]
        succsc = exp_dict[name]["perf_by_rep"]["success_ctrl"] # 'success' to control locations
        ntrial = length(trialnums) #number of trials in this session
        push!(all_succs, reduce(vcat, succs)) # store whether replays were successful
        push!(all_sim_as, reduce(vcat, sim_as)) # store the replayed actions
        push!(all_trialnums, reduce(vcat, [trialnums[i]*ones(length(succs[i])) for i = 1:ntrial])) # trial numbers
        push!(all_next_as, reduce(vcat, [next_as[i]*ones(length(succs[i])) for i = 1:ntrial])) # next physical actions
        push!(all_ns, ones(length(all_succs[end]))*i_n) # session numbers
        for i = 1:7 # for each control location
            push!(all_succsc[i], reduce(vcat, succsc[i, :])) # was the replay successful to this location?
        end
    end
    # concatenate the data into an array
    all_next_as, all_sim_as, all_succs, all_trialnums, all_ns = [reduce(vcat, arr) for arr = [all_next_as, all_sim_as, all_succs, all_trialnums, all_ns]]
    all_succsc = [reduce(vcat, succsc) for succsc in all_succsc]
    follow = Float64.(all_next_as .== all_sim_as) # did we physically follow the replayed action?

    for ictrl = [0, 1] # whether to consider home (0) or away (1) trials
        dats = zeros(length(rnames), 2) # initialize data array
        for n = 1:length(rnames) # for each session
            inds_s = findall((all_succs .== 1) .& ((all_trialnums .% 2) .== ictrl) .& (all_ns .== n)) #indices of successful replays
            # only consider unsuccessful replays that were successful to some location to try to match statistics a bit
            inds_n = [findall((succsc .== 1) .& ((all_trialnums .% 2) .== ictrl) .& (all_ns .== n)) for succsc = all_succsc]
            inds_n = reduce(vcat, [setdiff(inds, inds_s) for inds = inds_n])
            dats[n, :] = [mean(follow[inds_s]) mean(follow[inds_n])] # follow frequency for successful and unsuccessful replays
        end

        # mean and sem across sessions
        ms, ss = nanmean(dats, dims = 1)[:], nanstd(dats, dims = 1)[:] ./ sqrt.(sum(.~isnan.(dats), dims = 1))[:]

        # print some results and stats
        rands, trueval = permutation_test(dats[:,1], dats[:,2]) # are successful and unsuccessful different
        println("follow $(["home"; "away"][ictrl+1]): p = ", mean(rands .> trueval), ", L = $(size(dats))")

        # plot our results
        ax = fig.add_subplot(grids[1,ictrl+1])
        ax.bar([1;2], ms, yerr = ss, color = [col_h, col_c][ictrl+1], capsize = capsize)
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1;2], ["succ", "un"])
        ax.set_title(["home\ntrials"; "away\ntrials"][ictrl+1], fontsize = fsize, pad = -10)

        # plot individual data points
        for i_n = 1:2 # for successful and unsuccessful
            corrs = dats[:, i_n]
            shifts = 1:length(corrs); shifts = (shifts .- mean(shifts))/std(shifts)*0.15 # add some jitter
            ax.scatter(i_n .+ shifts, corrs, color = col_point, marker = ".", s = 6, alpha = 1)
        end
        ax.set_ylim(ylims)
        # add some labels to first axis
        if ictrl == 0 ax.set_ylabel(L"$p(a_1 = \hat{a}_1)$"); ax.set_yticks(ticks) else ax.set_yticks([]) end
    end
end

# plot same thing for the computational model
succs = [mean(mod_dict[seed]["follow_succs"]) for seed = seeds] # p(follow) for successful rollouts
nons = [mean(mod_dict[seed]["follow_non"]) for seed = seeds] # unsuccessful
μ, s = mean(succs), std(succs)/sqrt(length(succs)) # summary statistics unsuccessful
μr, sr = mean(nons), std(nons)/sqrt(length(nons)) # summary statistics successful

# plot
ax = fig.add_subplot(grids[1,3])
ax.bar([1;2], [μ;μr], yerr = [s;sr], color = col_p, capsize = capsize)
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1;2], ["succ", "un"])
ax.set_title("model", fontsize = fsize)
for i_n = 1:2 # plot individual data points for successful and unsuccessful replays
    corrs = (if i_n == 1 succs else nons end) # data for this category
    shifts = 1:length(corrs); shifts = (shifts .- mean(shifts))/std(shifts)*0.15 # add some jitter
    ax.scatter(i_n .+ shifts, corrs, color = col_point, marker = ".", s = 15, alpha = 1) # plot data
end
ylims = [0;1.02]
ax.set_ylim(ylims); ax.set_yticks([])

# plot success frequency by replay number
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.52, right=1.00, bottom = 0.0, top = top, wspace=0.25)
minL = 3 # minimum number of replays in trial

if plot_experimental_replays
    # we start by regressing success frequency against time to account for this effect

    # initialize some arrays
    allsuccs, allts, allnums, all_tolds, allsuccsc = [], [], [], [], []
    for (i_n, name) = enumerate(rnames) #for each sessions
        # load the data for this session
        succs, repts, nums, tolds = [exp_dict[name]["perf_by_rep"][k] for k = ["success", "repts", "trialnums", "reptolds"]]
        ntrial = length(nums)
        succsc = exp_dict[name]["perf_by_rep"]["success_ctrl"] # success to control locations
        push!(allts, reduce(vcat, [repts[i] for i = 1:ntrial])) # time until end of drinking
        push!(all_tolds, reduce(vcat, [tolds[i] for i = 1:ntrial])) # time since beginning of drinking
        push!(allsuccs, reduce(vcat, [succs[i] for i = 1:ntrial])) # whether trials were successful
        push!(allnums, reduce(vcat, [nums[i]*ones(length(succs[i])) for i = 1:ntrial])) # replay number within trial
        succsc_m = [mean(reduce(hcat, succsc[:, i]), dims = 2)[:] for i = 1:ntrial] # mean across ctrl locations
        push!(allsuccsc, reduce(vcat, [succsc_m[i][keep[i]] for i = 1:ntrial])) # store mean
    end

    tthresh = 15 # time limit for regressing replay success against
    # concatenate data from above
    allsuccs, allts, allnums, all_tolds, allsuccsc = [reduce(vcat, arr) for arr = [allsuccs, allts, allnums, all_tolds, allsuccsc]]
    X = [max.(allts, -tthresh) min.(all_tolds, tthresh) ones(length(allts))] # regressors for predicting replay success from time
    x1, y1 = X[allnums .% 2 .== 0, :], allsuccs[allnums .% 2 .== 0] # home trials
    x2, y2 = X[allnums .% 2 .== 1, :], allsuccs[allnums .% 2 .== 1] # away trials
    #linear regression for success vs time
    y1, y2 = y1 .- 1*mean(y1), y2 .- 1*mean(y2) # subtract mean since we care about the change in success
    a1 = (x1' * x1)^(-1) * x1' * y1 # home trials
    a2 = (x2' * x2)^(-1) * x2' * y2 # away trials
    #separate regression for control locations
    y1c, y2c = allsuccsc[allnums .% 2 .== 0], allsuccsc[allnums .% 2 .== 1]; y1c, y2c = y1c .- mean(y1c), y2c .- mean(y2)
    a1c = (x1' * x1)^(-1) * x1' * y1c # control home
    a2c = (x2' * x2)^(-1) * x2' * y2c # control away

    # now we perform our analyses by replay number
    exp_, ctrl, norm, norm_ctrl = [], [], [], [] # initialize some arrays
    for (i_n, name) = enumerate(rnames) #for each session
        # load some data
        trialnums, succs, repts, tolds = [exp_dict[name]["perf_by_rep"][k] for k = ["trialnums", "success", "repts", "reptolds"]]
        succsc = exp_dict[name]["perf_by_rep"]["success_ctrl"] # success for control location
        Ls = [length(succ) for succ = succs] # number of replays
        minL_inds = findall(Ls .>= minL) # trials with at least minL replays
        if length(minL_inds) > 0.5 # if we have at least one trial for this session
            success_L = reduce(hcat, [succs[i][1:minL] for i = minL_inds])' # ntrial x minL
            succsc_L = [reduce(hcat, [succsc[j, :][i][1:minL] for i = minL_inds])' for j = 1:7] # control locations
            succtrials_L = [trialnums[i] for i = minL_inds] # trial number
            succts_L = reduce(hcat, [repts[i][1:minL] for i = minL_inds])' # time until end of trial
            tolds_L = reduce(hcat, [tolds[i][1:minL] for i = minL_inds])' # time since beginning
            X_L = cat(max.(succts_L, -tthresh), min.(tolds_L, tthresh), dims = 3) # construct regressors
            n = size(X_L, 3) # number of trials
            #compute time correction factors
            reg1, reg2 = [sum(X_L  .* reshape(a, (1,1,length(a)))[:,:,1:n], dims = 3)[:, :, 1] for a = [a1, a2]] # home and away
            reg1, reg2 = [reg .- mean(reg, dims = 2) for reg = [reg1, reg2]] #don't change mean across replays
            #separate correction for control locations
            reg1c, reg2c = [sum(X_L  .* reshape(a, (1,1,length(a)))[:,:,1:n], dims = 3)[:, :, 1] for a = [a1c, a2c]]
            reg1c, reg2c = [reg .- mean(reg, dims = 2) for reg = [reg1c, reg2c]] #don't change mean across replays

            inds1, inds2 = findall((succtrials_L .% 2) .== 0), findall((succtrials_L .% 2) .== 1) # indices for home and away
            newexp = success_L[inds1, :] - reg1[inds1, :] # subtract effect of time for home trials
            newctrl = success_L[inds2, :] - reg2[inds2, :] # subtract effect of time for away trials
            for i = inds1 # for home trials and control locations
                for j = 1:7 push!(norm, succsc_L[j][i:i, :] - prefac*reg1c[i:i, :]) end # subtract effect of time
            end
            for i = inds2 # for away trials and control locations
                for j = 1:7 push!(norm_ctrl, succsc_L[j][i:i, :] - prefac*reg2c[i:i, :]) end # subtract effect of time
            end
            push!(exp_, newexp) #home trials
            push!(ctrl, newctrl) #away trials
        end
    end

    # concatenate data
    exp_, ctrl, norm, norm_ctrl = [reduce(vcat, arr) for arr = [exp_, ctrl, norm, norm_ctrl]]

    function calc_norm_mu(exp_,norm)
        # function for normalizing by control locations
        μ, s = mean(exp_, dims = 1)[:], std(exp_, dims = 1)[:] / sqrt(size(exp_, 1)) # true mean and sem
        μn, sn = mean(norm, dims = 1)[:], std(norm, dims = 1)[:] / sqrt(size(norm, 1)) # reverse mean and sem
        #relative errors
        s = (μ ./ μn) .* sqrt.( (s./μ).^2 + (sn./μn).^2 )
        μ= μ ./ μn # normalized mean
        return μ, s # return results
    end
    μ, s = calc_norm_mu(exp_, norm) # calculate normalized value ('overrepresentation')
    μr, sr = calc_norm_mu(ctrl, norm_ctrl) # calculate normalized value ('overrepresentation')
    println("mean over-representation: ", μ) # print result

    # run a permutation test
    Ndat = size(exp_, 1); Nshuff = 10000; shuffs = zeros(Nshuff, length(μ))
    for n = 1:Nshuff # for each permutation
        perm = reduce(vcat, [randperm(minL)' for i = 1:Ndat]) # shuffled indices for each trial
        perm2 = repeat(perm, inner = (7,1)) # repeat for each control location
        new_exp = reduce(vcat, [exp_[i:i, perm[i, :]] for i = 1:Ndat]) # shuffled true data
        new_norm = reduce(vcat, [norm[i:i, perm2[i, :]] for i = 1:7*Ndat]) # shuffled control data
        shuffs[n,:] = calc_norm_mu(new_exp, new_norm)[1] # shuffled normalized value
    end

    deltas = [μ[2]-μ[1] μ[3]-μ[2] μ[3]-μ[1]] # true changes in overrepresentation
    deltas_shuff = [shuffs[:,2]-shuffs[:,1] shuffs[:,3]-shuffs[:,2] shuffs[:,3]-shuffs[:,1]] # controls
    for i = 1:3 println(i, ": p = ", mean((deltas_shuff[:,i] .- deltas[i]) .>= 0.0)) end # p values

    # plot result
    xs = 1:minL
    ax = fig.add_subplot(grids[1,1])
    ax.errorbar(xs, μ, yerr = s, fmt = "k-", capsize = 6)
    ax.set_xlim(1-0.5, xs[end]+0.5)
    ax.set_xticks(1:minL)
    ax.set_ylabel("over-representation")
    ax.set_xlabel("replay #")
    ax.set_title("experiment", fontsize = fsize)
    minv, maxv = minimum(μ-s), maximum(μ+s)
    ax.set_ylim(minv-0.1*(maxv-minv), maxv+0.1*(maxv-minv))
end

# now plot results for computational model
# load model results
cat_succ = reduce(hcat, [mean(mod_dict[seed]["suc_by_rep_min$(minL)"], dims = 1)[:] for seed = seeds])' # true
cat_succ_ctrl = reduce(hcat, [mean(mod_dict[seed]["suc_by_rep_min$(minL)_ctrl"], dims = (1,2))[:] for seed = seeds])' # control locations
cat_succ = cat_succ ./ cat_succ_ctrl #normalize by ctrl locations
μ, s = mean(cat_succ, dims = 1)[:], std(cat_succ, dims = 1)[:] / sqrt(size(cat_succ, 1)) # mean and sem across seeds

# plot result
ax = fig.add_subplot(grids[1,2])
rang = maximum(cat_succ) - minimum(cat_succ)
ylims = [minimum(cat_succ)-0.1*rang; maximum(cat_succ)+0.1*rang]
plot_comparison(ax, cat_succ; xticklabs= 1:minL, ylab = nothing, ylims = ylims, xlab = "replay #", col = col_p, col2 = 1.2*col_p, plot_title = "model")#, ylims = [0.9;1.55])


# add labels and save
y1 = 1.07
y2 = 0.46
x1, x2, x3 = -0.07, 0.28, 0.65
fsize = fsize_label
plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
plt.text(x2-0.02,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
plt.text(x3,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
plt.text(x1,y2,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
plt.text(0.48,y2,"E";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)

fname = "fig_replays"
savefig("./figs/$fname.pdf", bbox_inches = "tight")
savefig("./figs/$fname.png", bbox_inches = "tight")
close()

