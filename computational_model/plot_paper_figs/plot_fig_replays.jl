#This script plots Figure 4 of Jensen et al.

include("plot_utils.jl")
using BSON: @load
using Random, NaNStatistics, Statistics, PyCall
cm = 1/2.54

rnames, exp_dict = load_exp_data()

resdir = "../validate_data/data/"
@load resdir * "result_dict.bson" res_dict;
mod_dict = res_dict
seeds = sort([k for k = keys(mod_dict)])
Nseed = length(seeds)

bot, top = 0.62, 0.38
fig = figure(figsize = (15*cm, 7.5*cm))

### plot example replays ###
example_rep = load_pickle(widdir*"figs/replay/widloski_Billy3_20181207_s1_examples.p")
reps = example_rep["reps"]; walls = example_rep["walls"]; home = example_rep["home"]

grids = fig.add_gridspec(nrows=1, ncols=1, left=-0.04, right=0.22, bottom = bot-0.08, top = 1.0, wspace=0.15)
ax = fig.add_subplot(grids[1,1])
#plot arena
for x = (0:5).-0.5
    ax.plot([x;x], [-0.5;4.5], color = "k", lw = lw_arena*0.8)
    ax.plot([-0.5;4.5], [x;x], color = "k", lw = lw_arena*0.8)
end
for x = 1:3
    for y = 1:3
        ax.scatter([x], [y], s = 200, marker = ".", color = "k")
    end
end
ax.axis("off")
for w = walls
    plot(w[:,1], w[:,2], color = "k", lw = lw_wall*4/5)
end
cols = [col_p1, col_p1, col_p2, col_p2]
labels = ["successful"; nothing; "unsuccessful"; nothing]
lw_rep = 2
for (irep, rep) = enumerate(reps)
    rep = rep .+ 0.7*(irep/(length(reps)+1) .- 0.5)
    plot(rep[:,1], rep[:,2], color = cols[irep], label = labels[irep], lw = lw_rep, zorder = 800)
    ax.scatter([rep[1,1]], [rep[1,2]], color = cols[irep], s = 150, zorder = 1000, marker = ".")
end
ax.plot([home[1]], [home[2]], color = "k", marker="x", markersize=12*0.8, ls="", mew=3)
ax.legend(frameon = false, fontsize = fsize_leg, loc = "upper center", bbox_to_anchor = (0.5, 1.20), handlelength=1.5, handletextpad=0.5, borderpad = 0.0, labelspacing = 0.05)


### plot wall avoidance ###

grids = fig.add_gridspec(nrows=1, ncols=2, left=0.35, right=0.60, bottom = bot, top = 1.0, wspace=0.15)

ylims = [0;0.45]
batch_wall_probs = [exp_dict[n]["wall_crossings"][1] for n = rnames]
batch_rand_wall_probs = [mean(exp_dict[n]["wall_crossings"][2:end]) for n = rnames]
inds = findall((batch_wall_probs .> 0) .|| (batch_rand_wall_probs .> 0))
batch_wall_probs, batch_rand_wall_probs = batch_wall_probs[inds], batch_rand_wall_probs[inds]
μ, s = mean(batch_wall_probs), std(batch_wall_probs)/sqrt(length(batch_wall_probs))
μr, sr = mean(batch_rand_wall_probs), std(batch_rand_wall_probs)/sqrt(length(batch_rand_wall_probs))

ax = fig.add_subplot(grids[1,1])
plot_comparison(ax, [batch_wall_probs batch_rand_wall_probs]; xticklabs=["true", "ctrl"], ylab = L"$p$"*"(cross wall)", col2 = col_c, ylims = ylims, plot_title = "experiment")

#permutation test
function permute_test(arr1, arr2)
    #test whetter arr1 is larger than arr2
    rands = zeros(npermute)
    for n = 1:npermute
        inds = Bool.(rand(0:1, length(arr1)))
        b1, b2 = [arr1[inds]; arr2[.~inds]], [arr1[.~inds]; arr2[inds]]
        rands[n] = nanmean(b1-b2)
    end
    trueval = nanmean(arr1-arr2)
    return rands, trueval
end

rands, trueval = permute_test(batch_rand_wall_probs, batch_wall_probs)
println("walls: d = ", round(trueval, digits = 2) ,", p = ", mean(rands .> trueval))

#model
batch_wall_probs = [mean(mod_dict[seed]["batch_wall_probs"]) for seed = seeds]
batch_rand_wall_probs = [mean(mod_dict[seed]["batch_rand_wall_probs"]) for seed = seeds]

μ, s = mean(batch_wall_probs), std(batch_wall_probs)/sqrt(Nseed)
μr, sr = mean(batch_rand_wall_probs), std(batch_rand_wall_probs)/sqrt(Nseed)
ax = fig.add_subplot(grids[1,2])
plot_comparison(ax, [batch_wall_probs batch_rand_wall_probs]; xticklabs=["true", "ctrl"], ylab = nothing, col = col_p, col2 = 1.2*col_p, ylims = ylims, plot_title = "model", yticks = [])


### plot goal frequency ###

grids = fig.add_gridspec(nrows=1, ncols=2, left=0.75, right=1.00, bottom = bot, top = 1.0, wspace=0.15)
ylims = [0;0.8]
key = "goaldir_freq"
true_succs = [exp_dict[n][key][1] for n = rnames]
false_succs = [nanmean(exp_dict[n]["goaldir_freq"][2]) for n = rnames]
μ, s = mean(true_succs), std(true_succs)/sqrt(length(true_succs))
μr, sr = mean(false_succs), std(false_succs)/sqrt(length(false_succs))
ax = fig.add_subplot(grids[1,1])
plot_comparison(ax, [true_succs false_succs]; xticklabs=["true", "ctrl"], ylab = L"$p$"*"(goal)", col2 = col_c, ylims = ylims, plot_title = "experiment")

rands, trueval = permute_test(true_succs, false_succs)
println("goal: d = ", round(trueval, digits = 2) ,", p = ", mean(rands .> trueval))

# model
true_succs = [mean(mod_dict[seed]["true_succs"]) for seed = seeds]
false_succs = [mean(mod_dict[seed]["false_succs"]) for seed = seeds]
μ, s = mean(true_succs), std(true_succs)/sqrt(Nseed)
μr, sr = mean(false_succs), std(false_succs)/sqrt(Nseed)
ax = fig.add_subplot(grids[1,2])
plot_comparison(ax, [true_succs false_succs]; xticklabs=["true", "ctrl"], ylab = nothing, col = col_p, col2 = 1.2*col_p, ylims = ylims, plot_title = "model", yticks = [])

###newline ; plot p(follow) ###

grids = fig.add_gridspec(nrows=1, ncols=3, left=0.0, right=0.40, bottom = 0.0, top = top, wspace=0.15)
ylims = [0; 0.8]; ticks = 0:0.2:0.8
all_next_as, all_sim_as, all_succs, all_trialnums = [], [], [], []
all_succsc = [[] for _ = 1:7]
all_ns = []
for (i_n, name) = enumerate(rnames) #for each session
    trialnums, succs, next_as, sim_as = [exp_dict[name]["perf_by_rep"][k] for k = ["trialnums", "success", "a0s", "rep_a0s"]]
    succsc = exp_dict[name]["perf_by_rep"]["success_ctrl"]
    ntrial = length(trialnums) #number of trials in this session
    push!(all_succs, reduce(vcat, succs))
    push!(all_sim_as, reduce(vcat, sim_as))
    push!(all_trialnums, reduce(vcat, [trialnums[i]*ones(length(succs[i])) for i = 1:ntrial]))
    push!(all_next_as, reduce(vcat, [next_as[i]*ones(length(succs[i])) for i = 1:ntrial]))
    push!(all_ns, ones(length(all_succs[end]))*i_n)
    for i = 1:7
        push!(all_succsc[i], reduce(vcat, succsc[i, :]))
    end
end
all_next_as, all_sim_as, all_succs, all_trialnums, all_ns = [reduce(vcat, arr) for arr = [all_next_as, all_sim_as, all_succs, all_trialnums, all_ns]]
all_succsc = [reduce(vcat, succsc) for succsc in all_succsc]
follow = Float64.(all_next_as .== all_sim_as)

for ictrl = [0, 1]
    dats = zeros(length(rnames), 2)
    for n = 1:length(rnames)
        inds_s = findall((all_succs .== 1) .& ((all_trialnums .% 2) .== ictrl) .& (all_ns .== n))
        inds_n = [findall((succsc .== 1) .& ((all_trialnums .% 2) .== ictrl) .& (all_ns .== n)) for succsc = all_succsc]
        inds_n = reduce(vcat, [setdiff(inds, inds_s) for inds = inds_n])
        #inds_n = reduce(vcat, [inds for inds = inds_n])
        dats[n, :] = [mean(follow[inds_s]) mean(follow[inds_n])]
    end
    ms, ss = nanmean(dats, dims = 1)[:], nanstd(dats, dims = 1)[:] ./ sqrt.(sum(.~isnan.(dats), dims = 1))[:]

    rands, trueval = permute_test(dats[:,1], dats[:,2])
    println("follow $(["home"; "away"][ictrl+1]): p = ", mean(rands .> trueval), ", L = $(size(dats))")

    ax = fig.add_subplot(grids[1,ictrl+1])
    ax.bar([1;2], ms, yerr = ss, color = [col_h, col_c][ictrl+1], capsize = capsize)
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1;2], ["succ", "un"])

    ax.set_title(["home\ntrials"; "away\ntrials"][ictrl+1], fontsize = fsize, pad = -10)

    if plot_points
        ylims = [0;1.02]
        ticks = 0:0.5:1.0
        for i_n = 1:2
            corrs = dats[:, i_n]; col = col_point
            shifts = 1:length(corrs); shifts = (shifts .- mean(shifts))/std(shifts)*0.15
            ax.scatter(i_n .+ shifts, corrs, color = col, marker = ".", s = 6, alpha = 1)
            println("\nminimax corr: ", nanminimum(corrs), " ", nanmaximum(corrs), "\n")
        end
    end
    ax.set_ylim(ylims)
    if ictrl == 0 ax.set_ylabel(L"$p(a_1 = \hat{a}_1)$"); ax.set_yticks(ticks) else ax.set_yticks([]) end
end

#model
succs = [mean(mod_dict[seed]["follow_succs"]) for seed = seeds]
nons = [mean(mod_dict[seed]["follow_non"]) for seed = seeds]
μ, s = mean(succs), std(succs)/sqrt(length(succs))
μr, sr = mean(nons), std(nons)/sqrt(length(nons))

ax = fig.add_subplot(grids[1,3])
ax.bar([1;2], [μ;μr], yerr = [s;sr], color = col_p, capsize = capsize)
ax.set_xlim(0.5, 2.5)
ax.set_xticks([1;2], ["succ", "un"])
ax.set_title("model", fontsize = fsize)
if plot_points
    for i_n = 1:2
        corrs = (if i_n == 1 succs else nons end); col = col_point
        shifts = 1:length(corrs); shifts = (shifts .- mean(shifts))/std(shifts)*0.15
        ax.scatter(i_n .+ shifts, corrs, color = col, marker = ".", s = 15, alpha = 1)
    end
    ylims = [0;1.02]
end
ax.set_ylim(ylims); ax.set_yticks([])

### plot success by replay number ###
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.52, right=1.00, bottom = 0.0, top = top, wspace=0.25)

allsuccs, allts, allnums, all_tolds, allsuccsc = [], [], [], [], [] #tolds is relative to beginning of drinking, ts relative to end
for (i_n, name) = enumerate(rnames) #for each sessions
    succs, repts, nums, tolds = [exp_dict[name]["perf_by_rep"][k] for k = ["success", "repts", "trialnums", "reptolds"]]
    ntrial = length(nums)
    succsc = exp_dict[name]["perf_by_rep"]["success_ctrl"]
    succsc_u = [unique(reduce(vcat, [findall(Bool.(succsc[j, i])) for j = 1:7])) for i = 1:ntrial] 
    keep = [sort(union(succsc_u[i], findall(Bool.(succs[i])))) for i = 1:ntrial] #replays that went somewhere
    keep = [1:length(succs[i]) for i = 1:ntrial]
    push!(allts, reduce(vcat, [repts[i][keep[i]] for i = 1:ntrial]))
    push!(all_tolds, reduce(vcat, [tolds[i][keep[i]] for i = 1:ntrial]))
    push!(allsuccs, reduce(vcat, [succs[i][keep[i]] for i = 1:ntrial]))
    push!(allnums, reduce(vcat, [nums[i]*ones(length(keep[i])) for i = 1:ntrial]))
    succsc_m = [mean(reduce(hcat, succsc[:, i]), dims = 2)[:] for i = 1:ntrial] #mean across ctrl locations
    push!(allsuccsc, reduce(vcat, [succsc_m[i][keep[i]] for i = 1:ntrial]))
end


tthresh = 15
allsuccs, allts, allnums, all_tolds, allsuccsc = [reduce(vcat, arr) for arr = [allsuccs, allts, allnums, all_tolds, allsuccsc]]
X = [max.(allts, -tthresh) min.(all_tolds, tthresh) ones(length(allts))]
x1, y1 = X[allnums .% 2 .== 0, :], allsuccs[allnums .% 2 .== 0]
x2, y2 = X[allnums .% 2 .== 1, :], allsuccs[allnums .% 2 .== 1]
#linear regression for success vs time
y1, y2 = y1 .- 1*mean(y1), y2 .- 1*mean(y2)
a1 = (x1' * x1)^(-1) * x1' * y1
a2 = (x2' * x2)^(-1) * x2' * y2
#separate regression for control locations
y1c, y2c = allsuccsc[allnums .% 2 .== 0], allsuccsc[allnums .% 2 .== 1]; y1c, y2c = y1c .- mean(y1c), y2c .- mean(y2)
a1c = (x1' * x1)^(-1) * x1' * y1c
a2c = (x2' * x2)^(-1) * x2' * y2c

minL = 3
exp_, ctrl, norm, norm_ctrl = [], [], [], []
for (i_n, name) = enumerate(rnames) #for each session
    #println(i_n)
    trialnums, succs, repts, tolds = [exp_dict[name]["perf_by_rep"][k] for k = ["trialnums", "success", "repts", "reptolds"]]
    succsc = exp_dict[name]["perf_by_rep"]["success_ctrl"]
    Ls = [length(succ) for succ = succs]
    minL_inds = findall(Ls .>= minL)
    if length(minL_inds) > 0.5
        success_L = reduce(hcat, [succs[i][1:minL] for i = minL_inds])' # ntrial x minL
        succsc_L = [reduce(hcat, [succsc[j, :][i][1:minL] for i = minL_inds])' for j = 1:7] 
        succtrials_L = [trialnums[i] for i = minL_inds]
        succts_L = reduce(hcat, [repts[i][1:minL] for i = minL_inds])'
        tolds_L = reduce(hcat, [tolds[i][1:minL] for i = minL_inds])'
        #X_L = cat(succts_L, tolds_L, ones(size(succts_L, 1), minL), dims = 3)
        X_L = cat(max.(succts_L, -tthresh), min.(tolds_L, tthresh), dims = 3)
        n = size(X_L, 3)
        #compute zero-mean correction factors
        reg1, reg2 = [sum(X_L  .* reshape(a, (1,1,length(a)))[:,:,1:n], dims = 3)[:, :, 1] for a = [a1, a2]]
        reg1, reg2 = [reg .- mean(reg, dims = 2) for reg = [reg1, reg2]] #don't change mean across replays
        #separate correction for control locations
        reg1c, reg2c = [sum(X_L  .* reshape(a, (1,1,length(a)))[:,:,1:n], dims = 3)[:, :, 1] for a = [a1c, a2c]]
        reg1c, reg2c = [reg .- mean(reg, dims = 2) for reg = [reg1c, reg2c]] #don't change mean across replays

        inds1, inds2 = findall((succtrials_L .% 2) .== 0), findall((succtrials_L .% 2) .== 1)
        prefac = 1
        newexp = success_L[inds1, :] - prefac*reg1[inds1, :]
        newctrl = success_L[inds2, :] - prefac*reg2[inds2, :]
        for i = inds1
            for j = 1:7 push!(norm, succsc_L[j][i:i, :] - prefac*reg1c[i:i, :]) end
        end
        for i = inds2
            for j = 1:7 push!(norm_ctrl, succsc_L[j][i:i, :] - prefac*reg2c[i:i, :]) end
        end
        # for j = 1:7
        #     push!(norm, succsc_L[j][inds1, :] - reg1[inds1, :])
        #     push!(norm_ctrl, succsc_L[j][inds2, :] - reg2[inds2, :])
        # end
        push!(exp_, newexp) #home trials
        push!(ctrl, newctrl) #away trials
    end
end

exp_, ctrl, norm, norm_ctrl = [reduce(vcat, arr) for arr = [exp_, ctrl, norm, norm_ctrl]]

function calc_norm_mu(exp_,norm)
    μ, s = mean(exp_, dims = 1)[:], std(exp_, dims = 1)[:] / sqrt(size(exp_, 1))
    μn, sn = mean(norm, dims = 1)[:], std(norm, dims = 1)[:] / sqrt(size(norm, 1))
    #relative errors
    s = (μ ./ μn) .* sqrt.( (s./μ).^2 + (sn./μn).^2 )
    μ= μ ./ μn
    return μ, s
end
μ, s = calc_norm_mu(exp_, norm)
μr, sr = calc_norm_mu(ctrl, norm_ctrl)
println("mean over-representation: ", μ)

μraw, sraw = mean(exp_, dims = 1)[:], std(exp_, dims = 1)[:] / sqrt(size(exp_, 1))

Ndat = size(exp_, 1); Nshuff = 10000; shuffs = zeros(Nshuff, length(μ))
for n = 1:Nshuff
    perm = reduce(vcat, [randperm(minL)' for i = 1:Ndat])
    perm2 = repeat(perm, inner = (7,1))
    new_exp = reduce(vcat, [exp_[i:i, perm[i, :]] for i = 1:Ndat])
    new_norm = reduce(vcat, [norm[i:i, perm2[i, :]] for i = 1:7*Ndat])
    shuffs[n,:] = calc_norm_mu(new_exp, new_norm)[1]
end

if minL == 2
    deltas = μ[2]-μ[1]
    deltas_shuff = shuffs[:,2]-shuffs[:,1]
    println(mean((deltas_shuff .- deltas) .>= 0.0))
else
    deltas = [μ[2]-μ[1] μ[3]-μ[2] μ[3]-μ[1]]
    deltas_shuff = [shuffs[:,2]-shuffs[:,1] shuffs[:,3]-shuffs[:,2] shuffs[:,3]-shuffs[:,1]]
    for i = 1:3 println(i, ": p = ", mean((deltas_shuff[:,i] .- deltas[i]) .>= 0.0)) end
end

xs = 1:minL
ax = fig.add_subplot(grids[1,1])
ax.errorbar(xs, μ, yerr = s, fmt = "k-", capsize = 6)
ax.set_xlim(1-0.5, xs[end]+0.5)
ax.set_xticks(1:minL)
#ax.set_ylabel(L"p_{norm}"*"(goal)")
ax.set_ylabel("over-representation")
ax.set_xlabel("replay #")
ax.set_title("experiment", fontsize = fsize)
minv, maxv = minimum(μ-s), maximum(μ+s)
ax.set_ylim(minv-0.1*(maxv-minv), maxv+0.1*(maxv-minv))

#model
cvstr = "_cv"
cat_succ = reduce(hcat, [mean(mod_dict[seed]["suc_by_rep_min$minL$cvstr"], dims = 1)[:] for seed = seeds])'
cat_succ_ctrl = reduce(hcat, [mean(mod_dict[seed]["suc_by_rep_min$(minL)_ctrl$cvstr"], dims = (1,2))[:] for seed = seeds])'
cat_succ = cat_succ ./ cat_succ_ctrl #normalize by ctrl locations
μ, s = mean(cat_succ, dims = 1)[:], std(cat_succ, dims = 1)[:] / sqrt(size(cat_succ, 1))
ax = fig.add_subplot(grids[1,2])

rang = maximum(cat_succ) - minimum(cat_succ)
ylims = [minimum(cat_succ)-0.1*rang; maximum(cat_succ)+0.1*rang]
plot_comparison(ax, cat_succ; xticklabs= 1:minL, ylab = nothing, ylims = ylims, xlab = "replay #", col = col_p, col2 = 1.2*col_p, plot_title = "model")#, ylims = [0.9;1.55])


### add labels and save ###

add_labels = true
if add_labels
    y1 = 1.07
    y2 = 0.46
    x1, x2, x3 = -0.07, 0.28, 0.65
    fsize = fsize_label
    plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
    plt.text(x2-0.02,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x3,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x1,y2,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(0.48,y2,"E";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
end

fname = "fig_replays"
savefig("./figs/$fname.pdf", bbox_inches = "tight")
savefig("./figs/$fname.png", bbox_inches = "tight")
close()

