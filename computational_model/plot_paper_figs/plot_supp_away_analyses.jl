include("plot_utils.jl")
using BSON: @load
using Random, NaNStatistics, Statistics
cm = 1/2.54

fig = figure(figsize = (10*cm, 3*cm))

rnames, exp_dict = load_exp_data()


grids = fig.add_gridspec(nrows=1, ncols=1, left=0.0, right=0.3, bottom = 0, top = 1.0, wspace=0.15)
ylims = [0;0.8]
true_succs = [exp_dict[n]["goaldir_freqa"][1] for n = rnames]
false_succs = [nanmean(exp_dict[n]["goaldir_freqa"][2]) for n = rnames]
μ, s = mean(true_succs), std(true_succs)/sqrt(length(true_succs))
μr, sr = mean(false_succs), std(false_succs)/sqrt(length(false_succs))
ax = fig.add_subplot(grids[1,1])
plot_comparison(ax, [true_succs false_succs]; xticklabs=["true", "ctrl"], ylab = L"$p$"*"(goal)", col2 = col_c, ylims = ylims, plot_title = "away trials")


### plot success by replay number ###
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.50, right=1.00, bottom = 0.0, top = 1.0, wspace=0.25)

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
for (i_n, name) = enumerate(rnames) #for each sessions
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
        reg1c, reg2c = [sum(X_L  .* reshape(a, (1,1,length(a)))[:,:,1:n], dims = 3)[:, :, 1] for a = [a1c, a2c]]
        reg1c, reg2c = [reg .- mean(reg, dims = 2) for reg = [reg1c, reg2c]] #don't change mean across replays
        
        inds1, inds2 = findall((succtrials_L .% 2) .== 0), findall((succtrials_L .% 2) .== 1)
        newctrl = success_L[inds2, :] - reg2[inds2, :]
        for j = 1:7
            push!(norm_ctrl, succsc_L[j][inds2, :] - reg2c[inds2, :])
        end
        push!(ctrl, newctrl)
    end
end

ctrl, norm_ctrl = [reduce(vcat, arr) for arr = [ctrl, norm_ctrl]]
μr, sr = mean(ctrl, dims = 1)[:], std(ctrl, dims = 1)[:] / sqrt(size(ctrl, 1))
μnr, snr = mean(norm_ctrl, dims = 1)[:], std(norm_ctrl, dims = 1)[:] / sqrt(size(norm_ctrl, 1))
#relative errors
sr = (μr ./ μnr) .* sqrt.( (sr./μr).^2 + (snr./μnr).^2 )
μr = μr ./ μnr

xs = 1:minL
ax = fig.add_subplot(grids[1,1])
ax.errorbar(xs, μr, yerr = sr, fmt = "k-", capsize = 6)
ax.set_xlim(1-0.5, xs[end]+0.5)
ax.set_xticks(1:minL)
#ax.set_ylabel(L"p_{norm}"*"(goal)")
ax.set_ylabel("over-representation")
ax.set_xlabel("replay #")
ax.set_title("away trials", fontsize = fsize)
minv, maxv = minimum(μr-sr), maximum(μr+sr)
ax.set_ylim(minv-0.1*(maxv-minv), maxv+0.1*(maxv-minv))


add_labels = true
if add_labels
    y1 = 1.16
    x1, x2 = -0.13, 0.32
    fsize = fsize_label
    plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
    plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
end

fname = "supp_away_analyses"
savefig("./figs/$fname.pdf", bbox_inches = "tight")
savefig("./figs/$fname.png", bbox_inches = "tight")
close()
