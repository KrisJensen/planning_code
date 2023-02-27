include("plot_utils.jl")
using BSON: @load
using Random, NaNStatistics, Statistics
cm = 1/2.54

epoch = plan_epoch

if weiji
    savename = "RT_predictions_N100_Lplan8_1000_weiji"
    @load "$datadir/$savename.bson" data
else
    @load "$(datadir)RT_predictions_new_$epoch.bson" data
end

res, allsims, RTs, pplans, dists, steps = [data[k] for k = ["residuals"; "correlations"; "RTs"; "pplans"; "dists"; "steps"]];
RTs_by_u, pplans_by_u, dists_by_u, steps_by_u = [data[k] for k = ["RTs_by_u", "pplans_by_u", "dists_by_u", "steps_by_u"]];

fig = figure(figsize = (10*cm, 3*cm))
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.00, right=1.00, bottom = 0.0, top = 1.0, wspace=0.35)

xsteps = 1:3
xdists = 1:6
cols = [[[0;0;0], [0.4;0.4;0.4], [0.7;0.7;0.7]], [[0.00;0.19;0.52], [0.24;0.44;0.77], [0.49;0.69;1.0]]]

for (idat, dat) = enumerate([RTs_by_u, pplans_by_u])
    fig.add_subplot(grids[1,idat])
    for xstep = xsteps
        mus = zeros(length(dat), length(xdists)) .+ NaN
        for u = 1:length(dat)
            for d = xdists
                inds = findall((dists_by_u[u] .== d) .& (steps_by_u[u] .== -xstep))
                if length(inds) >= 1
                    mus[u, d] = mean(dat[u][inds])
                end
            end
        end
        m = nanmean(mus, dims = 1)[:]
        s = nanstd(mus, dims = 1)[:] ./ sqrt.(sum(.~isnan.(mus), dims = 1)[:])

        plot(xdists, m, label = "step = $xstep", color = cols[idat][xstep])
        fill_between(xdists, m-s, m+s, color = cols[idat][xstep], alpha = 0.2)
    end
    xlabel("distance to goal")
    if idat == 1
        ylabel("thinking time")
    else
        ylabel(L"$\pi($"*"rollout"*L"$)$")
    end
    legend(frameon = false, fontsize = fsize_leg)
end

add_labels = true
if add_labels
    y1 = 1.16
    x1, x2 = -0.13, 0.45
    fsize = fsize_label
    plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
    plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
end

savefig("./figs/supp_plan_by_step.pdf", bbox_inches = "tight")
savefig("./figs/supp_plan_by_step.png", bbox_inches = "tight")
close()


### compute residual correlation ###

#compute residual
allcors = []
for u = 1:length(RTs_by_u)
    mean_sub_RTs, mean_sub_pplans = [], []
    for dist = 1:20 #for each distance-to-goal
        for xstep = 1:100 #for each step-within-trial
            inds = (dists_by_u[u] .== dist) .& (steps_by_u[u] .== -xstep)
            if sum(inds) >= 2 #require at least 2 data points
                new_RTs, new_pplans = RTs_by_u[u][inds], pplans_by_u[u][inds] #find the corresponding RTs and pi(rollout)
                mean_sub_RTs = [mean_sub_RTs; new_RTs .- mean(new_RTs)] #subtract mean of RTs and append
                mean_sub_pplans = [mean_sub_pplans; new_pplans .- mean(new_pplans)] #subtract mean of pi(rollout) and append
            end
        end
    end
    push!(allcors, cor(mean_sub_RTs, mean_sub_pplans))
    #println(allcors[end])
end
println(mean(allcors), " ", std(allcors)/sqrt(length(allcors)))

