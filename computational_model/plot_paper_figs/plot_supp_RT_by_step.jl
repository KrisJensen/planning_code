#This script plots Figure S2 of Jensen et al.

include("plot_utils.jl")

# start by loading the data for our human participants
@load "$datadir/RT_predictions_N100_Lplan8_1000.bson" data
RTs_by_u, pplans_by_u, dists_by_u, steps_by_u = [data[k] for k = ["RTs_by_u", "pplans_by_u", "dists_by_u", "steps_by_u"]];

fig = figure(figsize = (10*cm, 3*cm))
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.00, right=1.00, bottom = 0.0, top = 1.0, wspace=0.35)

xsteps = 1:3 # steps within trial to consider
xdists = 1:6 # distance to goal to consider
# colors for plots
cols = [[[0;0;0], [0.4;0.4;0.4], [0.7;0.7;0.7]], [[0.00;0.19;0.52], [0.24;0.44;0.77], [0.49;0.69;1.0]]]

for (idat, dat) = enumerate([RTs_by_u, pplans_by_u]) # for humans and model
    fig.add_subplot(grids[1,idat])
    for xstep = xsteps # for each step within trial
        mus = zeros(length(dat), length(xdists)) .+ NaN # initialize data array
        for u = 1:length(dat) # for each model/participant
            for d = xdists # for each distance to goal
                # relevant actions
                inds = findall((dists_by_u[u] .== d) .& (steps_by_u[u] .== -xstep))
                if length(inds) >= 1 # if we have at least one action satisfying these criteria
                    mus[u, d] = mean(dat[u][inds]) # store the data
                end
            end
        end
        m = nanmean(mus, dims = 1)[:] # mean across models/participants
        s = nanstd(mus, dims = 1)[:] ./ sqrt.(sum(.~isnan.(mus), dims = 1)[:]) # standard error

        plot(xdists, m, label = "step = $xstep", color = cols[idat][xstep]) # plot mean
        fill_between(xdists, m-s, m+s, color = cols[idat][xstep], alpha = 0.2) # plot standard error
    end
    xlabel("distance to goal")
    if idat == 1
        ylabel("thinking time") # human
    else
        ylabel(L"$\pi($"*"rollout"*L"$)$") # model
    end
    legend(frameon = false, fontsize = fsize_leg)
end

# add labels and save figure
y1 = 1.16
x1, x2 = -0.13, 0.45
fsize = fsize_label
plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)

savefig("./figs/supp_plan_by_step.pdf", bbox_inches = "tight")
savefig("./figs/supp_plan_by_step.png", bbox_inches = "tight")
close()


# compute and print residual correlations between pi(rollout) and human RT
allcors = [] # array to store correlations
for u = 1:length(RTs_by_u) # for each user
    mean_sub_RTs, mean_sub_pplans = [], [] # mean subtracted thinking times and rollouts probs
    for dist = 1:20 #for each distance-to-goal
        for xstep = 1:100 #for each step-within-trial
            inds = (dists_by_u[u] .== dist) .& (steps_by_u[u] .== -xstep) # corresponding indices
            if sum(inds) >= 2 #require at least 2 data points
                new_RTs, new_pplans = RTs_by_u[u][inds], pplans_by_u[u][inds] #find the corresponding RTs and pi(rollout)
                mean_sub_RTs = [mean_sub_RTs; new_RTs .- mean(new_RTs)] #subtract mean of RTs and append
                mean_sub_pplans = [mean_sub_pplans; new_pplans .- mean(new_pplans)] #subtract mean of pi(rollout) and append
            end
        end
    end
    push!(allcors, cor(mean_sub_RTs, mean_sub_pplans)) # store residual correlation for this participant
end
# print result
println("mean and standard error of residual correlation:")
println(mean(allcors), " ", std(allcors)/sqrt(length(allcors)))

