include("plot_utils.jl")
using BSON: @load
using Random, NaNStatistics, Statistics
using ToPlanOrNotToPlan
cm = 1/2.54

@load "$(datadir)/human_RT_and_rews_play.bson" data; data_play = data
@load "$(datadir)/human_RT_and_rews_follow.bson" data; data_follow = data
keep = findall([nanmean(RTs) for RTs = data_follow["all_RTs"]] .< 690)
Nkeep = length(keep)

mean_RTs = [[nanmean(RTs) for RTs = data["all_RTs"]] for data = [data_follow, data_play]]
mean_rews = [[sum(rews)/size(rews, 1) for rews = data["all_rews"]] for data = [data_follow, data_play]]


bot, top = 0.0, 1.0
fig = figure(figsize = (15*cm, 3*cm))

grids = fig.add_gridspec(nrows=1, ncols=3, left=0.00, right=0.78, bottom = 0, top = 1.0, wspace=0.5)

### plot rew vs RT follow & play ###

for i = 1:2
    ax = fig.add_subplot(grids[1,i])
    ax.scatter(mean_RTs[i][keep], mean_rews[i][keep], color = "k", marker = ".", s = 60)
    ax.set_xlabel("mean RT (ms)")
    ax.set_ylabel("mean reward")
    ax.set_title(["guided"; "non-guided"][i], fontsize = fsize)
    ax.set_yticks(4:2:12)
end

### plot opt vs RT play ###

Larena = 4; ed = EnvironmentDimensions(4^2, 2, 5, 50, Larena)
@load "$(datadir)/human_all_data_follow.bson" data;
all_states, all_ps, all_as, all_wall_loc, all_rews, all_RTs, all_trial_nums, all_trial_time = data
all_opts = []
for i = keep
    opts = []
    for b = 1:size(all_as[i], 1)
        dists = dist_to_rew(all_ps[i][:, b:b], all_wall_loc[i][:, :, b:b], Larena)
        if sum(all_rews[i][b, :]) > 0.5
            for t = findall(all_rews[i][b,:] .> 0.5)[1]+1:sum(all_as[i][b,:] .> 0.5)
                pi_opt = optimal_policy(Int.(all_states[i][:,b,t]), all_wall_loc[i][:,:,b], dists, ed)
                push!(opts, Float64(pi_opt[Int(all_as[i][b,t])] > 1e-2))
            end
        end
    end
    push!(all_opts, mean(opts))
end
RTs = [nanmean(RTs) for RTs = all_RTs[keep]]
rcor = cor(RTs, all_opts)
ctrls = zeros(10000); for i = 1:10000 ctrls[i] = cor(RTs, all_opts[randperm(Nkeep)]) end

ax = fig.add_subplot(grids[1,3])
ax.scatter(RTs, all_opts, color = "k", marker = ".", s = 60)
ax.set_xlabel("mean RT (ms)")
ax.set_ylabel(L"$p$"*"(optimal)")


### plot process and action time distributions ###

grids = fig.add_gridspec(nrows=1, ncols=1, left=0.90, right=1.0, bottom = 0, top = 1.0, wspace=0.40)

@load "$datadir/guided_lognormal_params_delta.bson" params #mu, sigma, delta
#note labels are swapped
action_times = params["initial"][:, 3]+exp.(params["initial"][:, 1]+params["initial"][:, 2].^2/2)
process_times = params["later"][:, 3]+exp.(params["later"][:, 1]+params["later"][:, 2].^2/2)

ax = fig.add_subplot(grids[1,1])
mus = [mean(action_times[keep]); mean(process_times[keep])]
ss = [std(action_times[keep]); std(process_times[keep])]
ax.bar(1:2, mus, color = col_c)
ax.scatter(ones(Nkeep)+randn(Nkeep)*0.1, action_times[keep], marker = ".", s = 6, color = "k")
ax.scatter(ones(Nkeep)*2+randn(Nkeep)*0.1, process_times[keep], marker = ".", s = 6, color = "k")
ax.set_xticks(1:2, ["initial"; "later"], rotation = 45, ha = "right")
ax.set_ylabel("time (ms)")

### add labels and save ###

add_labels = true
if add_labels
    y1 = 1.16
    y2 = 0.46
    x1, x2, x3, x4 = -0.09, 0.21, 0.49, 0.80
    fsize = fsize_label
    plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
    plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x3,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x4,y1,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
end

savefig("./figs/supp_human_data.pdf", bbox_inches = "tight")
savefig("./figs/supp_human_data.png", bbox_inches = "tight")
close()

println("correlation between thinking time and optimality: ", rcor, ", p = ", mean(ctrls .> rcor))
println("correlation between play and follow: ", cor(mean_RTs[1][keep], mean_RTs[2][keep]))

#rcor2 = cor(RTs[all_opts .> 0.8], all_opts[all_opts .> 0.8])
#ctrls2 = zeros(10000); for i = 1:10000 ctrls2[i] = cor(RTs[all_opts .> 0.8], all_opts[all_opts .> 0.8][randperm(sum(all_opts .> 0.8))]) end
#println("correlation between thinking time and optimality: ", rcor2, ", p = ", mean(ctrls2 .> rcor2))
