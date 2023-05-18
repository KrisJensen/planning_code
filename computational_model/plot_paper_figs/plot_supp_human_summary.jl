#This script plots Figure S1 of Jensen et al.

include("plot_utils.jl")
using ToPlanOrNotToPlan
Random.seed!(1) # set random seed (for jitter in panel D)

# We start by loading some of our human behavioural data
@load "$(datadir)/human_RT_and_rews_play.bson" data; data_play = data # non-guided episodes
@load "$(datadir)/human_RT_and_rews_follow.bson" data; data_follow = data # guided episodes
keep = findall([nanmean(RTs) for RTs = data_follow["all_RTs"]] .< 690) # non-outlier users
Nkeep = length(keep)

# mean response times and rewards for all users
mean_RTs = [[nanmean(RTs) for RTs = data["all_RTs"]] for data = [data_follow, data_play]]
mean_rews = [[sum(rews)/size(rews, 1) for rews = data["all_rews"]] for data = [data_follow, data_play]]

fig = figure(figsize = (15*cm, 3*cm))
grids = fig.add_gridspec(nrows=1, ncols=3, left=0.00, right=0.78, bottom = 0, top = 1.0, wspace=0.5)

# plot average reward against average response time for all users

for i = 1:2 # plot data for guided and non-guided episodes
    ax = fig.add_subplot(grids[1,i])
    ax.scatter(mean_RTs[i][keep], mean_rews[i][keep], color = "k", marker = ".", s = 60)
    ax.set_xlabel("mean RT (ms)")
    ax.set_ylabel("mean reward")
    ax.set_title(["guided"; "non-guided"][i], fontsize = fsize)
    ax.set_yticks(4:2:12)
end

# plot mean probability of optimal action against mean response time

# load the data
@load "$(datadir)/human_all_data_follow.bson" data;
all_states, all_ps, all_as, all_wall_loc, all_rews, all_RTs, _, _ = data # extract relevant data
all_opts = []
Larena = 4; ed = EnvironmentDimensions(4^2, 2, 5, 50, Larena) # Environment parameters
for i = keep # for each user
    opts = [] # list of optimality of actions
    for b = 1:size(all_as[i], 1) # for each episode
        dists = dist_to_rew(all_ps[i][:, b:b], all_wall_loc[i][:, :, b:b], Larena) # distance to goal from each state
        if sum(all_rews[i][b, :]) > 0.5 # if at least 1 trial was completed
            for t = findall(all_rews[i][b,:] .> 0.5)[1]+1:sum(all_as[i][b,:] .> 0.5) # for each action
                # extract optimal policy
                pi_opt = optimal_policy(Int.(all_states[i][:,b,t]), all_wall_loc[i][:,:,b], dists, ed)
                push!(opts, Float64(pi_opt[Int(all_as[i][b,t])] > 1e-2)) # was the action taken optimal?
            end
        end
    end
    push!(all_opts, mean(opts)) # store results for this user
end
RTs = [nanmean(RTs) for RTs = all_RTs[keep]] # corresponding response times
rcor = cor(RTs, all_opts) # correlations between response times and optimality
ctrls = zeros(10000); for i = 1:10000 ctrls[i] = cor(RTs, all_opts[randperm(Nkeep)]) end # permutation test

# plot result
ax = fig.add_subplot(grids[1,3])
ax.scatter(RTs, all_opts, color = "k", marker = ".", s = 60)
ax.set_xlabel("mean RT (ms)")
ax.set_ylabel(L"$p$"*"(optimal)")


# plot means of the prior distributions for each user

@load "$datadir/guided_lognormal_params_delta.bson" params # parameters of prior distributions

# parameters for initial action of a trial and later actions of a trial
initial_delays = params["initial"][:, 3]+exp.(params["initial"][:, 1]+params["initial"][:, 2].^2/2)
later_delays = params["later"][:, 3]+exp.(params["later"][:, 1]+params["later"][:, 2].^2/2)

grids = fig.add_gridspec(nrows=1, ncols=1, left=0.90, right=1.0, bottom = 0, top = 1.0, wspace=0.40)

ax = fig.add_subplot(grids[1,1])
mus = [mean(initial_delays[keep]); mean(later_delays[keep])] # mean across users
ax.bar(1:2, mus, color = col_c) # bar plot
# plot individual data points
ax.scatter(ones(Nkeep)+randn(Nkeep)*0.1, initial_delays[keep], marker = ".", s = 6, color = "k")
ax.scatter(ones(Nkeep)*2+randn(Nkeep)*0.1, later_delays[keep], marker = ".", s = 6, color = "k")
ax.set_xticks(1:2, ["initial"; "later"], rotation = 45, ha = "right")
ax.set_ylabel("time (ms)")

# print some results as well
println("correlation between thinking time and optimality: ", rcor, ", p = ", mean(ctrls .> rcor))

# add labels and save
y1 = 1.16
x1, x2, x3, x4 = -0.09, 0.21, 0.49, 0.80
plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label, )
plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x3,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x4,y1,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
savefig("./figs/supp_human_data.pdf", bbox_inches = "tight")
savefig("./figs/supp_human_data.png", bbox_inches = "tight")
close()
