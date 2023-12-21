include("plot_utils.jl") #various global settings

@load "$datadir/value_function_eval.bson" data

as = [data[seed]["as"] for seed = seeds]
Vs = [data[seeds[i]]["Vs"][as[i] .> 0.5] for i = 1:length(seeds)]
rtg = [data[seeds[i]]["rew_to_go"][as[i] .> 0.5] for i = 1:length(seeds)]
ts = [data[seeds[i]]["ts"][as[i] .> 0.5]/51*20 for i = 1:length(seeds)]
accs = [Vs[i] - rtg[i] for i = 1:length(seeds)]
all_accs = reduce(vcat, accs)
all_rtg = reduce(vcat, rtg)

all_last_as = []
for a = as
    last_inds = sum(a .> 0.5, dims = 2)[:]
    push!(all_last_as, reduce(hcat, [a[i, last_inds[i]-9:last_inds[i]] for i = 1:size(a, 1)])')
end
all_last_as = reduce(vcat, all_last_as)

### first just plot some general statistics
fig = figure(figsize = (15*cm, 7*cm))
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.1, right=0.9, bottom = 0.6, top = 1.0, wspace=0.4)
axs = [fig.add_subplot(grids[1,i]) for i = 1:2]

# plot histogram of rew-to-go and prediction errors
bins1 = -7:1:7
bins2 = -7:1:7
axs[1].hist(all_accs, alpha = 0.5, color = col_p, bins = bins1, label = "value function", zorder = 10000)
axs[1].hist(mean(all_rtg) .- all_rtg, color = col_c, alpha = 0.5, bins = bins2, label = "constant")
axs[1].set_xlabel("prediction error")
axs[1].set_ylabel("frequency")
axs[1].set_yticks([])
axs[1].legend(fontsize = fsize_leg, ncol = 2, loc = "upper center", bbox_to_anchor = (0.5, 1.2), frameon = false)

# plot prediction errors vs. time
bins = 0:1:20
xs = 0.5*(bins[1:length(bins)-1]+bins[2:end])
res = zeros(length(seeds), length(xs))
for i = 1:length(seeds)
    res[i, :] = [mean(abs.(accs[i][(ts[i] .> bins[j]) .& (ts[i] .<= bins[j+1])])) for j = 1:length(xs)]
end
m, s = mean(res, dims = 1)[:], std(res, dims = 1)[:]/sqrt(length(seeds))
axs[2].plot(xs, m, color = col_p)
axs[2].fill_between(xs, m-s, m+s, alpha = 0.2, color = col_p)
axs[2].set_xlabel("time within episode (s)")
axs[2].set_ylabel("prediction error")


###

plan_lengths = 1:5
plan_nums = 0:plan_lengths[end]
keys = ["tot_plans", "plan_nums", "suc_rolls", "num_suc_rolls", "Vs", "rew_to_go"]
all_accs, all_vals, all_vals0, all_accs0 = [zeros(length(seeds), length(plan_lengths), length(plan_nums)) for _ = 1:4]
for (iseed, seed) = enumerate(seeds)
    tot_plans, plan_nums, suc_rolls, num_suc_rolls, Vs, rew_to_go = [data[seed][key] for key = keys]
    accuracy = abs.(Vs - rew_to_go)
    for (ilength, plan_length) = enumerate(plan_lengths)
        for (inum, number) = enumerate(0:plan_length)
            inds = ((tot_plans .== plan_length) .& (plan_nums .== number) .& (suc_rolls .< 10.5))
            inds0 = (inds .& (num_suc_rolls .< 0.5)) # sequences with no successful rollouts
            accs = accuracy[inds]
            accs0 = accuracy[inds0]
            vals = Vs[inds]
            vals0 = Vs[inds0]
            rtg = rew_to_go[inds]
            #println(plan_length, " ", number, " ", length(accs), " ", mean(accs), " ", std(accs)/sqrt(length(accs)), " ", mean(vals))
            all_accs[iseed, ilength, inum] = mean(accs)
            all_accs0[iseed, ilength, inum] = mean(accs0)
            all_vals[iseed, ilength, inum] = mean(vals)
            all_vals0[iseed, ilength, inum] = mean(vals0)
        end
    end
end

##

cols = [[0.00; 0.09; 0.32], [0.00;0.19;0.52], [0.19;0.39;0.72], [0.34;0.54;0.87], [0.49;0.69;1.0]]
grids = fig.add_gridspec(nrows=1, ncols=4, left=0.0, right=1, bottom = 0.0, top = 0.35, wspace=0.6)
axs = [fig.add_subplot(grids[1,i]) for i = 1:4]
for (ilength, plan_length) = enumerate(plan_lengths)
    for (idat, dat) = enumerate([all_vals, all_vals0, all_accs, all_accs0])
        m = mean(dat[:, ilength, :], dims = 1)[1:plan_length+1]
        s = std(dat[:, ilength, :], dims = 1)[1:plan_length+1]/sqrt(size(dat, 1))
        xs = 0:plan_length
        axs[idat].plot(xs, m, label = (if idat == 1 "$plan_length rollouts" else nothing end), color = cols[ilength])
        axs[idat].fill_between(xs, m-s, m+s, alpha = 0.2, color = cols[ilength])
    end
end
#axs[1].legend()
ylabels = ["value", "value [failed]", "error", "error [failed]"]
for i = 1:4 axs[i].set_ylabel(ylabels[i]) end
for ax = axs ax.set_xlabel("rollout number") end

## labels and save

y1, y2 = 1.07, 0.42
x1, x2, x3, x4 = -0.07, 0.195, 0.46, 0.745
plt.text(x1+0.13,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label, )
plt.text(x3+0.035,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x1,y2,"C"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label, )
plt.text(x2,y2,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x3,y2,"E";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)
plt.text(x4,y2,"F";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize_label,)

savefig("./figs/supp_value_function.pdf", bbox_inches = "tight")
savefig("./figs/supp_value_function.png", bbox_inches = "tight")
close()

