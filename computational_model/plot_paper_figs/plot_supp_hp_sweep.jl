include("plot_utils.jl")
using BSON: @load
using Random, NaNStatistics, Statistics
cm = 1/2.54

fig = figure(figsize = (17*cm, 7.5*cm))
bot, top = 0.72, 0.38
params = [(60,4),(60,8),(60,12),(100,4),(100,8),(100,12),(140,4),(140,8),(140,12)]
prefix = "hp_sweep_"
Nseed = length(seeds)

### plot human RT corr ###
ylims = [0;0.2]; ticks = [0;0.2]
grids = fig.add_gridspec(nrows=1, ncols=length(params), left=0.04, right=0.47, bottom = bot, top = 1.0, wspace=0.35)
for (ip, p) = enumerate(params)
    N, Lplan = p
    savename = "hp_sweep_N$(N)_Lplan$(Lplan)_1000"
    if weiji savename = savename * "_weiji" end
    @load "$datadir/RT_predictions_$savename.bson" data;
    allsims = data["correlations"]
    m1, s1 = mean(allsims[:, 1]), std(allsims[:, 1])/sqrt(size(allsims, 1))
    ax = fig.add_subplot(grids[1,ip])
    ax.bar([1], [m1], yerr = [s1], capsize = capsize,color = col_p)
    
    if weiji ax.set_ylim(0,0.22) end
    ax.set_xticks([1])
    ax.set_xticklabels([string(p)], rotation = 45, ha = "right")
    println(p, ": ", m1, " ", s1)

    if plot_points
        ylims = [-0.05;0.35]
        ticks = 0:0.1:0.3
        corrs = allsims[:, 1]; col = col_point
        shifts = 1:length(corrs); shifts = (shifts .- mean(shifts))/std(shifts)*0.15
        ax.scatter(1 .+ shifts, corrs, color = col, marker = ".", s = 3, alpha = 0.5)
        println(minimum(corrs), " ", maximum(corrs))
    end
    ax.set_ylim(ylims)
    if ip == 1 ylabel("correlation with\nthinking time"); ax.set_yticks(ticks) else yticks([]) end
end

### plot delta perf with rollouts ###

grids = fig.add_gridspec(nrows=1, ncols=length(params), left=0.57, right=1.0, bottom = bot, top = 1.0, wspace=0.35)

for (ip, p) = enumerate(params)
    N, Lplan = p; savename = "$(prefix)N$(N)_Lplan$(Lplan)"
    @load "$datadir/perf_by_n_$(savename).bson" res_dict
    seeds = sort([k for k = keys(res_dict)])
    Nseed = length(seeds)
    ms = []
    t0, t1 = 1, 6
    for (is, seed) = enumerate(seeds)
        dts, mindists, policies = [res_dict[seed][k] for k = ["dts"; "mindists"; "policies"]]
        keepinds = findall((.~isnan.(sum(dts[1, :, t0:t1], dims = (2))[:])) .& (mindists[:, 2] .>= 0))
        new_dts = dts[:, keepinds, :]
        m = mean(new_dts[1,:,:], dims = 1)[:]
        push!(ms, m[t0]-m[t1])
    end
    ms = reduce(hcat, ms)
    m1, s1 = mean(ms), std(ms)/sqrt(Nseed)

    ax = fig.add_subplot(grids[1,ip])
    ax.bar([1], [m1], yerr = [s1], capsize = capsize,color = col_p)
    if plot_points
        shifts = 1:length(ms); shifts = (shifts .- mean(shifts))/std(shifts)*0.2
        ax.scatter(1 .+ shifts, ms, color = col_point, marker = ".", s = 15)
    end
    if ip == 1 ylabel(L"$\Delta$"*"steps") else yticks([]) end
    ax.set_ylim(0, 1.5)
    ax.set_xticks([1])
    ax.set_xticklabels([string(p)], rotation = 45, ha = "right")
end

### plot delta pi(a1) ###


grids = fig.add_gridspec(nrows=1, ncols=length(params), left=0.00, right=1.0, bottom = 0, top = top, wspace=0.35)
for (ip, p) = enumerate(params)
    N, Lplan = p
    all_ms = []
    for i = 1:2 #rewarded and non-rewarded sim
        ms = []
        for seed = 51:55
            if p == (140,12)
                @load "$datadir/causal_N$(N)_Lplan$(Lplan)_$(seed)_1000_single.bson" data
            else   
                @load "$datadir/causal_N$(N)_Lplan$(Lplan)_$(seed)1000_single.bson" data
            end
            p_simulated_actions, p_simulated_actions_old = data["p_simulated_actions"], data["p_simulated_actions_old"]
            p_initial_sim, p_continue_sim = data["p_initial_sim"], data["p_continue_sim"]
            p_simulated_actions ./= (1 .- p_continue_sim)
            p_simulated_actions_old ./= (1 .- p_initial_sim)
            inds = findall(.~isnan.(sum(p_simulated_actions, dims = 1)[:])) #data for all
            #old and new probabilities
            push!(ms, [mean(p_simulated_actions_old[i, inds]); mean(p_simulated_actions[i, inds])])
        end
        ms = reduce(hcat, ms)
        push!(all_ms, ms[2, :]-ms[1,:])
    end

    ms = [mean(ms) for ms = all_ms]
    ss = [std(ms)/sqrt(Nseed) for ms = all_ms]
    ax = fig.add_subplot(grids[1,ip])
    ax.bar(1:2, ms, yerr = ss, color = [col_p1, col_p2], capsize = capsize)
    if plot_points
        shifts = 1:length(all_ms[1]); shifts = (shifts .- mean(shifts))/std(shifts)*0.2
        ax.scatter([1 .+ shifts; 2 .+ shifts], [all_ms[1]; all_ms[2]], color = col_point, marker = ".", s = 15)
    end
    ax.set_xticks(1:2, ["succ"; "un"])
    if ip == 1
        ax.set_ylabel(L"$\Delta \pi(\hat{a}_1)$", labelpad = 0)
        ax.set_yticks([-0.4;0;0.4])
    else
        ax.set_yticks([])
    end
    ax.set_title(string(p), fontsize = fsize)
    ax.set_ylim(-0.4,0.4)
    ax.set_xlim([0.4; 2.6])
    ax.axhline(0.0, color = "k", lw = 1)
end



### add labels and save ###

add_labels = true
if add_labels
    y1, y2 = 1.07, 0.48
    x1, x2 = -0.09, 0.50
    fsize = fsize_label
    plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
    plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x1,y2,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
end

savefig("./figs/supp_hp_sweep.pdf", bbox_inches = "tight")
savefig("./figs/supp_hp_sweep.png", bbox_inches = "tight")
close()
