include("plot_utils.jl")
using BSON: @load
using Random, NaNStatistics, Statistics, LaTeXStrings
using ToPlanOrNotToPlan

forG = true

cm = 1/2.54

bot, top = 0.00, 1.00

### plot G's schematic ###
if forG
    fig = figure(figsize = (17*cm, 3.0*cm))
else
    fig = figure(figsize = (15*cm, 3.0*cm))
    grids = fig.add_gridspec(nrows=1, ncols=1, left=-0.05, right=0.17, bottom = -0.3, top = 1.05, wspace=0.05)
    ax = fig.add_subplot(grids[1,1])
    plot_G_schematic(ax)
end

### plot projection of PG stuff ###
@load "$(datadir)planning_as_pg_new.bson" res_dict
cat3(a, b) = cat(a, b, dims = 3)
seeds = sort([k for k = keys(res_dict)])

grids = fig.add_gridspec(nrows=1, ncols=1, left=0.19, right=0.42, bottom = bot-0.24, top = 0.98, wspace=0.05)
ax = fig.add_subplot(grids[1,1], projection="3d")

seed = 62
alphas = res_dict[seed]["jacs"]
actions = Int.(res_dict[seed]["sim_as"])
using MultivariateStats
key = "sim_gs"
betas = res_dict[seed][key]
betas = reduce(vcat, [betas[i:i, :, actions[i]] for i = 1:length(actions)])
pca = MultivariateStats.fit(PCA, (betas .- mean(betas, dims = 1))'; maxoutdim=3)
Zb = predict(pca, (betas .- mean(betas, dims = 1))')
Za = predict(pca, (alphas .- mean(alphas, dims = 1))')
cols = [col_c, col_p, "g", "c"]
for a = 1:4
    #if a == 1 lab1, lab2 = L"{\bf \alpha}^{PG}_{\hat{a}_1=1}", L"{\bf \alpha}^{RNN}_{\hat{a}_1=1}"
    #elseif a == 2 lab1, lab2 = L"{\bf \alpha}^{PG}_{\hat{a}_1=2}", L"{\bf \alpha}^{RNN}_{\hat{a}_1=2}"
    #else lab1, lab2 = nothing, nothing end
    lab1, lab2 = nothing, nothing
    meanb = mean(Zb[:, actions .== a], dims = 2)[:]
    meanb = meanb / sqrt(sum(meanb.^2))
    ax.plot3D([0; meanb[1]], [0; meanb[2]], [0; meanb[3]], ls = "-", color = cols[a], lw = 2, label = lab1)
    ax.scatter3D([meanb[1]], [meanb[2]], [meanb[3]], color = cols[a], s = 50)
    meana = mean(Za[:, actions .== a], dims = 2)[:]
    meana = meana / sqrt(sum(meana.^2))
    ax.plot3D([0; meana[1]], [0; meana[2]], [0; meana[3]], ls = ":", color = cols[a], lw = 3, label = lab2)
end

ax.plot3D(zeros(2), zeros(2), zeros(2), ls = "-", color = "k", lw = 2, label = L"{\bf \alpha}^\mathrm{PG}_{1}")
ax.plot3D(zeros(2), zeros(2), zeros(2), ls = ":", color = "k", lw = 3, label = L"{\bf \alpha}^\mathrm{RNN}")
ax.set_xlabel("PC 1", labelpad = -16, rotation = 9);
ax.set_ylabel("PC 2", labelpad = -17, rotation = 107);
ax.set_zlabel("PC 3", labelpad = -17, rotation = 92);

ax.view_init(elev=35., azim=75.)
#ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_zlabel("PC 3")
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.legend(frameon = false, ncol = 2, bbox_to_anchor = (0.52, 1.05), loc = "upper center",columnspacing=1,
        fontsize = fsize_leg, borderpad = 0.0, labelspacing = 0.2, handlelength = 1.3,handletextpad=0.4, handleheight=1)

### plot angles ###

meanspca, meanspca2 = [[[[], []] for j = 1:3] for _ = 1:2]
for (ij, jkey) = enumerate(["jacs"; "jacs_shift"; "jacs_shift2"])
for seed = seeds
    #println(jkey, " ", seed)
    cosses, cosses2, sim_as, sim_a2s = [res_dict[seed][k] for k = ["cosses", "cosses2", "sim_as", "sim_a2s"]]
    sim_as_c, sim_a2s_c = sim_as .% 4 .+ 1, sim_a2s .% 4 .+ 1
    jacs, gs, gs2 = [copy(res_dict[seed][k]) for k = [jkey, "sim_gs", "sim_gs2"]]
    #if jkey != "jacs" jacs = reduce(vcat, jacs) end
    inds, inds2 = 1:length(sim_as), findall(.~isnan.(sim_a2s))
    jacs = jacs .- mean(jacs, dims = 1)
    betas=[reduce(vcat,[gs[i:i,:,Int(as[i])] for i=inds]) for as = [sim_as, sim_as_c]]
    betas2=[reduce(vcat,[gs2[i:i,:,Int(as[i])] for i=inds2]) for as = [sim_a2s, sim_a2s_c]]
    jacs, gs, gs2 = [arr ./ sqrt.(sum(arr.^2, dims = 2)) for arr = [jacs,gs,gs2]]
    ### also compute angles in PC space
    for i = 1:2
        pca, pca2 = [MultivariateStats.fit(PCA, (beta .- mean(beta, dims = 1))'; maxoutdim=3) for beta = [betas[i], betas2[i]]]
        Za,Zb = [predict(pca, (vals .- mean(vals, dims = 1))') for vals = [jacs, betas[i]]]
        Za2,Zb2 = [predict(pca2, (vals .- mean(vals, dims = 1))') for vals = [jacs[inds2,:], betas2[i]]]
        Za,Za2,Zb,Zb2 = [arr ./ sqrt.(sum(arr.^2, dims = 2)) for arr = [Za,Za2, Zb, Zb2]]
        push!(meanspca[ij][i], mean(sum(Za .* Zb, dims = 2)[:, 1, :], dims = 1)[1])
        push!(meanspca2[ij][i], mean(sum(Za2 .* Zb2, dims = 2)[:, 1, :], dims = 1)[1])
    end
end
end

grids = fig.add_gridspec(nrows=1, ncols=2, left=0.49, right=0.76, bottom = bot, top = 1.0, wspace=0.25)
meanspca, meanspca2 = [reduce((a,b)->cat(a,b,dims=3), [reduce(hcat, m) for m = ms] ) for ms = [meanspca, meanspca2]]
labs = repeat([L"$1^\mathrm{st}$ action", L"$2^\mathrm{nd}$ action"], outer = 2)
global limy = nothing
for (ires, res) = enumerate([meanspca, meanspca2])
    ax = fig.add_subplot(grids[1,ires])
    res = res[:, 1, [1; ires+1]]
    mus, ss = mean(res, dims = 1)[:], std(res, dims = 1)[:]/sqrt(length(seeds))
    println("action $ires: mu = ", mus, ", sem = ", ss)
    ax.bar([1;2], mus, yerr = ss, color = col_p, capsize = capsize)
    if plot_points
        shifts = 1:size(res, 1); shifts = (shifts .- mean(shifts))/std(shifts)*0.2
        ax.scatter([1 .+ shifts; 2 .+ shifts], [res[:, 1]; res[:, 2]], color = col_point, marker = ".", s = 15)
    end
    ax.axhline(0, color = col_c, lw = 2) 
    if ires in [1,3]
        vmin, vmax = minimum(mus-ss), maximum(mus+ss)
        global limy = [vmin - 0.05*(vmax-vmin); vmax+0.05*(vmax-vmin)]
        ax.set_ylabel(L"$\cos \theta$", labelpad = -0.07)
        #ax.set_xticks([1;2], [L"${\bf \alpha}^{RNN}$", L"${\bf \alpha}^{RNN}_{ctrl}$"])
    else
        global limy = limy
        ax.set_yticks([])
        #ax.set_xticks([1;2], [L"${\bf \alpha}^{RNN}$", L"${\bf \alpha}^{RNN}_{ctrl}$"])
    end
    ax.set_xticks([1;2], [L"${\bf \alpha}^\mathrm{RNN}$", L"${\bf \alpha}^\mathrm{RNN}_\mathrm{ctrl}$"])
    ax.set_ylim(limy)
    ax.set_title(labs[ires], fontsize = fsize, pad = 0.04)
    ax.set_xlim(0.5, 2.5)
end

### add planning by network size ###

grids = fig.add_gridspec(nrows=1, ncols=1, left=0.87, right=1.0, bottom = 0, top = top, wspace=0.15)
ax = fig.add_subplot(grids[1,1])
@load "$datadir/rew_and_plan_by_n_model.bson" res_dict
meanrews, pfracs, seeds, Nhiddens, epochs, biases = [res_dict[k] for k = ["meanrews", "planfracs", "seeds", "Nhiddens", "epochs", "biases"]]
hinds = [2;3;4]
epochs = epochs[epochs .<= plan_epoch]

mms = mean(meanrews, dims = 2)[hinds, 1, 2:length(epochs)]
sms = std(meanrews, dims = 2)[hinds, 1, 2:length(epochs)] / sqrt(length(seeds))
mps = mean(pfracs, dims = 2)[hinds, 1, 2:length(epochs)]
sps = std(pfracs, dims = 2)[hinds, 1, 2:length(epochs)] / sqrt(length(seeds))
Nhiddens = Nhiddens[hinds]
xs = epochs[2:end]
Nhid = length(Nhiddens)
cols = [[0, i/(Nhid+1), 1-i/(Nhid+1)] for i = 1:Nhid]

ax.axhline(0.2, ls = "-", color = col_c)
for (ihid, Nhidden) = enumerate(Nhiddens)
    frac = (Nhidden - minimum(Nhiddens))/(maximum(Nhiddens) - minimum(Nhiddens))
    frac = (0.45*frac .+ 0.76)
    col = col_p * frac
    if ihid % 1 == 0 label = Nhidden else label = nothing end
    ax.plot(mms[ihid, :], mps[ihid, :], ls = "-", color = col, label = label)
    ax.fill_between(mms[ihid, :], mps[ihid, :]-sps[ihid, :], mps[ihid, :]+sps[ihid, :], color = col, alpha = 0.2)
end
ax.set_xlabel("mean reward")
ax.set_ylabel(L"$p$"*"(rollout)")
ax.set_ylim(0, 0.65)
ax.set_xticks([0;4;8])
ax.legend(frameon = false, fontsize = fsize_leg, handlelength=1.5, handletextpad=0.5, borderpad = 0.0,
        labelspacing = 0.05, loc = "lower center", bbox_to_anchor = (0.75, -0.035))


### add labels and save ###

add_labels = true
if add_labels
    y1 = 1.18
    y2 = 0.46
    x1, x2, x3, x4 = -0.06, 0.18, 0.44, 0.78
    fsize = fsize_label
    if ~forG
    plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
    plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x3,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x4,y1,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    end
end

if forG
    savefig("./figs/fig_mechanism_neural_nolab.pdf", bbox_inches = "tight")
    savefig("./figs/fig_mechanism_neural_nolab.png", bbox_inches = "tight")
else
    savefig("./figs/fig_mechanism_neural.pdf", bbox_inches = "tight")
    savefig("./figs/fig_mechanism_neural.png", bbox_inches = "tight")
end

close()


