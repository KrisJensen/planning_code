include("plot_utils.jl")
using ToPlanOrNotToPlan
using BSON: @load
using Random, NaNStatistics, Statistics, ImageFiltering, StatsBase
cm = 1/2.54
bot, top = 0.65, 0.36
fig = figure(figsize = (15*cm, 7.5*cm))

rnames, resdict = load_exp_data(summary = true)

function smooth_data(data; smooth = 3, binsize = 1, minval = 0, maxval = 50)
    data = data[.~isnan.(data)]
    data_hist = fit(Histogram, data, minval:binsize:maxval)
    weights = data_hist.weights;
    xs = 0.5*(data_hist.edges[1][2:end]+data_hist.edges[1][1:length(weights)])
    ker = ImageFiltering.Kernel.gaussian((smooth/binsize,))
    newdata = imfilter(weights, ker)
    return xs, newdata
end

### plot distribution of trial counts

Ntrials = [length(resdict[n]["durs"]["home"]) for n = rnames]
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.00, right=0.25, bottom = bot, top = 1.0, wspace=0.05)
xs, smooth = smooth_data(Ntrials, minval = minimum(Ntrials)-4, maxval = maximum(Ntrials)+4, binsize = 1, smooth = 3)
ax = fig.add_subplot(grids[1,1])
ax.plot(xs, 100*smooth/length(Ntrials), color = "k")
ax.scatter(Ntrials, zeros(length(Ntrials)), marker = ".", color = "k", s = 40)
ax.set_xlabel("number of home trials")
ax.set_ylabel("frequency (%)")

### plot time-to-reward vs trial number ###

Nmax = 20
dur1, dur2 = [zeros(Nmax, length(rnames)) .+ NaN for _ = 1:2]
for (i_n, n) = enumerate(rnames)
    home, away = [resdict[n]["durs"][type] for type = ["home", "away"]]
    dur1[1:min(Nmax, length(home)), i_n] = (home[1:min(Nmax, length(home))] .< 5)
    dur2[1:min(Nmax, length(away)), i_n] = (away[1:min(Nmax, length(away))] .< 5)
end
m1, s1 = nanmean(dur1, dims = 2)[:], nanstd(dur1, dims = 2)[:]/sqrt(length(rnames))
m2, s2 = nanmean(dur2, dims = 2)[:], nanstd(dur2, dims = 2)[:]/sqrt(length(rnames))
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.37, right=0.62, bottom = bot, top = 1.0, wspace=0.05)
ax = fig.add_subplot(grids[1,1])
c1, c2 = "k", col_c
ax.plot(1:Nmax, m1, color = c1, label = "home")
ax.fill_between(1:Nmax, m1-s1, m1+s1, alpha = 0.2, color = c1)
ax.plot(1:Nmax, m2, color = c2, label = "away")
ax.fill_between(1:Nmax, m2-s2, m2+s2, alpha = 0.2, color = c2)
ax.set_xlabel("trial number")
ax.set_ylabel(L"$p(\Delta t < 5$" * " s"*L"$)$")
ax.legend(frameon = false, ncol = 2, loc = "upper center", bbox_to_anchor = (0.5, 1.25), fontsize = fsize_leg)

### plot distribution of number of neurons recorded ###
Nneurons = [resdict[n]["N_neurons"] for n = rnames]
#Nneurons = PyAny.(Nneurons)
grids = fig.add_gridspec(nrows=1, ncols=1, left=0.75, right=1.0, bottom = bot, top = 1.0, wspace=0.05)
binsize = 10
xs, smooth = smooth_data(Nneurons, minval = minimum(Nneurons)-30, maxval = maximum(Nneurons)+30, binsize = binsize, smooth = 15)
ax = fig.add_subplot(grids[1,1])
ax.plot(xs, 100*smooth/length(Nneurons)/binsize, color = "k")
ax.scatter(Nneurons, zeros(length(Nneurons)), marker = ".", color = "k", s = 40)
ax.set_xlabel("# recorded neurons")
ax.set_ylabel("frequency (%)")

println("N neurons range: ", minimum(Nneurons), " ", maximum(Nneurons))

### plot even/odd or first/second half tuning curve correlation across all neurons ###

grids = fig.add_gridspec(nrows=1, ncols=2, left=0.0, right=0.5, bottom = 0, top = top, wspace=0.30)
for (ikey, key) = enumerate(["corr_alternate", "cor_half"])
    ax = fig.add_subplot(grids[1,ikey])
    cors = reduce(vcat, [resdict[n][key] for n = rnames])
    ax.hist(cors, bins = [0.9:0.005:1, -1:0.1:1][ikey], color = "k")
    ax.set_xlabel("tuning curve cor.")
    if ikey == 1
        ax.set_ylabel("count (x1000)")
        ax.set_yticks([0;1000;2000], [0;1;2])
    else
        ax.set_yticks([])
    end
    ax.set_ylim(0, 2000)
    ax.set_title(["alternate bins"; "first/second half"][ikey], fontsize = fsize)
end

### plot distribution of replay lengths ###
rnames, rep_dict = load_exp_data();
home_len, away_len = [], []
for n = rnames
    tnums, lens = [rep_dict[n]["replay_lengths"][key] for key = ["trialnums", "lengths"]]
    push!(home_len, lens[tnums .% 2 .== 0])
    push!(away_len, lens[tnums .% 2 .== 1])
end
home_len, away_len = [reduce(vcat, arr) for arr = [home_len, away_len]]
grids = fig.add_gridspec(nrows=1, ncols=2, left=0.65, right=1.0, bottom = 0, top = top, wspace=0.30)
for (i, arr) = enumerate([home_len, away_len])
    ax = fig.add_subplot(grids[1,i])
    lens = 3:10
    counts = [log10(sum(arr .== l)) for l = lens]
    ax.bar(lens, counts, color = "k")
    ax.set_xlabel("replay length")
    if i == 1
        ax.set_ylabel("count")
        ax.set_yticks([0;1;2;3], [1;10;100;1000])
    else
        ax.set_yticks([])
    end
    ax.set_ylim(0, 3.6)
    ax.set_title(["home"; "away"][i], fontsize = fsize)
    ax.set_xticks([3;5;7;9])
end

### add labels and save ###

add_labels = true
if add_labels
    y1 = 1.07
    y2 = 0.46
    x1, x2, x3 = -0.07, 0.3, 0.63
    fsize = fsize_label
    plt.text(x1,y1,"A"; ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize, )
    plt.text(x2,y1,"B";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x3,y1,"C";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x1,y2,"D";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    plt.text(x3-0.06,y2,"E";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
    #plt.text(x3-0.,y2,"F";ha="left",va="top",transform=fig.transFigure,fontweight="bold",fontsize=fsize,)
end

savefig("./figs/supp_replay_data.pdf", bbox_inches = "tight")
savefig("./figs/supp_replay_data.png", bbox_inches = "tight")
close()
