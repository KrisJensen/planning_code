import Pkg
Pkg.activate("../")
using Revise
using PyPlot, PyCall, LaTeXStrings
using Random, Statistics, NaNStatistics, Distributions
using BSON: @save, @load
@pyimport matplotlib.gridspec as gspec
@pyimport matplotlib.patches as patch
Random.seed!(1)

global fsize = 10
global fsize_leg = 8
global fsize_label = 12
global cm = 1 / 2.54
global datadir = "../analysis_scripts/results/"
global figdir = "./figs/"
global lw_wall = 5
global lw_arena = 1.3
global linewidth = 3
global npermute = 10000 #how many permutations for permutation tests
global plot_points = true # plot individual data points for n < 10 (required by NN)
global capsize = 3.5

# set some plotting parameters
rc("font", size = fsize)
rc("pdf", fonttype = 42)
rc("lines", linewidth = linewidth)
rc("axes", linewidth = 1)

PyCall.PyDict(matplotlib."rcParams")["axes.spines.top"] = false
PyCall.PyDict(matplotlib."rcParams")["axes.spines.right"] = false

rc("font", family="sans-serif")
PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = "arial"


### set global color scheme ###
global col_h = [0;0;0]/255 # human data
global col_p = [76;127;210]/255 # RL agent
global col_p1 = col_p * 0.88 #darker
global col_p2 = col_p .+ [0.45; 0.35; 0.175] #lighter
global col_c = [0.6,0.6,0.6] #ctrl
global col_point = 0.5*(col_c+col_h) #individual data points

### select global models

global seeds = 61:65
global plan_epoch = 1000

function get_human_inds()
    #get indices of human participants to analyse
    @load "$(datadir)/human_RT_and_rews_follow.bson" data #load data
    keep = findall([nanmean(RTs) for RTs = data["all_RTs"]] .< 690) #less than 690 ms RT on guided trials
    return keep
end


function plot_comparison(ax, data; xticklabs = ["", ""], ylab = "", xlab = nothing, col = "k", col2 = nothing, ylims = nothing, plot_title = nothing, yticks = nothing, rotation = 0)
    if col2 == nothing col2 = col end
    niters = size(data, 1)
    m = nanmean(data, dims = 1)[:]
    s = nanstd(data, dims = 1)[:] / sqrt(niters)
    xs = 1:size(data, 2)

    for n = 1:niters
        ax.scatter(xs, data[n, :], color = col2, s = 50, alpha = 0.6, marker = ".")
    end
    for n = 1:niters
        ax.plot(xs, data[n, :], ls = ":", color = col2, alpha = 0.6, linewidth = linewidth*2/3)
    end
    ax.errorbar(xs, m, yerr = s, fmt = "-", color = col, capsize = capsize)

    ax.set_xlim(1-0.5, xs[end]+0.5)
    if rotation == 0 ha = "center" else ha = "right" end
    ax.set_xticks(xs, xticklabs, rotation = rotation, ha = ha, rotation_mode = "anchor")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_ylim(ylims)
    #println(ylims, " ", yticks)
    if ~isnothing(yticks) ax.set_yticks(yticks) end
    ax.set_title(plot_title, fontsize = fsize)
end

# lognormal helper function

Phi(x) = cdf(Normal(), x) #standard normal pdf
function calc_post_mean(r; deltahat=0, muhat=0, sighat=0, mode = false)
    #compute posterior mean thinking time for a given response time 'r'
    if (r < deltahat+1) return 0.0 end #if response time lower than minimum delay, return 0

    if mode
        post_delay = deltahat+exp(muhat-sighat^2)
        post_delay = min(r, post_delay) #can at most be response time
        return r - post_delay
    end

    k1, k2 = 0, r - deltahat #integration limits
    term1 = exp(muhat+sighat^2/2)
    term2 = Phi((log(k2)-muhat-sighat^2)/sighat) - Phi((log(k1)-muhat-sighat^2)/sighat)
    term3 = Phi((log(k2)-muhat)/sighat) - Phi((log(k1)-muhat)/sighat)
    post_delay = (term1*term2/term3 + deltahat) #add back delta for posterior mean delay
    return r - post_delay #posterior mean thinking time is response minus mean delay
end

# Stats helper function


#permutation test
function permutation_test(arr1, arr2)
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

# Experimental data helper function

global replaydir = "../../replay_analyses/"
global plot_experimental_replays = false # False by default in case we haven't run these analyses
function load_exp_data(;summary = false)
    # This function loads experimental replay data
    # If 'summary' we load summary data for our supplementary figure
    # Else load the full experimental dataset

    if summary
        resdir = replaydir*"results/summary_data/"
    else
        resdir = replaydir*"results/decoding/"
    end

    # Filenames to load
    fnames = readdir(resdir); fnames = fnames[[~occursin("succ", f) for f = fnames]]
    rnames = [f[10:length(f)-2] for f = fnames] # Animal names+id for each file (i.e. sessions)
    res_dict = Dict() # Dictionary for storing results
    for (i_f, f) = enumerate(fnames) # For each file
        res = load_pickle("$resdir$f") # Load content of file
        res_dict[rnames[i_f]] = res # Store in our result dict
    end
    return rnames, res_dict # Return session names and results
end
