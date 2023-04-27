import Pkg
Pkg.activate("../")
using Revise
using PyPlot, PyCall, LaTeXStrings
using Random, Statistics, Distributions, Random
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

#rc("font", family="serif", serif="calibri")
rc("font", size = fsize)
rc("pdf", fonttype = 42)
#rc("text", usetex = true)
rc("lines", linewidth = linewidth)
rc("axes", linewidth = 1)

PyCall.PyDict(matplotlib."rcParams")["axes.spines.top"] = false
PyCall.PyDict(matplotlib."rcParams")["axes.spines.right"] = false

rc("font", family="sans-serif")
PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = "arial"
#PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = "calibri"

### set global color scheme ###

global col_h = [0;0;0]/255 # human data

global col_o = [46;94;76]/255 # original
global col_a = [92; 194; 168]/255 # auxiliary loss
global col_p = [76;127;210]/255 # planning
global col_e = [35;35;110]/255 # euclidean prior

global col_p1 = col_p * 0.88
global col_p2 = col_p .+ [0.45; 0.35; 0.175]

global col_c = [0.6,0.6,0.6] #ctrl

global col_point = 0.5*(col_c+col_h)

### select global models

global seeds = 61:65
global plan_epoch = 1000


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

### lognormal helper functions ###

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
