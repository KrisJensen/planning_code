import Pkg
#Pkg.activate("/scratches/sagarmatha/ktj21_2/research/meta-rl/to-plan-or-not-to-plan/")
Pkg.activate("/scratches/enigma/ktj21/hennequin/metaRL/to-plan-or-not-to-plan/")
Pkg.activate("../")
using Revise
using PyPlot, PyCall, Distributions, Random
@pyimport matplotlib.gridspec as gspec
@pyimport matplotlib.patches as patch
Random.seed!(1)

global fsize = 10
global fsize_leg = 8
global fsize_label = 12
global cm = 1 / 2.54
global datadir = "../analysis_scripts/results/"
global figdir = "./figs/"
global widdir = "/scratches/enigma/ktj21/hennequin/widloski/"
global lw_wall = 5
global lw_arena = 1.3
global linewidth = 3
global weiji = true #whether to use Weiji's thinking time estimates
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

py"""
import pickle
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""
load_pickle = py"load_pickle"

function load_exp_data(;summary = false)
    ### load experimental analyses ###

    if summary
        resdir = widdir*"results/summary_data/"
    else
        resdir = widdir*"results/decoding/"
    end

    fnames = readdir(resdir); fnames = fnames[[~occursin("succ", f) for f = fnames]]
    rnames = [f[10:length(f)-2] for f = fnames]
    res_dict = Dict()
    for (i_f, f) = enumerate(fnames)
        res = load_pickle("$resdir$f")
        res_dict[rnames[i_f]] = res
    end
    return rnames, res_dict
end


function plot_G_schematic(ax)

    p_bbox = patch.FancyBboxPatch((0, 0.1), 1, 0.9,
    boxstyle="Round,pad=0.00,rounding_size=0.1",
    ec="k", fc="grey", alpha = 0.3
    )
    ax.add_patch(p_bbox)

    p_bbox2 = patch.FancyBboxPatch((0, -1.0), 1, 0.9,
    boxstyle="Round,pad=0.00,rounding_size=0.1",
    ec="k", fc="grey", alpha = 0.3
    )
    ax.add_patch(p_bbox2)

    x1, x2, x3 = 0.15, 0.5, 0.85
    dx, dy = 0.5, 0.30
    y1, y2, y3 = 0.23, 0.39, 0.85
    size = 10
    hl, hw, lw = 0.08,0.06,1.5

    function draw_arrow(x,y,dx,dy)
    plt.arrow(x, y, dx, dy, color = "k", head_length = hl, head_width = hw, lw = lw, length_includes_head = true)
    end
    function text(x, y, s)
    plt.text(x, y, s, ha = "center", va = "center", size = size)
    end

    text(x1, y1, L"$h_1$")
    text(x3, y1, L"$h_2$")
    text(x2, y2-0.03, L"$\Delta h^\mathrm{PG}$")
    text(x2, y3, L"$\hat{\tau}, R_{\hat{\tau}}$")
    draw_arrow(x1+0.1, y1, dx, 0)
    draw_arrow(x2, y3-0.12, 0, -dy+0.06)

    text(x1, -y1, L"$h_1$")
    text(x3, -y1, L"$h_2$")
    text(x2, -y2-0.02, L"$\Delta h^\mathrm{RNN}$")
    text(x1, -y3, L"$\hat{\tau}, R_{\hat{\tau}}$")
    text(x3, -y3, L"$x_f$")
    draw_arrow(x1+0.1, -y1, dx, 0)
    draw_arrow(x1+0.17, -y3, dx-0.07, 0)
    draw_arrow(x1, -y1-0.15, 0, -dy)
    draw_arrow(x3, -y3+0.15, 0, dy)

    plt.text(-0.08, 0.57, "PG", rotation = 90, ha = "center", va = "center", size = size*1.2)
    plt.text(-0.08, -0.52, "RNN", rotation = 90, ha = "center", va = "center", size = size*1.2)

    ax.set_xlim(-0.1, 1.01)
    ax.set_ylim(-1.01, 1.01)
    ax.axis("off")
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
