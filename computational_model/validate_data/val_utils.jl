import Pkg
#Pkg.activate("/scratches/sagarmatha/ktj21_2/research/meta-rl/to-plan-or-not-to-plan/")
Pkg.activate("/scratches/enigma/ktj21/hennequin/metaRL/to-plan-or-not-to-plan/")
Pkg.activate("../")
using PyPlot, PyCall
@pyimport matplotlib.gridspec as gspec

global fsize = 26
global fsize_leg = 21
global cm = 1 / 2.54
#global figdir = "/scratches/sagarmatha/ktj21_2/research/meta-rl/to-plan-or-not-to-plan/rldm_2022/poster_figs/figs/"
global figdir = "/scratches/enigma/ktj21/hennequin/metaRL/to-plan-or-not-to-plan/figs/maze/"
global datadir = "/scratches/sagarmatha/ktj21_2/research/meta-rl/to-plan-or-not-to-plan/rldm_2022/poster_figs/data/"
global figdir = "./figs/"

#rc("font", family="serif", serif="calibri")
rc("font", size = fsize)
rc("pdf", fonttype = 42)
#rc("text", usetex = true)
rc("lines", linewidth = 5)
rc("axes", linewidth = 2)

PyCall.PyDict(matplotlib."rcParams")["axes.spines.top"] = false
PyCall.PyDict(matplotlib."rcParams")["axes.spines.right"] = false

rc("font", family="sans-serif")
PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = "calibri"

# rc("font", family="sans-serif")
#PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = ["Helvetica"]

#rc("font", family="DejaVu Sans")
# rc("font", family = "serif", serif = "Times")


### set global color scheme ###

global col_h = [0;0;0]/255 # human data
global col_o = [50;100;255]/255 # original
global col_a = [50;100;200]/255 # auxiliary loss
global col_p = [50;100;150]/255 # planning
global col_e = [50;80;120]/255 # euclidean prior


global col_o = [46;94;76]/255 # original
global col_a = [92; 194; 168]/255 # auxiliary loss
global col_p = [76;127;210]/255 # planning
global col_e = [35;35;110]/255 # euclidean prior

global col_p1 = col_p * 0.88
global col_p2 = col_p .+ [0.45; 0.35; 0.175]

### select global models

global seeds = [90;91;92;93;96]
global plan_epoch = 1000


### helper function

function plot_comparison(data, fname; xticklabs = ["", ""], ylab = "", xlab = nothing, col = "k", col2 = nothing, ylims = nothing, plot_title = nothing)
    if col2 == nothing col2 = col end
    niters = size(data, 1)
    m = mean(data, dims = 1)[:]
    s = std(data, dims = 1)[:] / sqrt(niters)
    xs = 1:size(data, 2)

    figure(figsize = (size(data, 2)*1.5, 4.00))
    for n = 1:niters
        scatter(xs, data[n, :], color = col2, s = 70, alpha = 0.6)
    end
    for n = 1:niters
        plot(xs, data[n, :], ls = ":", color = col2, alpha = 0.6, lw = 3)
    end
    errorbar(xs, m, yerr = s, fmt = "-", color = col, capsize = 10)

    xlim(1-0.5, xs[end]+0.5)
    xticks(xs, xticklabs)
    xlabel(xlab)
    ylabel(ylab)
    ylim(ylims)
    title(plot_title)
    savefig(fname, bbox_inches = "tight")
    close()
end


