import Pkg
Pkg.activate("../")
using Revise
using PyPlot, PyCall
using Distributions
@pyimport matplotlib.gridspec as gspec

global fsize = 26
global fsize_leg = 21
global cm = 1 / 2.54
global datadir = "./results/"
global figdir = "../figs/maze/"

rc("font", size = fsize)
rc("pdf", fonttype = 42)
rc("lines", linewidth = 5)
rc("axes", linewidth = 2)

PyCall.PyDict(matplotlib."rcParams")["axes.spines.top"] = false
PyCall.PyDict(matplotlib."rcParams")["axes.spines.right"] = false

rc("font", family="sans-serif")
PyCall.PyDict(matplotlib."rcParams")["font.sans-serif"] = "calibri"


### set global color scheme ###
global col_p = [76;127;210]/255 # planning
global col_p1 = col_p * 0.88
global col_p2 = col_p .+ [0.45; 0.35; 0.175]

### select global models
global seeds = 61:65
global plan_epoch = 350

### lognormal helper functions ###
function lognorm(x; mu = 0, sig = 0, delta = 0)
    #pdf for shifted lognormal distribution
    if x <= delta return 0 end
    return 1 / ((x-delta) * sig * sqrt(2*pi)) * exp(- (log(x-delta) - mu)^2 / (2*sig^2))
end

Phi(x) = cdf(Normal(), x) #standard normal pdf
function calc_post_mean(r; deltahat=0, muhat=0, sighat=0)
    #compute posterior mean thinking time for a given response time 'r'
    if r < deltahat+1 return 0 end
    k1, k2 = 0, r - deltahat #integration limits
    term1 = exp(muhat+sighat^2/2)
    term2 = Phi((log(k2)-muhat-sighat^2)/sighat) - Phi((log(k1)-muhat-sighat^2)/sighat)
    term3 = Phi((log(k2)-muhat)/sighat) - Phi((log(k1)-muhat)/sighat)
    post_delay = (term1*term2/term3 + deltahat) #add back delta for posterior mean delay
    return r - post_delay #posterior mean thinking time is response minus mean delay
end
