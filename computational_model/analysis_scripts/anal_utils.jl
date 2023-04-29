#this script loads some useful libraries and sets various global defaults.

# load some libraries that we generally need
import Pkg
Pkg.activate("../")
using Revise
using PyPlot, PyCall
using Distributions, Statistics, Random, StatsBase
using Flux, Zygote
using BSON: @save, @load
@pyimport matplotlib.gridspec as gspec

#set some default paths
global datadir = "./results/" #directory to write results to
global loaddir = "../models/" #directory to load models from

# select default global models
global seeds = 61:65 #random seeds
global plan_epoch = 1000 #training epoch to use for evaluation (1000 is final epoch)
global greedy_actions = true #sample actions greedily at test time
global N = 100 #number of units
global Lplan = 8 #planning horizon
global Larena = 4 #arena size
global prefix = "" #model name prefix
global epoch = plan_epoch #redundant
global run_default_analyses = true #run analyses when loading analysis functions

### lognormal helper functions ###
function lognorm(x; mu = 0, sig = 0, delta = 0)
    #pdf for shifted lognormal distribution (shift = delta)
    if x <= delta return 0 end
    return 1 / ((x-delta) * sig * sqrt(2*pi)) * exp(- (log(x-delta) - mu)^2 / (2*sig^2))
end

Phi(x) = cdf(Normal(), x) #standard normal pdf
function calc_post_mean(r; deltahat=0, muhat=0, sighat=0)
    #compute posterior mean thinking time for a given response time 'r'
    #deltahat, muhat, and sighat are the parameters of the lognormal prior over delays
    if r < deltahat+1 return 0 end #if response is faster than delta, no thinking
    k1, k2 = 0, r - deltahat #integration limits
    term1 = exp(muhat+sighat^2/2)
    term2 = Phi((log(k2)-muhat-sighat^2)/sighat) - Phi((log(k1)-muhat-sighat^2)/sighat)
    term3 = Phi((log(k2)-muhat)/sighat) - Phi((log(k1)-muhat)/sighat)
    post_delay = (term1*term2/term3 + deltahat) #add back delta for posterior mean delay
    return r - post_delay #posterior mean thinking time is response minus mean delay
end
