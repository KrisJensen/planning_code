#in this script, we compare path lengths in the 'Euclidean' and 'non-Euclidean' arenas

include("anal_utils.jl")
using ToPlanOrNotToPlan

N = 10000 # number of arenas to compare
cat3(a, b) = cat(a, b, dims = 3)
w_wraps = reduce(cat3, [maze(4, wrap = true) for _ in 1:N]) # generate toroidal mazes
w_nowraps = reduce(cat3, [maze(4, wrap = false) for _ in 1:N]) # generate Euclidean mazes

ps = onehot_from_loc(4, 1:16) # possible goal locations
dists_wraps = zeros(N, 16, 16) # all-to-all distances
dists_nowraps = zeros(N, 16, 16)
for i1 = 1:N # for each maze
    for i2 = 1:16 # for each goal location
        # compute distance from all start
        dists_wraps[i1, i2, :] = dist_to_rew(ps[:, i2:i2], w_wraps[:, :, i1:i1], 4)
        dists_nowraps[i1, i2, :] = dist_to_rew(ps[:, i2:i2], w_nowraps[:, :, i1:i1], 4)
    end 
end

dists_wraps = dists_wraps[dists_wraps .> 0.5] # ignore self-distances
dists_nowraps = dists_nowraps[dists_nowraps .> 0.5] # ignore self-distances

dists = [dists_wraps, dists_nowraps]
@save "$(datadir)/wrap_and_nowrap_pairwise_dists.bson" dists

