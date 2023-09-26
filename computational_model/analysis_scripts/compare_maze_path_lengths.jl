
include("anal_utils.jl")
using ToPlanOrNotToPlan


# figure(figsize = (4,4))
# wall_loc = reshape(permutedims(maz, [2, 1, 3]), prod(size(maz)[1:2]), 4)
# wall_loc = maze(4, wrap = true)
# arena_lines([], wall_loc, 4, rew = false)
# savefig("./test.png", bbox_inches = "tight")
# close()

N = 10000
cat3(a, b) = cat(a, b, dims = 3)
w_wraps = reduce(cat3, [maze(4, wrap = true) for _ in 1:N])
w_nowraps = reduce(cat3, [maze(4, wrap = false) for _ in 1:N])

ps = onehot_from_loc(4, 1:16)
dists_wraps = zeros(N, 16, 16)
dists_nowraps = zeros(N, 16, 16)
for i1 = 1:N
    for i2 = 1:16
        dists_wraps[i1, i2, :] = dist_to_rew(ps[:, i2:i2], w_wraps[:, :, i1:i1], 4)
        dists_nowraps[i1, i2, :] = dist_to_rew(ps[:, i2:i2], w_nowraps[:, :, i1:i1], 4)
    end 
end

dists_wraps = dists_wraps[dists_wraps .> 0.5]
dists_nowraps = dists_nowraps[dists_nowraps .> 0.5]

dists = [dists_wraps, dists_nowraps]
@save "$(datadir)/wrap_and_nowrap_pairwise_dists.bson" dists

# xs = 1:12
# hist_wraps = [sum(dists_wraps .== d) for d = xs]
# hist_nowraps = [sum(dists_nowraps .== d) for d = xs]
# figure()
# bar(xs, hist_wraps, alpha = 0.3)
# bar(xs, hist_nowraps, alpha = 0.3)
# axvline(mean(dists_wraps))
# axvline(mean(dists_nowraps))
# savefig("./test_hist.png", bbox_inches = "tight")
# close()

