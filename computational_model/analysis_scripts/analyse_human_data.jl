## load scripts and model
include("anal_utils.jl")
using ToPlanOrNotToPlan
using NaNStatistics, MultivariateStats, Flux, Random, Statistics, SQLite, DataFrames
using ImageFiltering, StatsBase
using BSON: @save

for game_type = ["play"; "follow"]

### build RL environment ###
T = 100
Larena = 4
environment_dimensions = EnvironmentDimensions(Larena^2, 2, 5, T, Larena)

all_RTs, all_trial_nums, all_trial_time, all_rews, all_states, all_shot = [], [], [], [], [], []
all_wall_loc, all_ps, all_as = [], [], []
Nepisodes = []

db = SQLite.DB("../../human_data/prolific_data.sqlite")
users = (DBInterface.execute(db, "SELECT id FROM users") |> DataFrame)[:, "id"]
if game_type == "play" nskip = 2 else nskip = 8 end #number of initial episodes to discard

for user_id = users
    user_eps = DBInterface.execute(db, "SELECT * FROM episodes WHERE user_id = "*string(user_id[1])) |> DataFrame #episode data
    usize = size(user_eps, 1) #total number of episodes for this user
    info = DBInterface.execute(db, "SELECT * FROM users WHERE id = "*string(user_id)) |> DataFrame
    if (usize >= 58) && (length(info[1, "token"]) == 24) #finished task and prolific-sized token
        println(user_id)
        rews, as, states, wall_loc, ps, times, trial_nums, trial_time, RTs, shot = extract_maze_data(db, user_id, Larena, max_RT = Inf, game_type = game_type, skip_init = nskip)
        append!(all_RTs, [RTs])
        append!(all_rews, [rews])
        append!(all_trial_nums, [trial_nums])
        append!(all_trial_time, [trial_time])
        append!(all_states, [states])
        append!(all_shot, [shot])
        append!(all_wall_loc, [wall_loc])
        append!(all_ps, [ps])
        append!(all_as, [as])
        append!(Nepisodes, size(ps, 2))
    end
end
valid_users = 1:length(all_rews)

data = [all_states, all_ps, all_as, all_wall_loc, all_rews, all_RTs, all_trial_nums, all_trial_time]
@save "$(datadir)/human_all_data_$game_type.bson" data

data = Dict("all_rews" => all_rews, "all_RTs" => all_RTs)
@save "$(datadir)/human_RT_and_rews_$game_type.bson" data

cat_RTs, cat_rews, cat_trial_nums, cat_trial_time, cat_as = [reduce(vcat, arr) for arr = [all_RTs, all_rews, all_trial_nums, all_trial_time, all_as]]
cat_states, cat_shot = reduce(hcat, all_states), reduce(hcat, all_shot)
cat3(x1, x2) = cat(x1, x2, dims = 3)
cat_ps, cat_wall_loc = reduce(hcat, all_ps), reduce(cat3, all_wall_loc)

### compute steps per trial
function comp_rew_by_step(rews; Rmin = 3)
    keep_inds = findall( sum(rews .> 0.5, dims = 2)[:] .>= Rmin )
    all_durs = zeros(length(keep_inds), Rmin)
    for (ib, b) = enumerate(keep_inds)
        sortrew = sortperm(-rews[b, :])
        rewtimes = [0; sortrew[1:Rmin]]
        durs = rewtimes[2:Rmin+1] - rewtimes[1:Rmin]
        all_durs[ib, :] = durs
    end
    μ = mean(all_durs, dims = 1)[:]
    s = std(all_durs, dims = 1)[:] / sqrt(length(keep_inds))
    return μ, s
end

μs, ss = [], []
Rmin = 4
for i = valid_users
    μ, s = comp_rew_by_step(all_rews[i], Rmin = Rmin)
    push!(μs, μ)
    push!(ss, s)
end
μs = reduce(hcat, μs)
ss = reduce(hcat, ss)

data = [Rmin, μs, ss]
@save "$(datadir)/human_by_trial_$game_type.bson" data

## RT by difficulty
function human_RT_by_difficulty(as, T, rews, ps, wall_loc, Larena, trial_nums, trial_time, RTs, states)
    trials = 15
    new_RTs = zeros(trials, size(as, 1), T) .+ NaN
    new_dists = zeros(trials, size(as, 1)) .+ NaN
    for b = 1:size(as, 1)
        rew = rews[b, :] #rewards in this episode
        min_dists = dist_to_rew(ps[:, b:b], wall_loc[:, :, b:b], Larena) #minimum distances to goal for each state
        for trial = 2:trials
            if sum(rew .> 0.5) .> (trial - 0.5) #finish trial
                inds = findall((trial_nums[b, :] .== trial) .& (trial_time[b, :] .> 0.5)) #all timepoints within trial
                new_RTs[trial, b, 1:length(inds)] = RTs[b, inds] #reaction times
                state = states[:, b, inds[1]] #initial state
                new_dists[trial, b] = min_dists[Int(state[1]), Int(state[2])]
            end
        end
    end
    return new_RTs, new_dists
end

### repeat by person ###
RTs, dists = [], []
for u = valid_users
    new_RTs, new_dists = human_RT_by_difficulty(all_as[u], T, all_rews[u], all_ps[u], all_wall_loc[u], Larena, all_trial_nums[u], all_trial_time[u], all_RTs[u], all_states[u])
    push!(RTs, new_RTs); push!(dists, new_dists)
end
data = [RTs, dists, all_trial_nums, all_trial_time]
@save "$(datadir)RT_by_complexity_by_user_$game_type.bson" data

## compute RT by unique states visited ##

all_unique_states = []
for i = 1:length(all_RTs)
    states, rews, as = all_states[i], all_rews[i], all_as[i]
    unique_states = zeros(size(all_RTs[i])) .+ NaN #how many states had been seen when the action was taken
    for b = 1:size(rews,1)
        if sum(rews[b, :]) == 0 tmax = sum(as[b, :] .> 0.5) else tmax = findall(rews[b, :] .== 1)[1] end
        visited = Bool.(zeros(16)) #which states have been visited
        for t = 1:tmax
            visited[Int(state_ind_from_state(Larena, states[:,b,t])[1])] = true
            unique_states[b, t] = sum(visited)
        end
    end
    push!(all_unique_states, unique_states)
end
data = [all_RTs, all_unique_states]
@save "$(datadir)unique_states_$game_type.bson" data

end

