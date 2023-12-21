# in this script, we load the human behavioural data and save some useful summary statistics

# load scripts and model
include("anal_utils.jl")
include("euclidean_prolific_ids.jl")
using ToPlanOrNotToPlan
using SQLite, DataFrames, ImageFiltering
using NaNStatistics

println("loading and processing human behavioural data")
wraparound = true

#perform analyses for both non-guided ("play") and guided ("follow") episodes
for game_type = ["play"; "follow"]

# build RL environment
T = 100
Larena = 4
environment_dimensions = EnvironmentDimensions(Larena^2, 2, 5, T, Larena)

#initialize some arrays for storing data
all_RTs, all_trial_nums, all_trial_time, all_rews, all_states = [], [], [], [], []
all_wall_loc, all_ps, all_as = [], [], []
Nepisodes, tokens = [], []

if wraparound
    db = SQLite.DB("../../human_data/prolific_data.sqlite")
    wrapstr = ""
else
    db = SQLite.DB("../../human_data/Euclidean_prolific_data.sqlite")
    wrapstr = "_euclidean"
end

users = (DBInterface.execute(db, "SELECT id FROM users") |> DataFrame)[:, "id"]
if game_type == "play" nskip = 2 else nskip = 8 end #number of initial episodes to discard

println("loading users for game type: $(game_type)")
i_user = 0
for user_id = users
    user_eps = DBInterface.execute(db, "SELECT * FROM episodes WHERE user_id = "*string(user_id[1])) |> DataFrame #episode data
    usize = size(user_eps, 1) #total number of episodes for this user
    info = DBInterface.execute(db, "SELECT * FROM users WHERE id = "*string(user_id)) |> DataFrame
    token = info[1, "token"]
    if (usize >= 58) && (length(token) == 24) #finished task and prolific-sized token
        if wraparound || (token in euclidean_ids)
        i_user += 1; if i_user % 10 == 0 println(i_user) end
        rews, as, states, wall_loc, ps, times, trial_nums, trial_time, RTs, shot = extract_maze_data(db, user_id, Larena, game_type = game_type, skip_init = nskip)
        append!(all_RTs, [RTs]) #reaction times
        append!(all_rews, [rews]) #rewards
        append!(all_trial_nums, [trial_nums]) #trial numbes
        append!(all_trial_time, [trial_time]) #time within trial
        append!(all_states, [states]) #subject locations
        append!(all_wall_loc, [wall_loc]) #wall locations
        append!(all_ps, [ps]) #reward locations
        append!(all_as, [as]) #actions taken
        append!(Nepisodes, size(ps, 2)) #number of episodes for this user
        push!(tokens, info[1, "token"])
        end
    end
end
valid_users = 1:length(all_rews)

println("processing data for $(length(valid_users)) users")

#store all data
data = [all_states, all_ps, all_as, all_wall_loc, all_rews, all_RTs, all_trial_nums, all_trial_time]
@save "$(datadir)/human_all_data_$game_type$wrapstr.bson" data

#store some generally useful data
data = Dict("all_rews" => all_rews, "all_RTs" => all_RTs)
@save "$(datadir)/human_RT_and_rews_$game_type$wrapstr.bson" data

# compute steps by trial number
function comp_rew_by_step(rews; Rmin = 4)
    keep_inds = findall( sum(rews .> 0.5, dims = 2)[:] .>= Rmin ) #only consider episodes with at least Rmin completed trials
    all_durs = zeros(length(keep_inds), Rmin) #container for durations of each trial (in steps)
    for (ib, b) = enumerate(keep_inds) #loop through episodes
        sortrew = sortperm(-rews[b, :]) #find reward times
        rewtimes = [0; sortrew[1:Rmin]]
        durs = rewtimes[2:Rmin+1] - rewtimes[1:Rmin] #difference between reward times
        all_durs[ib, :] = durs #store
    end
    μ = mean(all_durs, dims = 1)[:] #mean
    s = std(all_durs, dims = 1)[:] / sqrt(length(keep_inds)) #standard error
    return μ, s
end

μs, ss = [], []
Rmin = 4
for i = valid_users #compute for each user
    μ, s = comp_rew_by_step(all_rews[i], Rmin = Rmin)
    push!(μs, μ)
    push!(ss, s)
end
μs = reduce(hcat, μs) #combine data
ss = reduce(hcat, ss)

#save data
data = [Rmin, μs, ss]
@save "$(datadir)/human_by_trial_$game_type$wrapstr.bson" data

# RT by distance step and distance to goal
function human_RT_by_difficulty(T, rews, ps, wall_loc, Larena, trial_nums, trial_time, RTs, states)
    trials = 20 #maximum number of trials
    new_RTs = zeros(trials, size(rews, 1), T) .+ NaN #RTs
    new_dists = zeros(trials, size(rews, 1)) .+ NaN #distances to goal
    for b = 1:size(rews, 1) #for each episode
        rew = rews[b, :] #rewards in this episode
        min_dists = dist_to_rew(ps[:, b:b], wall_loc[:, :, b:b], Larena) #minimum distances to goal for each state
        for trial = 2:trials #consider only exploitation
            if sum(rew .> 0.5) .> (trial - 0.5) #finished trial
                inds = findall((trial_nums[b, :] .== trial) .& (trial_time[b, :] .> 0.5)) #all timepoints within trial
                new_RTs[trial, b, 1:length(inds)] = RTs[b, inds] #reaction times
                state = states[:, b, inds[1]] #initial state
                new_dists[trial, b] = min_dists[Int(state[1]), Int(state[2])] #distance to goal from initial state
            end
        end
    end
    return new_RTs, new_dists #return RTs and distances
end

# repeat by participant
RTs, dists = [], []
for u = valid_users
    new_RTs, new_dists = human_RT_by_difficulty(T, all_rews[u], all_ps[u], all_wall_loc[u], Larena, all_trial_nums[u], all_trial_time[u], all_RTs[u], all_states[u])
    push!(RTs, new_RTs); push!(dists, new_dists) #add results to container
end

#write result to a file
data = [RTs, dists, all_trial_nums, all_trial_time]
@save "$(datadir)RT_by_complexity_by_user_$game_type$wrapstr.bson" data

# compute RT by unique states visited during exploration
all_unique_states = []
for i = 1:length(all_RTs) #for each user
    states, rews, as = all_states[i], all_rews[i], all_as[i] #extract states, rewards and actions
    unique_states = zeros(size(all_RTs[i])) .+ NaN #how many states had been seen when the action was taken
    for b = 1:size(rews,1) #for each episode
        if sum(rews[b, :]) == 0 #if there are no finished trials
            tmax = sum(as[b, :] .> 0.5) #iterate until end
        else
            tmax = findall(rews[b, :] .== 1)[1] #iterate until first reward
        end
        visited = Bool.(zeros(16)) #which states have been visited
        for t = 1:tmax #for each action in trial 1
            visited[Int(state_ind_from_state(Larena, states[:,b,t])[1])] = true #visited corresponding state
            unique_states[b, t] = sum(visited) #number of unique states
        end
    end
    push!(all_unique_states, unique_states) #add to container
end

#write data to file
data = [all_RTs, all_unique_states]
@save "$(datadir)unique_states_$game_type$wrapstr.bson" data

end

