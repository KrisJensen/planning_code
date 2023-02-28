using SQLite, DataFrames, Statistics, HypothesisTests

adict = Dict("[\"Up\"]" => 3, "[\"Down\"]" => 4, "[\"Right\"]" => 1, "[\"Left\"]" => 2)

function find_seps(str)
    seps = [
        findall("]],[[", str; overlap=true)
        findall("]],[]", str; overlap=true)
        findall("[],[[", str; overlap=true)
        findall("[],[]", str; overlap=true)
    ]
    return sort(reduce(hcat, seps)[3, :])
end

function get_wall_rep(wallstr, arena)
    seps = find_seps(wallstr)
    columns = [
        wallstr[3:(seps[4] - 2)],
        wallstr[(seps[4] + 2):(seps[8] - 2)],
        wallstr[(seps[8] + 2):(seps[12] - 2)],
        wallstr[(seps[12] + 2):(length(wallstr) - 2)],
    ]
    subseps = [[0; find_seps(col); length(col) + 1] for col in columns]

    wdict = Dict(
        "[\"Top\"]" => 3, "[\"Bottom\"]" => 4, "[\"Right\"]" => 1, "[\"Left\"]" => 2
    )

    new_walls = zeros(16, 4)
    for (i, col) in enumerate(columns)
        for j in 1:4
            ind = state_ind_from_state(arena, [i; j])[1]
            s1, s2 = subseps[i][j:(j + 1)]
            entries = split(col[((s1 + 2):(s2 - 2))], ",")
            for entry in entries
                if length(entry) > 0.5
                    new_walls[ind, wdict[entry]] = 1
                end
            end
        end
    end
    return new_walls
end

function extract_maze_data(db, user_id, Larena; T=100, max_RT=5000, game_type = "play",
                            skip_init = 1, skip_finit = 0)
    Nstates = Larena^2

    epis = DataFrame(DBInterface.execute(
        db, "SELECT * FROM episodes WHERE user_id = $user_id AND game_type = '$game_type'"
    ))

    if "attention_problem" in names(epis) #discard episodes with a failed attention check
        atts = epis[:, "attention_problem"]
        keep = findall(atts .== "null")
        epis = epis[keep, :]
    end

    ids = epis[:, "id"] #episode ids
    inds = (1+skip_init):(length(ids)-skip_finit) #allow for discarding the first/last few episodes
    ids = ids[inds]
    batch_size = length(ids)

    rews, as, times = zeros(batch_size, T), zeros(batch_size, T), zeros(batch_size, T)
    states = ones(2, batch_size, T)
    trial_nums, trial_time = zeros(batch_size, T), zeros(batch_size, T)
    wall_loc, ps = zeros(16, 4, batch_size), zeros(16, batch_size)
    for b in 1:batch_size
        steps = DataFrame(DBInterface.execute(
            db, "SELECT * FROM steps WHERE episode_id = " * string(ids[b])
        ))
        trial_num = 1
        t0 = 0

        wall_loc[:, :, b] = get_wall_rep(epis[inds[b], "walls"], Larena)
        ps[:, b] = onehot_from_state(Larena, 
            [parse(Int, epis[inds[b], "reward"][i]) for i in [2; 4]] .+ 1
        )
        Tb = size(steps, 1) #steps on this trial

        for i in reverse(1:Tb) #steps are stored in reverse order
            t = steps[i, "step"]
            if t > 0.5 #last action of previous episode can carry over
                times[b, t] = steps[i, "action_time"]
                rews[b, t] = Int(steps[i, "outcome"] == "[\"Hit_reward\"]")
                as[b, t] = adict[steps[i, "action_type"]]
                states[:, b, t] = [parse(Int, steps[i, "agent"][j]) for j in [2; 4]] .+ 1

                trial_nums[b, t] = trial_num
                trial_time[b, t] = t - t0
                if rews[b, t] > 0.5 #found reward
                    trial_num += 1 #next trial
                    t0 = t #reset trial_time
                end
            end
        end
    end

    RTs = [times[:, 1:1] (times[:, 2:T] - times[:, 1:(T - 1)])]
    RTs[RTs .< 0.5] .= NaN #end of trial
    for b in 1:batch_size
        rewtimes = findall(rews[b, 1:T] .> 0.5)
        RTs[b, rewtimes .+ 1] .-= (8 * 50) #after update; subtract the 400 ms showing that we are at reward
    end

    shot = zeros(Nstates, size(states, 2), size(states, 3)) .+ NaN
    for b in 1:size(states, 2)
        Tb = sum(as[1, :] .> 0.5)
        shot[:, b, 1:Tb] = onehot_from_state(Larena, Int.(states[:, b, 1:Tb]))
    end

    inds = 1:size(RTs, 1)
    if max_RT < Inf
        inds = findall(nanmaximum(RTs[:, 1:end]; dims=2)[:] .<= max_RT) #discard trials with a big break
    end
    return (
        rews[inds, :],
        as[inds, :],
        states[:, inds, :],
        wall_loc[:, :, inds],
        ps[:, inds],
        times[inds, :],
        trial_nums[inds, :],
        trial_time[inds, :],
        RTs[inds, :],
        shot[:, inds, :],
    )
end
