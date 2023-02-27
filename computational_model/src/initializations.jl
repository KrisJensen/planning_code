function reset_agent_state(Larena, reward_location, batch)
    Nstates = Larena^2
    agent_state = rand(Categorical(ones(Larena) / Larena), 2, batch) #random starting location (2xbatch)
    #make sure we cannot start at reward!
    for b in 1:batch
        tele_reward_location = ones(Nstates) / (Nstates - 1)
        tele_reward_location[Bool.(reward_location[:, b])] .= 0
        agent_state[:, b] = state_from_loc(
            Larena, rand(Categorical(tele_reward_location), 1, 1)
        )
    end
    return agent_state
end

### task-specific initialization ###
function gen_maze_walls(
    Larena, batch
)
    wall_loc = zeros(Float32, Larena^2, 4, batch) #whether there is a wall between neighboring agent_states
    for b in 1:batch
        wall_loc[:, :, b] = maze(Larena)
    end
    return wall_loc
end

function initialize_arena(reward_location, agent_state, batch, model_properties, environment_properties, initial_plan_state;initial_params = [])
    Zygote.ignore() do
        Larena=environment_properties.dimensions.Larena; Nstates = Larena^2
        rew_loc = rand(Categorical(ones(Nstates) / Nstates), batch)
        if maximum(reward_location) <= 0
            reward_location = zeros(Float32, Nstates, batch) #Nstates x batch
            for b in 1:batch
                reward_location[rew_loc[b], b] = 1.0f0
            end
        end

        if maximum(agent_state) <= 0
            agent_state = reset_agent_state(Larena, reward_location, batch)
        end

        if length(initial_params) > 0 #load environment
            wall_loc = initial_params
        else
            wall_loc = gen_maze_walls(Larena, batch)
        end

        #note: start at t=1 for backwards compatibility
        world_state = WorldState(;
            environment_state=WallState(;
                wall_loc=Int32.(wall_loc), reward_location=Float32.(reward_location), time = ones(Float32, batch),
            ),
            agent_state=Int32.(agent_state),
            planning_state = initial_plan_state(batch)
        )

        ahot = zeros(Float32, environment_properties.dimensions.Naction, batch) #should use 'Naction' from somewhere
        rew = zeros(Float32, 1, batch) #no reward or actions yet
        x = gen_input(world_state, ahot, rew, environment_properties, model_properties)

        return world_state, Float32.(x)
    end
end
