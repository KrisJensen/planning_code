using Flux, Statistics, Random, Distributions, StatsFuns, Zygote
const GRUind = 1

"""
    sample_actions(model, policy_logits)
function for sampling actions from the policy of the agent
"""
function sample_actions(mod, policy_logits)
    Zygote.ignore() do #don't differentiate through this sampling process
        batch = size(policy_logits, 2) #batch size
        a = zeros(Int32, 1, batch) #initialize action array

        πt = exp.(Float64.(policy_logits)) #probability of actions (up/down/right/left/stay)
        πt ./= sum(πt; dims=1) #normalize over actions
        if any(isnan.(πt)) || any(isinf.(πt)) #set to argmax(log pi) if exponentiating gave infs
            for b in 1:batch
                if any(isnan.(πt[:, b])) || any(isinf.(πt[:, b]))
                    πt[:, b] = zeros(size(πt[:, b]))
                    πt[argmax(policy_logits[:, b]), b] = 1
                end
            end
        end

        if mod.model_properties.greedy_actions #optionally select greedy action
            a[:] = Int32.(argmax.([πt[:, b] for b = 1:batch]))
        else #sample action from policy
            a[:] = Int32.(rand.(Categorical.([πt[:, b] for b = 1:batch])))
        end

        return a
    end
end

"""
set reward, action and input to zero for episodes that are finished
"""
function zeropad_data(rew, agent_input, a, active)
    Zygote.ignore() do
        finished = findall( .~ active) #episodes that are not active are finished
        if length(finished) > 0.5 #if there are finished episodes
            rew[:, finished] .= 0f0 #set reward to zero
            agent_input[:, finished] .= 0f0 #set input to zero
            a[:, finished] .= 0f0 #set action to zero
        end
        return rew, agent_input, a #return zero-padded data
    end
end

"""
calculate prediction loss for the transition component of the internal world model.
cross-entropy loss between output distribution and true next state.
"""
function calc_prediction_loss(agent_output, Naction, Nstates, s_index, active)
    new_Lpred = 0.0f0 #initialize prediction loss
    spred = agent_output[(Naction + 1 + 1):(Naction + 1 + Nstates), :] #predicted next states (Nstates x batch)
    spred = spred .- Flux.logsumexp(spred; dims=1) #softmax over states
    for b in findall(active) #only active episodes
        new_Lpred -= spred[s_index[b], b] # -log p(s_true)
    end
    return new_Lpred #return summed loss
end

"""
calculate prediction loss for the reward component of the internal world model.
cross-entropy loss between output distribution and true reward location.
"""
function calc_rew_prediction_loss(agent_output, Naction, Nstates, r_index, active)
    new_Lpred = 0.0f0 #initialize prediction loss
    i1 = (Naction + Nstates + 3) #first index of output distribution
    i2 = (Naction + Nstates + 2 + Nstates) #second index of output distribution
    rpred = agent_output[i1:i2, :] #output distribution
    rpred = rpred .- Flux.logsumexp(rpred; dims=1) #softmax over states
    for b in findall(active) #only active episodes
        new_Lpred -= rpred[r_index[b], b] # -log p(r_true)
    end
    return new_Lpred #return summed loss
end

"""
take action indices and convert to one-hot representation
"""
function construct_ahot(a, Naction)
    Zygote.ignore() do #no gradients through this
        batch = size(a, 2)
        ahot = zeros(Float32, Naction, batch)
        for b = 1:batch ahot[Int(a[1, b]), b] = 1f0 end #set each element to 1
        return Float32.(ahot)
    end
end

"""
    forward_modular(model, env_dimensions, input, hidden_state)
computes the function h_rnn, y_rnn = phi(h_rnn, x_rnn)
"""
function forward_modular(mod, ed::EnvironmentDimensions, x, h_rnn)
    Naction, Nstates, batch = ed.Naction, ed.Nstates, size(x, 2) #useful variables

    h_rnn, ytemp = mod.network[GRUind].cell(h_rnn, x) #forward pass through recurrent part of RNN
    logπ_V = mod.policy(ytemp) #apply policy (and value) network
    logπ = logπ_V[1:Naction, :] #policy is the first few rows
    V = logπ_V[(Naction+1):(Naction+1), :] #value function is the next row

    #optionally impose that only physical actions can be chosen
    no_planning = mod.model_properties.no_planning
    if (typeof(no_planning) == Bool)
        if no_planning logπ = logπ .- [0f0;0f0;0f0;0f0;Inf32] end #reduce p(plan) to zero
    else
        logπ = logπ .- Flux.logsumexp(logπ; dims=1) #softmax
        logπ = [logπ[1:4, :]; logπ[5:5, :] .+ Float32(no_planning)] #reduce by some finite log probability
    end
    logπ = logπ .- Flux.logsumexp(logπ; dims=1) #softmax for normalization
    a = sample_actions(mod, logπ) #1 x batch
    ahot = construct_ahot(a, Naction) #one hot representation of actions

    prediction_input = [ytemp; ahot] #input to prediction module (concatenation of hidden state and action)
    prediction_output = mod.prediction(prediction_input) #output of prediction module

    return h_rnn, [logπ; V; prediction_output], a # return hidden state, network output, and sampled action
end

"""
    calc_deltas(rews, Vs)
Function for computing TD errors.
'rews' indicate empirical rewards on each time step (batch x T)
Vs indicate the value function outputted by the network on the corresponding time step (batch x T)
note that V should approximate the cumulative future reward.
In this work, we do not use bootstrapping or temporal discounting.
This reduces bias (to zero) at the expense of variance.
"""
function calc_deltas(rews, Vs)
    batch, N = size(rews)
    δs = Float32.(zeros(batch, N)) #initialize TD errors
    R = zeros(batch) #cumulative reward
    for t in 0:(N - 1) #for each iteration (moving backward!)
        R = rews[:, N - t] + R #cumulative reward
        δs[:, N - t] = R - Vs[:, N - t] #TD error
    end
    return Float32.(δs)
end
Zygote.@nograd calc_deltas #don't take gradients of this!

"""
    run_episode(model, environment, loss_hp, reward_location, agent_state, hidden, batch, calc_loss, initial_params)
function for running a full episode of the dynamics maze task.
model: Flux model containing the parameters of the RL agent.
environment: environment struct
loss_hp: hyperparameters used to compute loss
reward location [empty]: impose a reward location instead of sampling
agent_state [empty]: impose initial agent state instead of sampling
hidden [true]: whether to collect and return hidden states
batch [2]: batch size
calc_loss [false]: whether to compute the loss
initial params [empty]: impose wall configuration instead of sampling
"""
function run_episode(
    mod,
    environment::Environment,
    loss_hp::LossHyperparameters;
    reward_location=zeros(2),
    agent_state=zeros(2),
    hidden=true,
    batch=2,
    calc_loss = true,
    initial_params = []
)

    #extract some useful variables
    ed = environment.dimensions
    Nout = mod.model_properties.Nout
    Nhidden = mod.model_properties.Nhidden
    Nstates = ed.Nstates
    Naction = ed.Naction
    T = ed.T

    ### initialize reward location, walls, and agent state ###
    world_state, agent_input = environment.initialize(
        reward_location, agent_state, batch, mod.model_properties, initial_params = initial_params
    )

    #arrays for storing outputs y
    agent_outputs = Array{Float32}(undef, Nout, batch, 0)
    if hidden #also store and return hidden states and world states
        hs = Array{Float32}(undef, Nhidden, batch, 0) #hidden states
        world_states = Array{WorldState}(undef, 0) #world states
    end
    actions = Array{Int32}(undef, 1, batch, 0) #list of actions
    rews = Array{Float32}(undef, 1, batch, 0) #list of rewards

    # project initial hidden state to batch_size
    h_rnn = mod.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch))

    Lpred = 0.0f0 #accumulate prediction loss
    Lprior = 0.0f0 #accumulate regularization loss

    #run until all episodes in the batch are finished
    #note that t0 = 1, can take T actions --> last action at T, no more actions at T+1
    while any(world_state.environment_state.time .< (T+1 - 1e-2))
        if hidden
            hs = @Zygote.ignore cat(hs, h_rnn; dims=3) #append hidden state
            world_states = @Zygote.ignore [world_states; world_state] #append world state
        end

        #agent_input is Nin x batch
        h_rnn, agent_output, a = mod.forward(mod, ed, agent_input, h_rnn) #RNN step (agent_output_t \in (nout x batch))
        active = (world_state.environment_state.time .< (T+1 - 1e-2)) #active episodes within the batch (not yet finished)

        # now compute regularization loss
        if calc_loss Lprior += prior_loss(agent_output, world_state.agent_state, active, mod) end
        #step the environment
        rew, agent_input, world_state, predictions = environment.step(
            agent_output, a, world_state, environment.dimensions, mod.model_properties,
            mod, h_rnn
        )
        rew, agent_input, a = zeropad_data(rew, agent_input, a, active) #mask if episode is finished

        agent_outputs = cat(agent_outputs, agent_output; dims=3) #store output (y_t); Nout x batch x T
        rews = cat(rews, rew; dims=3) #store reward (r_t); 1 x batch x T
        actions = cat(actions, a; dims=3) #store action (a_t)

        if calc_loss
            s_index, r_index = predictions #true state and reward locations (to be predicted)
            Lpred += calc_prediction_loss(agent_output, Naction, Nstates, s_index, active)
            Lpred += calc_rew_prediction_loss(agent_output, Naction, Nstates, r_index, active)
        end
    end

    #rews is batch x T, V is batch x T, δs is batch x T
    δs = calc_deltas(rews[1, :, :], agent_outputs[Naction + 1, :, :]) #TD errors

    L = Float32(0.0)
    if calc_loss N = size(rews, 3) else N = 0 end #total number of iterations to compute loss for
    for t = 1:N #iterations
        active = Float32.(actions[1, :, t] .> 0.5) #zero for finished episodes
        Vterm = δs[:, t] .* agent_outputs[Naction + 1, :, t] #value function (batch)
        L -= sum(loss_hp.βv * Vterm .* active) #loss (sum over active episodes through multiplication by 'active')
        for b in findall(Bool.(active)) #for each active episode
            RPE_term = δs[b, t] * agent_outputs[actions[1, b, t], b, t] #PG term
            L -= loss_hp.βr * RPE_term #add to loss
        end
    end
    L += (loss_hp.βp * Lpred) #add predictive loss for internal world model
    L -= loss_hp.βe * Lprior #add prior loss (formulated as a likelihood above)
    L /= batch #normalize by batch

    if hidden
        return L, agent_outputs, rews[1, :, :], actions[1, :, :], world_states, hs #also return environment and hidden states
    end
    return L, agent_outputs, rews[1, :, :], actions[1, :, :], world_state.environment_state
end

"""
wrapper for Flux that takes empty data and return scalar loss
"""
function model_loss(mod, environment::Environment, loss_hp, batch_size)
    loss = run_episode(mod, environment, loss_hp; hidden=false, batch=batch_size)[1] #hidden false does not return hidden states
    return loss
end
