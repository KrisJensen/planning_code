using Flux, Statistics, Random, Distributions, StatsFuns, Zygote
Tmin = 1
const GRUind = 1

function sample_actions(mod, policy_logits)
    Zygote.ignore() do
        batch = size(policy_logits, 2)
        a = zeros(Int32, 1, batch)

        πt = exp.(Float64.(policy_logits)) #probability of actions (up/down/right/left/stay)
        πt ./= sum(πt; dims=1) #normalize over actions
        if any(isnan.(πt)) || any(isinf.(πt)) #set to argmax(log pi) if the exponential gives infs
            for b in 1:batch
                if any(isnan.(πt[:, b])) || any(isinf.(πt[:, b]))
                    πt[:, b] = zeros(size(πt[:, b]))
                    πt[argmax(policy_logits[:, b]), b] = 1
                end
            end
        end

        if mod.model_properties.greedy_actions #select greedy action
            a[:] = Int32.(argmax.([πt[:, b] for b = 1:batch]))
        else #sample action
            a[:] = Int32.(rand.(Categorical.([πt[:, b] for b = 1:batch])))
        end

        return a
    end
end

function zeropad_data(rew, agent_input, a, active)
    #set reward/action/input to zero for episodes that are finished
    Zygote.ignore() do
        finished = findall( .~ active)
        if length(finished) > 0.5
            rew[:, finished] .= 0f0
            agent_input[:, finished] .= 0f0
            a[:, finished] .= 0f0
        end
        return rew, agent_input, a
    end
end

function calc_prediction_loss(agent_output, Naction, Nstates, s_index, active)
    new_Lpred = 0.0f0
    spred = agent_output[(Naction + 1 + 1):(Naction + 1 + Nstates), :] #predicted states (Nstates x batch)
    spred = spred .- Flux.logsumexp(spred; dims=1) #softmax over states
    for b in findall(active)
        new_Lpred -= spred[s_index[b], b] # -log p(s_true)
    end
    return new_Lpred
end

function calc_rew_prediction_loss(agent_output, Naction, Nstates, r_index, active)
    #println("predicting reward")
    new_Lpred = 0.0f0
    i1 = (Naction + Nstates + 2)
    i2 = (Naction + Nstates + 1 + Nstates)
    rpred = agent_output[i1:i2, :]
    rpred = rpred .- Flux.logsumexp(rpred; dims=1) #softmax over states
    for b in findall(active)
        new_Lpred -= rpred[r_index[b], b] # -log p(r_true)
    end
    return new_Lpred
end

function construct_ahot(a, Naction)
    Zygote.ignore() do
        batch = size(a, 2)
        ahot = zeros(Float32, Naction, batch)
        for b = 1:batch ahot[Int(a[1, b]), b] = 1f0 end
        return Float32.(ahot)
    end
end

function get_xpred(x, wall_loc, Nstates, Naction)
    Zygote.ignore() do
        xpred = zeros(Float32,Nstates+4, size(x,2))
        shot = x[(Naction + 3):(Naction + 2 + Nstates), :]
        xpred[1:Nstates, :] = shot
        for b = 1:size(x, 2)
            xpred[Nstates+1 : Nstates+4, b] = wall_loc[argmax(shot[:, b]), :, b] #local wall inofmration
        end
        return Float32.(xpred) #Nstates+4 x batch
    end
end

function nograd_input(x)
    Zygote.ignore() do
        return x
    end
end

function forward_modular(mod, ed::EnvironmentDimensions, x, h_rnn)
    #this just adds a softmax to the policy output
    #m is a model
    #x is Nin x batch
    Naction, Nstates, batch = ed.Naction, ed.Nstates, size(x, 2)

    h_rnn, ytemp = mod.network[GRUind].cell(h_rnn, x)
    logπ_V = mod.policy(ytemp) #policy is linear readout
    logπ = logπ_V[1:Naction, :]
    V = logπ_V[(Naction+1):(Naction+1), :]

    #option not to stand still
    no_planning = mod.model_properties.no_planning
    if (typeof(no_planning) == Bool)
        if no_planning logπ = logπ .- [0f0;0f0;0f0;0f0;Inf32] end
    else
        logπ = logπ .- Flux.logsumexp(logπ; dims=1) #softmax
        logπ = [logπ[1:4, :]; logπ[5:5, :] .+ Float32(no_planning)]
    end
    logπ = logπ .- Flux.logsumexp(logπ; dims=1) #softmax

    a = sample_actions(mod, logπ) #1 x batch
    ahot = construct_ahot(a, Naction)

    prediction_input = [ytemp; ahot] #input to prediction module

    prediction_output = mod.prediction(prediction_input)

    return h_rnn, [logπ; V; prediction_output], a # Nout x batch
end

function calc_deltas(rews, Vs)
    #compute RPEs
    #rews is batch x T
    #Vs is batch x T
    batch, N = size(rews)
    δs = Float32.(zeros(batch, N))
    R = zeros(batch)
    for t in 0:(N - 1) #compute RPE
        R = rews[:, N - t] + R #Reward
        δs[:, N - t] = R - Vs[:, N - t] #error
    end
    return Float32.(δs)
end
Zygote.@nograd calc_deltas #don't take gradients of this

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
    ed = environment.dimensions
    Nout = mod.model_properties.Nout
    Nhidden = mod.model_properties.Nhidden
    Nstates = ed.Nstates
    Naction = ed.Naction
    T = ed.T

    ### initialize reward probabilities and state ###
    world_state, agent_input = environment.initialize(
        reward_location, agent_state, batch, mod.model_properties, initial_params = initial_params
    )

    #arrays for storing output
    #agent_outputs = Matrix{Float32}(undef, Nout,0)
    agent_outputs = Array{Float32}(undef, Nout, batch, 0)
    if hidden #also store and return hidden states and world states
        hs = Array{Float32}(undef, Nhidden, batch, 0)
        world_states = Array{WorldState}(undef, 0) #Nstate_rep is the non-one-hot size
    end
    actions = Array{Int32}(undef, 1, batch, 0)
    rews = Array{Float32}(undef, 1, batch, 0)

    ### reset model and initialize input!! ###
    h_rnn = mod.network[GRUind].cell.state0 .+ Float32.(zeros(Nhidden, batch)) #expand hidden state

    Lpred = 0.0f0
    Lprior = 0.0f0
    Laux = 0.0f0
    #note that t0 = 1, can take T actions --> last action at T, no more actions at T+1
    while any(world_state.environment_state.time .< (T+1 - 1e-2))
        if hidden
            hs = @Zygote.ignore cat(hs, h_rnn; dims=3)
            world_states = @Zygote.ignore [world_states; world_state]
        end

        #agent_input is Nin x batch
        h_rnn, agent_output, a = mod.forward(mod, ed, agent_input, h_rnn) #RNN step agent_output_t = (pi_t(s_t), V_t(s_t)) (nout x batch)
        active = (world_state.environment_state.time .< (T+1 - 1e-2)) #active heads
        agent_input_old = agent_input

        ### now compute prior ###
        if calc_loss Lprior += prior_loss(agent_output, world_state.agent_state, active, mod) end

        rew, agent_input, world_state, predictions = environment.step(
            agent_output, a, world_state, environment.dimensions, mod.model_properties,
            mod, h_rnn
        )

        s_index, r_index = predictions

        rew, agent_input, a = zeropad_data(rew, agent_input, a, active) #mask if episode is finished

        agent_outputs = cat(agent_outputs, agent_output; dims=3) #store output (y_t); Nout x batch x T
        rews = cat(rews, rew; dims=3) #store reward (r_t); 1 x batch x T
        actions = cat(actions, a; dims=3) #store action (a_t)

        if calc_loss
            Lpred += calc_prediction_loss(agent_output, Naction, Nstates, s_index, active)
            Lpred += calc_rew_prediction_loss(agent_output, Naction, Nstates, r_index, active)
        end
    end

    δs = calc_deltas(
        rews[1, :, :], agent_outputs[Naction + 1, :, :]
    ) #rews is batch x T, V is batch x T, δs is batch x T
    L = Float32(0.0)

    if calc_loss N = size(rews, 3) else N = 0 end #total number of iterations
    for t = 1:N

        active = Float32.(actions[1, :, t] .> 0.5) #zero for finished batches; multiply with various batch-wise losses

        Vterm = δs[:, t] .* agent_outputs[Naction + 1, :, t] #value function (batch)
        L -= sum(loss_hp.βv * Vterm .* active) #loss (sum over active batches)
        for b in findall(Bool.(active))
            RPE_term = δs[b, t] * agent_outputs[actions[1, b, t], b, t] #policy term
            L -= loss_hp.βr * RPE_term
        end
    end
    L += (loss_hp.βp * Lpred)
    L -= loss_hp.βe * Lprior #add prior loss (formulated as a likelihood above)
    L /= batch #normalize by batch

    if hidden
        return L, agent_outputs, rews[1, :, :], actions[1, :, :], world_states, hs
    end
    return L, agent_outputs, rews[1, :, :], actions[1, :, :], world_state.environment_state
end

function model_loss(mod, environment::Environment, loss_hp, batch_size)
    #wrapper for Flux that takes empty data
    loss = run_episode(mod, environment, loss_hp; hidden=false, batch=batch_size)[1]
    return loss
end
