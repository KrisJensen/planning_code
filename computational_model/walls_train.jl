using Pkg; Pkg.activate(".")
using ToPlanOrNotToPlan
using Flux, Statistics, Random, Distributions
using StatsFuns, Zygote, ArgParse, NaNStatistics
using Logging

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--Nhidden"
        help = "Number of hidden units"
        arg_type = Int
        default = 100
        "--Larena"
        help = "Arena size (per side)"
        arg_type = Int
        default = 4
        "--T"
        help = "Number of timesteps per episode"
        arg_type = Int
        default = 50
        "--Lplan"
        help = "planning horizon"
        arg_type = Int
        default = 8
        "--load"
        help = "load previous model with same parameters"
        arg_type = Bool
        default = false
        "--load_epoch"
        help = "which epoch to load"
        arg_type = Int
        default = 0
        "--seed"
        help = "random seed to use"
        arg_type = Int
        default = 1
        "--task"
        help = "which task to learn"
        arg_type = String
        default = "maze"
        "--nonlinearity"
        help = "which nonlinearity to use in the RNN (GRU/relu)"
        arg_type = String
        default = "GRU"
        "--save_dir"
        help = "save directory"
        arg_type = String
        default = "./"
        "--beta_p"
        help = "predictive coding scale"
        arg_type = Float32
        default = 0.5f0
        "--no_planning"
        help = "prevent any planning"
        arg_type = Int
        default = 0
        "--prefix"
        help = "add prefix to model name"
        arg_type = String
        default = ""
        "--load_fname"
        help = "model to load (default to model name)"
        arg_type = String
        default = ""
        "--n_epochs"
        help = "total number of training epochs"
        arg_type = Int
        default = 1001
        "--batch_size"
        help = "batch size for each gradient step"
        arg_type = Int
        default = 40
        "--lrate"
        help = "learning rate"
        arg_type = Float64
        default = 1e-3
    end
    return parse_args(s)
end

function main()

    ##### global parameters #####

    args = parse_commandline()
    println(args)

    #load various helpter functions (these use Larena)
    task = args["task"]
    Larena = args["Larena"]
    Lplan = args["Lplan"]
    Nhidden = args["Nhidden"]
    T = args["T"]
    load = args["load"]
    seed = args["seed"]
    save_dir = args["save_dir"]
    prefix = args["prefix"]
    load_fname = args["load_fname"]
    n_epochs = Int(args["n_epochs"])
    βp = Float32(args["beta_p"])
    batch_size = Int(args["batch_size"])
    lrate = Float64(args["lrate"])
    βe_0 = 0.05f0
    no_planning = Bool(args["no_planning"])

    println("prior: ", prior_type, " ", epsilon)
    if no_planning println("We're preventing planning!!!") end

    Base.Filesystem.mkpath(save_dir)

    Random.seed!(seed)

    loss_hp = LossHyperparameters(;
        #predictive coding factor
        βp=βp,
        #value function approximation factor
        βv=0.05f0,
        #entropy cost
        βe=βe_0,
        #action cost
        βa=0.0f0,
        #reward factor
        βr=1.0f0,
        # discount factor
        γ=1.0f0,
        #train on predict loss?
        predict=true,
        #available actions
        Naction = 5,
        #arena size
        Larena = Larena,
    )

    arena = build_arena(Larena)
    model_properties, wall_environment, model_eval = build_environment(
        arena, Nhidden, T; Lplan, task=task, no_planning = no_planning
    )

    Nstates = wall_environment.properties.dimensions.Nstates
    m = build_model(model_properties, 5, nonlinearity=args["nonlinearity"], Nstates = Nstates)
    
    mod_name = create_model_name(
        Larena, Nhidden, T, seed, Lplan, βp = βp, no_planning = no_planning, prefix = prefix
    )

    #training parameters
    n_batches, save_every = 200, 100
    opt = ADAM(lrate)

    #used to keep track of progress
    rews, preds = [], []
    epoch = 0 #start at epoch 0

    if Lplan > 0 #do we plan?
        println("planning! ", model_properties.Nin, " ", model_properties.Nout)
    end

    @info "model name" mod_name
    @info "training info" n_epochs n_batches batch_size

    if load #if we load a previous model
        if load_fname == ""
            fname = "$save_dir/models/$task/" * mod_name * "_" * string(args["load_epoch"])
        else
            fname = "$save_dir/models/$task/" * load_fname
        end
        network, opt, store, hps, policy, prediction = recover_model(fname, modular = true)
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)

        rews, preds = store[1], store[2]
        println("previous steps: ", length(rews))
        epoch = length(rews) #start where we were
        if load_fname != "" #loaded from pretraining; reset opt
            opt = ADAM(lrate)
        end
    end

    prms = Params(Flux.params(m.network, m.policy, m.prediction)) #model parameters
    println("parameter length: ", length(prms))
    for p = prms println(size(p)) end


    Nthread = Threads.nthreads()
    multithread = Nthread > 1
    thread_batch_size = Int(ceil(batch_size / Nthread)) #distribute across threads
    function get_closure(multithread, ponder)
        if multithread
            @info "multithreading" Nthread
            return () ->
                model_loss(m, wall_environment, loss_hp, thread_batch_size, ponder = ponder) / Nthread
        else
            return () -> model_loss(m, wall_environment, loss_hp, batch_size, ponder = ponder) #function to optimize
        end
    end

    closure = get_closure(multithread, ponder)

    gmap_grads(g1, g2) = gmap(+, prms, g1, g2) #define map function for reducing gradients
    function loop!(batch, closure)
        if multithread #distribute across threads?
            all_gs = Vector{Zygote.Grads}(undef, Nthread) #vector of gradients for each thread
            Threads.@threads for i in 1:Nthread #on each thread
                rand_roll = rand(100*batch*i) #run through some random numbers
                gs = gradient(closure, prms) #compute gradient
                all_gs[i] = gs #save gradient
            end
            gs = reduce(gmap_grads, all_gs) #sum across our gradients
        else
            gs = gradient(closure, prms) #if we're not multithreading, just compute a simple gradient
        end
        return Flux.Optimise.update!(opt, prms, gs) #update model parameters
    end

    t0 = time()
    while epoch < n_epochs
        epoch += 1
        flush(stdout) #flush output in case we're on cluster

        Rmean, pred, mean_a, first_rew, bias = model_eval(
            m, batch_size, loss_hp, ponder = ponder
        ) #evaluate performance
        Flux.reset!(m) #reset model

        if (epoch - 1) % save_every == 0 #occasionally save our model in case we want to restart
            Base.Filesystem.mkpath("$save_dir/models/$task")
            filename = "$save_dir/models/$task/" * mod_name * "_" * string(epoch - 1)
            store = [rews, preds]
            save_model(m, store, opt, filename, wall_environment, loss_hp; ponder, Lplan)
        end

        elapsed_time = round(time() - t0; digits=1)
        @info "progress" epoch elapsed_time Rmean pred mean_a first_rew bias
        println("progress: epoch=$epoch t=$elapsed_time R=$Rmean pred=$pred plan=$mean_a first=$first_rew bias=$bias")
        push!(rews, Rmean)
        push!(preds, pred)
        plot_progress(rews, preds)

        for batch in 1:n_batches #for each batch
            loop!(batch, closure)
        end
    end

    Flux.reset!(m) #reset model state
    filename = "$save_dir/results/$task" * "_" * mod_name
    store = [rews, preds]
    return save_model(m, store, opt, filename, wall_environment, loss_hp; ponder, Lplan)
end
main()
