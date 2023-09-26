using Pkg; Pkg.activate(".")
using Revise
using ToPlanOrNotToPlan
using Flux, Statistics, Random, Distributions
using StatsFuns, Zygote, ArgParse, NaNStatistics
using Logging

function parse_commandline()
    s = ArgParseSettings()
    ### set default values of command line options ###
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
        help = "Number of timesteps per episode (in units of physical actions)"
        arg_type = Int
        default = 50
        "--Lplan"
        help = "Maximum planning horizon"
        arg_type = Int
        default = 8
        "--load"
        help = "Load previous model instead of initializing new model"
        arg_type = Bool
        default = false
        "--load_epoch"
        help = "which epoch to load"
        arg_type = Int
        default = 0
        "--seed"
        help = "Which random seed to use"
        arg_type = Int
        default = 1
        "--save_dir"
        help = "Save directory"
        arg_type = String
        default = "./"
        "--beta_p"
        help = "Relative importance of predictive loss"
        arg_type = Float32
        default = 0.5f0
        "--prefix"
        help = "Add prefix to model name"
        arg_type = String
        default = ""
        "--load_fname"
        help = "Model to load (default to default model name)"
        arg_type = String
        default = ""
        "--n_epochs"
        help = "Total number of training epochs"
        arg_type = Int
        default = 1001
        "--batch_size"
        help = "Batch size for each gradient step"
        arg_type = Int
        default = 40
        "--lrate"
        help = "Learning rate"
        arg_type = Float64
        default = 1e-3
        "--constant_rollout_time"
        help = "Do rollouts take a fixed amount of time irrespective of length"
        arg_type = Bool
        default = true
    end
    return parse_args(s)
end

function main()

    ##### global parameters #####
    args = parse_commandline()
    println(args)

    # extract command line arguments
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
    constant_rollout_time = Bool(args["constant_rollout_time"])

    Base.Filesystem.mkpath(save_dir)
    Random.seed!(seed) #set random seed

    loss_hp = LossHyperparameters(;
        # predictive loss weight
        βp=βp,
        # value function loss weight
        βv=0.05f0,
        # entropy loss cost
        βe=0.05f0,
        # reward loss cost
        βr=1.0f0,
    )

    # build RL environment
    model_properties, wall_environment, model_eval = build_environment(
        Larena, Nhidden, T; Lplan, constant_rollout_time
    )
    # build RL agent
    m = build_model(model_properties, 5)
    # construct summary string
    mod_name = create_model_name(
        Nhidden, T, seed, Lplan, prefix = prefix
    )

    #training parameters
    n_batches, save_every = 200, 50
    opt = ADAM(lrate) #initialize optimiser

    #used to keep track of progress
    rews, preds = [], []
    epoch = 0 #start at epoch 0
    @info "model name" mod_name
    @info "training info" n_epochs n_batches batch_size

    if load #if we load a previous model
        if load_fname == "" #filename not specified; fall back to default
            fname = "$save_dir/models/" * mod_name * "_" * string(args["load_epoch"])
        else #load specific model
            fname = "$save_dir/models/" * load_fname
        end
        #load the parameters and initialize model
        network, opt, store, hps, policy, prediction = recover_model(fname)
        m = ModularModel(model_properties, network, policy, prediction, forward_modular)

        #load the learning curve from the previous model
        rews, preds = store[1], store[2]
        epoch = length(rews) #start where we were
        if load_fname != "" #loaded pretrained model; reset optimiser
            opt = ADAM(lrate)
        end
    end

    prms = Params(Flux.params(m.network, m.policy, m.prediction)) #model parameters
    println("parameter length: ", length(prms))
    for p = prms println(size(p)) end

    Nthread = Threads.nthreads() #number of threads available
    multithread = Nthread > 1 #multithread if we can
    @info "multithreading" Nthread
    thread_batch_size = Int(ceil(batch_size / Nthread)) #distribute batch evenly across threads
    #construct function without arguments for Flux
    closure = () -> model_loss(m, wall_environment, loss_hp, thread_batch_size) / Nthread

    gmap_grads(g1, g2) = gmap(+, prms, g1, g2) #define map function for reducing gradients

    #function for training on a single match
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

    t0 = time() #wallclock time
    while epoch < n_epochs
        epoch += 1 #count epochs
        flush(stdout) #flush output

        Rmean, pred, mean_a, first_rew = model_eval(
            m, batch_size, loss_hp
        ) #evaluate performance
        Flux.reset!(m) #reset model

        if (epoch - 1) % save_every == 0 #occasionally save our model
            Base.Filesystem.mkpath("$save_dir/models")
            filename = "$save_dir/models/" * mod_name * "_" * string(epoch - 1)
            store = [rews, preds]
            save_model(m, store, opt, filename, wall_environment, loss_hp; Lplan)
        end

        #print progress
        elapsed_time = round(time() - t0; digits=1)
        println("progress: epoch=$epoch t=$elapsed_time R=$Rmean pred=$pred plan=$mean_a first=$first_rew")
        push!(rews, Rmean)
        push!(preds, pred)
        plot_progress(rews, preds) #plot prorgess

        for batch in 1:n_batches #for each batch
            loop!(batch, closure) #perform an update step
        end
    end

    Flux.reset!(m) #reset model state
    # save model
    filename = "$save_dir/results/" * mod_name
    store = [rews, preds]
    return save_model(m, store, opt, filename, wall_environment, loss_hp; Lplan)
end
main()
