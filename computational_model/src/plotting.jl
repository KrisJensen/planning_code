using PyPlot
using PyCall
### set some reasonable plotting defaults
rc("font"; size=16)
PyCall.PyDict(matplotlib."rcParams")["axes.spines.top"] = false
PyCall.PyDict(matplotlib."rcParams")["axes.spines.right"] = false

function plot_progress(rews, vals; fname="figs/progress.png")
    figure(; figsize=(6, 2.5))
    axs = 120 .+ (1:2)
    data = [rews, vals]
    ts = 1:length(rews)
    labs = ["reward", "prediction"]
    for i in 1:2
        subplot(axs[i])
        plot(ts, data[i], "k-")
        xlabel("epochs")
        ylabel(labs[i])
        title(labs[i])
    end
    tight_layout()
    savefig(fname; bbox_inches="tight")
    close()
    return nothing
end


## plotting utils

function arena_lines(ps, wall_loc, Larena; rew=true, col="k", rew_col = "k", lw_arena = 1., col_arena = ones(3)*0.3, lw_wall = 10)
    Nstates = Larena^2
    for i in 0:Larena
        axvline(i + 0.5; color=col_arena, lw = lw_arena)
        axhline(i + 0.5; color=col_arena, lw = lw_arena)
    end

    if rew
        rew_loc = state_from_onehot(Larena, ps)
        scatter([rew_loc[1]], [rew_loc[2]]; c=rew_col, marker="*", s=350, zorder = 50) #reward location
    end

    for s in 1:Nstates #for each state
        for i in 1:4 #for each neighbor
            if Bool(wall_loc[s, i])
                state = state_from_loc(Larena, s)
                if i == 1 #wall to the right
                    z1, z2 = state + [0.5; 0.5], state + [0.5; -0.5]
                elseif i == 2 #wall to the left
                    z1, z2 = state + [-0.5; 0.5], state + [-0.5; -0.5]
                elseif i == 3 #wall above
                    z1, z2 = state + [0.5; 0.5], state + [-0.5; 0.5]
                elseif i == 4 #wall below
                    z1, z2 = state + [0.5; -0.5], state + [-0.5; -0.5]
                end
                plot([z1[1]; z2[1]], [z1[2]; z2[2]]; color=col, ls="-", lw=lw_wall)
            end
        end
    end
    xlim(0.49, Larena + 0.52)
    ylim(0.48, Larena + 0.51)
    xticks([])
    yticks([])
    return axis("off")
end

function plot_arena(ps, wall_loc, Larena; ind=1)
    ps = ps[:, ind]
    wall_loc = wall_loc[:, :, ind]
    figure(; figsize=(6, 6))
    arena_lines(ps, wall_loc, Larena)
    savefig("figs/wall/test_arena.png"; bbox_inches="tight")
    return close()
end


function plot_rollout(state, rollout, wall, Larena)
    if Bool(rollout[end]) col = [0.5, 0.8, 0.5] else col = [0.5, 0.5, 0.8] end
    rollout = Int.(rollout)
    #new_state = state
    for a = rollout[1:length(rollout)-1]
        if a > 0.5
            if wall[state_ind_from_state(Larena, state)[1], a] > 0.5
                new_state = state
            else
                new_state = state + [[1;0],[-1;0],[0;1],[0;-1]][a]
            end
            new_state = (new_state .+ Larena .- 1) .% Larena .+ 1
            x1, x2 = [f(state[1], new_state[1]) for f = [min, max]]
            y1, y2 = [f(state[2], new_state[2]) for f = [min, max]]

            lw = 5
            #println(a, " ", state, " ", new_state, " ", x1, " ", x2, " ", y1, " ", y2)
            if x2 - x1 > 1.5
                plot([x1,x1-0.5], [y1,y2], ls = "-", color = col, lw = lw)
                plot([x2,x2+0.5], [y1,y2], ls = "-", color = col, lw = lw)
            elseif y2 - y1 > 1.5
                plot([x1,x2], [y1,y1-0.5], ls = "-", color = col, lw = lw)
                plot([x1,x2], [y2,y2+0.5], ls = "-", color = col, lw = lw)
            else
                plot([x1,x2], [y1,y2], ls = "-", color = col, lw = lw)
            end
            state = new_state
        end
    end
end

function plot_weiji_gif(
    ps,
    wall_loc,
    states,
    as,
    rews,
    Larena,
    RTs,
    fname;
    Tplot=10,
    res = 60,
    minframe = 3, #number of movement frames
    figsize = (4,4),
    rewT = 400, #delay at reward in ms
    fix_first_RT = true,
    first_RT = 500,
    plot_rollouts = false, #do we explicitly plot rollouts?
    rollout_time = 120, #duration of rollout plotting in ms
    rollouts = [], #array of actual rollouts,
    dpi = 80, #image resolution
    plot_rollout_frames = false #plot frames for rollouts even if we don't plot the rollouts
)
    #plot gif of agent moving through each batch
    #ps is Nstates x batch
    #wall_loc is Nstates x 4 x batch
    #states is 2 x batch x Tmax
    #as is batch x T
    #RTs are the reaction times for each step in ms
    #T_act is the time taken for an action in ms
    #T_rew is the time taken at reward in ms
    #res is the resolution (ms / frame)
    #Tplot is number of _seconds_ to plot (in real time)

    ##the minimum plotted reaction time is res*minframe

    run(`sh -c "mkdir -p $(fname)_temp"`)
    Tplot = Tplot*1e3/res #now in units of frames
    if fix_first_RT RTs[:, 1] .= first_RT end #fix the first RT since we think this is probably quite noisy

    for batch in 1:size(ps, 2)
        bstr = lpad(batch, 2, "0")
        rew_loc = state_from_onehot(Larena, ps[:, batch])
        ### plot arena
        ### plot movement
        Rtot = 0
        t = 0 # real time
        anum = 0 # number of actions
        rew_col = "lightgrey"

        while (anum < sum(as[batch, :] .> 0.5)) && (t < Tplot)
            anum += 1

            astr = lpad(anum, 3, "0")
            println(bstr, " ", astr)

            RT = RTs[batch, anum] #reaction time for this action
            nframe = max(Int(round(RT/res)), minframe) #number of frames to plot (at least three)
            rewframes = 0 #no reward frames

            if rews[batch, anum] > 0.5
                rewframes = Int(round(rewT/res)) #add a singular frame at reward
            end
            
            if (anum > 1.5) && (rews[batch, anum-1] > 0.5)
                #Rtot += 1 #update total reward
                rew_col = "k" #show that we've found the reward
            end

            R_increased = false #have we increased R for this action
            frames = (minframe - nframe + 1):(minframe+rewframes)
            frolls = Dict(f => 0 for f = frames) #dictionary pointing to rollout; 0 is None

            if plot_rollouts || plot_rollout_frames #either plot rollouts or the corresponding frames
                nroll = sum(rollouts[batch, anum, 1, :] .> 0.5) #how many rollouts?
                println("rolls: ", nroll)
                f_per_roll = Int(round(rollout_time/res)) #frames per rollout
                frames = min(frames[1], -nroll*f_per_roll+1):frames[end] #make sure there is enough frames for plotting rollouts
                frolls = Dict(f => 0 for f = frames) #dictionary pointing to rollout; 0 is None

                for roll = 1:nroll
                    new_rolls = (-(f_per_roll*roll-1):-(f_per_roll*(roll-1))) # f_per_roll frame intervals
                    if nroll == 1 frac = 0.5 else frac = (roll-1)/(nroll-1) end #[0, 1]
                    new_roll_1 = Int(round(frames[1]*frac - (f_per_roll-1)*(1-frac)))
                    new_rolls = new_roll_1:(new_roll_1+f_per_roll-1)
                    for r = new_rolls frolls[r] = nroll-roll+1 end #; println(roll, " ", r) end
                end
            end


            for f = frames
                state = states[:, batch, anum]
                fstr = lpad(f - frames[1], 3, "0")

                frac = min(max(0, (f - 1) / minframe), 1)
                figure(; figsize=figsize)

                arena_lines(ps[:, batch], wall_loc[:, :, batch], Larena; col="k", rew_col = rew_col)

                col = "b"
                if (rewframes > 0) && (frac >= 1)
                    col = "g" #colour green when at reward
                    if ~R_increased Rtot, R_increased = Rtot+1, true end #increase R because we found the reward
                end

                if plot_rollouts #plot the rollout
                    if frolls[f] > 0.5 plot_rollout(state, rollouts[batch, anum, :, frolls[f]], wall_loc[:, :, batch], Larena) end
                end

                a = as[batch, anum] #higher resolution
                state += frac * [Int(a == 1) - Int(a == 2); Int(a == 3) - Int(a == 4)] #move towards next state
                state = (state .+ Larena .- 0.5) .% Larena .+ 0.5
                scatter([state[1]], [state[2]]; marker="o", color=col, s=200, zorder = 100)

                tstr = lpad(t, 3, "0")
                t += 1
                realt = t*res*1e-3
                println(step, " ", f, " ", round(frac, digits = 2), " ", RT, " ", round(realt, digits=1))
                title("t = " * string(round(realt, digits=1)) * " (R = " * string(Rtot) * ")")

                astr = lpad(anum, 2, "0")
                if t <= Tplot
                    savefig(
                        "$(fname)_temp/temp" * bstr * "_" * tstr * "_" * fstr * "_" * astr * ".png";
                        bbox_inches="tight",
                        dpi=dpi,
                    )
                end
                close()
            end
        end
    end

    ###combine pngs to gif
    run(`convert -delay $(Int(round(res/10))) "$(fname)_temp/temp*.png" $fname.gif`)
    #return
    return run(`sh -c "rm -r $(fname)_temp"`)
end



