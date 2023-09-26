function neighbor(cell, dir, msize)
    neigh = ((cell + 1 * dir .+ msize .- 1) .% msize) .+ 1
    return neigh
end

function neighbors(cell, msize; wrap = true)
    dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    Ns = [cell+ dirs[a] for a = 1:4]
    as = 1:4
    if wrap # states outside arena pushed to other side
        Ns = [((N .+ msize .- 1) .% msize) .+ 1 for N in Ns]
    else # states outside arena not considered 'neighbors'
        inds = findall( (minimum.(Ns) .> 0.5) .& (maximum.(Ns) .< msize+0.5) )
        Ns, as = Ns[inds], as[inds]
    end
    return Ns, as
end

function walk(maz::Array, nxtcell::Vector, msize, visited::Vector=[]; wrap = true)
    dir_map = Dict(1 => 2, 2 => 1, 3 => 4, 4 => 3)
    push!(visited, (nxtcell[1] - 1) * msize + nxtcell[2]) #add to list of visited cells

    neighs, as = neighbors(nxtcell, msize, wrap = wrap) # get list of neighbors

    for nnum in randperm(length(neighs)) #for each neighbor in randomly shuffled list
        neigh, a = neighs[nnum], as[nnum] # corresponding state and action
        ind = (neigh[1] - 1) * msize + neigh[2] # convert from coordinates to index
        if ind ∉ visited #check that we haven't been there
            maz[nxtcell[1], nxtcell[2], a] = 0.0f0 #remove wall
            maz[neigh[1], neigh[2], dir_map[a]] = 0.0f0 #remove reverse wall
            maz, visited = walk(maz, neigh, msize, visited, wrap = wrap) #
        end
    end
    return maz, visited
end

function maze(msize; wrap = true)
    dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    dir_map = Dict(1 => 2, 2 => 1, 3 => 4, 4 => 3)
    maz = ones(Float32, msize, msize, 4) #start with walls everywhere
    cell = rand(1:msize, 2) #where do we start?
    maz, visited = walk(maz, cell, msize, wrap = wrap) #walk through maze

    # remove a couple of additional walls to increase degeneracy
    if wrap
        holes = Int(3 * (msize - 3)) #3 for Larena=4, 6 for Larena = 5
    else
        holes = Int(4 * (msize - 3)) #4 for Larena=4, 8 for Larena = 5
        # note permanent walls
        maz[msize, :, 1] .= 0.5f0; maz[1, :, 2] .= 0.5f0
        maz[:, msize, 3] .= 0.5f0; maz[:, 1, 4] .= 0.5f0
    end
    for _ in 1:holes
        walls = findall(maz .== 1)
        wall = rand(walls)
        cell, a = [wall[1]; wall[2]], wall[3]

        neigh = neighbor([cell[1]; cell[2]], dirs[a], msize)
        maz[cell[1], cell[2], a] = 0.0f0 #remove wall
        maz[neigh[1], neigh[2], dir_map[a]] = 0.0f0 #remove reverse wall
    end
    maz[maz .== 0.5] .= 1f0 # reinstate permanent walls

    maz = reshape(permutedims(maz, [2, 1, 3]), prod(size(maz)[1:2]), 4)

    return Float32.(maz)
end
