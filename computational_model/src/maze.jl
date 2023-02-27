function neighbor(cell, dir, msize)
    neigh = ((cell + 1 * dir .+ msize .- 1) .% msize) .+ 1
    return neigh
end

function walk(maz::Array, nxtcell::Vector, msize, visited::Vector=[])
    dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    dir_map = Dict(1 => 2, 2 => 1, 3 => 4, 4 => 3)
    push!(visited, (nxtcell[1] - 1) * msize + nxtcell[2]) #add to list of visited cells
    for idir in randperm(4) #for each neighbor in randomly shuffled list
        neigh = neighbor(nxtcell, dirs[idir], msize) #compute where we end up
        ind = (neigh[1] - 1) * msize + neigh[2]
        if ind ∉ visited #check that we haven't been there
            maz[nxtcell[1], nxtcell[2], idir] = 0.0f0 #remove wall
            maz[neigh[1], neigh[2], dir_map[idir]] = 0.0f0 #remove reverse wall
            maz, visited = walk(maz, neigh, msize, visited) #
        end
    end
    return maz, visited
end

function maze(msize)
    dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    dir_map = Dict(1 => 2, 2 => 1, 3 => 4, 4 => 3)
    maz = ones(Float32, msize, msize, 4) #start with walls everywhere
    cell = rand(1:msize, 2) #where do we start?
    maz, visited = walk(maz, cell, msize) #walk through maze

    # remove a couple of additional walls to increase degeneracy
    holes = Int(3 * (msize - 3)) #3 for Larena=4, 6 for Larena = 5
    for _ in 1:holes
        a = rand(1:4) #action
        indices = findall(maz[:, :, a] .== 1)
        if length(indices) > 0.5 #check for the super unlikely event that there are no walls satisfying this
            cell = rand(indices)
            neigh = neighbor([cell[1]; cell[2]], dirs[a], msize)
            maz[cell[1], cell[2], a] = 0.0f0 #remove wall
            maz[neigh[1], neigh[2], dir_map[a]] = 0.0f0 #remove reverse wall
        end
    end

    maz = reshape(permutedims(maz, [2, 1, 3]), prod(size(maz)[1:2]), 4)

    return Float32.(maz)
end
