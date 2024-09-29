import numpy as np
import multiprocessing as mp

from .core import (
    PartitionData,
    PartitionData_cp,
    RePartitionData,
    PrimitiveElasticGraphEmbedment,
    PrimitiveElasticGraphEmbedment_v2,
    PrimitiveElasticGraphEmbedment_v3,
    PrimitiveElasticGraphEmbedment_cp,
    PrimitiveElasticGraphEmbedment_cp_v2,
    PrimitiveElasticGraphBarycentricEmbedment,
    DecodeElasticMatrix2,
)
from .supervised import (
    gen_pseudotime_augmented_graph,
    gen_pseudotime_augmented_graph_by_path,
    old_gen_pseudotime_augmented_graph_by_path,
)
from .._EMAdjustment import AdjustByConstant


def proxy(Dict):
    return PrimitiveElasticGraphEmbedment(**Dict)


def proxy_cp(Dict):
    return PrimitiveElasticGraphEmbedment_cp(**Dict)


# Grammar function wrapper ------------------------------------------------


def GraphGrammarOperation(
    X,
    NodePositions,
    ElasticMatrix,
    AdjustVect,
    Type,
    partition,
    FixNodesAtPoints=[],
    MaxNumberOfGraphCandidatesDict={
        "AddNode2Node": float("inf"),
        "BisectEdge": float("inf"),
        "RemoveNode": float("inf"),
        "ShrinkEdge": float("inf"),
    },
    PointWeights=None,
):
    if Type == "addnode2node":
        return AddNode2Node(
            X,
            NodePositions,
            ElasticMatrix,
            partition,
            AdjustVect,
            FixNodesAtPoints,
            MaxNumberOfGraphCandidates=MaxNumberOfGraphCandidatesDict["AddNode2Node"],
            PointWeights=PointWeights,
        )
    elif Type == "addnode2node_1":
        return AddNode2Node(
            X,
            NodePositions,
            ElasticMatrix,
            partition,
            AdjustVect,
            FixNodesAtPoints,
            Max_K=1,
            PointWeights=PointWeights,
        )
    elif Type == "addnode2node_2":
        return AddNode2Node(
            X,
            NodePositions,
            ElasticMatrix,
            partition,
            AdjustVect,
            FixNodesAtPoints,
            Max_K=2,
            PointWeights=PointWeights,
        )
    elif Type == "removenode":
        return RemoveNode(NodePositions, ElasticMatrix, AdjustVect, FixNodesAtPoints)
    elif Type == "bisectedge":
        return BisectEdge(
            NodePositions,
            ElasticMatrix,
            AdjustVect,
            MaxNumberOfGraphCandidates=MaxNumberOfGraphCandidatesDict["BisectEdge"],
        )
    elif Type == "bisectedge_3":
        return BisectEdge(NodePositions, ElasticMatrix, AdjustVect, Min_K=3)
    elif Type == "shrinkedge":
        return ShrinkEdge(
            NodePositions,
            ElasticMatrix,
            AdjustVect,
            MaxNumberOfGraphCandidates=MaxNumberOfGraphCandidatesDict["ShrinkEdge"],
        )
    elif Type == "shrinkedge_3":
        return ShrinkEdge(NodePositions, ElasticMatrix, AdjustVect, Min_K=3)
    else:
        raise ValueError("Operation " + Type + " is not defined")


# Grammar functions ------------------------------------------------


def RemoveNode(NodePositions, ElasticMatrix, AdjustVect, FixNodesAtPoints):
    """
    ##  This grammar operation removes a leaf node (connectivity==1)
    """
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    Connectivities = (Lambda > 0).sum(axis=0)

    # Define sizes
    nNodes = ElasticMatrix.shape[0]
    if FixNodesAtPoints != []:
        nGraphs = (Connectivities == 1).sum() - len(FixNodesAtPoints)
        start = len(FixNodesAtPoints)
    else:
        nGraphs = (Connectivities == 1).sum()
        start = 0

    # Preallocate arrays
    NodePositionsArray = [
        np.zeros((nNodes - 1, NodePositions.shape[1])) for i in range(nGraphs)
    ]
    ElasticMatrices = [np.zeros((nNodes - 1, nNodes - 1)) for i in range(nGraphs)]
    NodeIndicesArray = np.zeros((nNodes - 1, nGraphs), dtype=int)
    AdjustVectArray = [[] for i in range(nGraphs)]

    k = 0
    for i in range(start, Connectivities.shape[0]):
        if Connectivities[i] == 1:
            # if terminal node remove it
            newInds = np.concatenate(
                (
                    np.arange(0, i, dtype=int),
                    np.arange(i + 1, nNodes, dtype=int),
                )
            )
            AdjustVectArray[k] = [AdjustVect[j] for j in newInds]
            NodePositionsArray[k] = NodePositions[newInds, :]
            tmp = np.repeat(False, nNodes)
            tmp[newInds] = True
            tmp2 = ElasticMatrix[tmp, :]
            ElasticMatrices[k] = tmp2[:, tmp]
            NodeIndicesArray[:, k] = newInds
            k += 1
    return (
        NodePositionsArray,
        ElasticMatrices,
        AdjustVectArray,
        NodeIndicesArray,
    )


def BisectEdge(
    NodePositions,
    ElasticMatrix,
    AdjustVect,
    Min_K=1,
    MaxNumberOfGraphCandidates=float("inf"),
):
    """
    # % This grammar operation inserts a node inside the middle of each edge
    # % The elasticity of the edges do not change
    # % The elasticity of the newborn star is chosen as
    # % mean over the neighbour stars if the edge connects two star centers
    # % or
    # % the one of the single neigbour star if this is a dangling edge
    # % or
    # % if one starts from a single edge, the star elasticities should be on
    # % one of two elements in the diagonal of the ElasticMatrix
    """
    # Decompose Elastic Matrix: Mus
    Mus = ElasticMatrix.diagonal()
    # Get list of edges
    Edges, _, _ = DecodeElasticMatrix2(ElasticMatrix)

    # Define some constants
    nNodes = NodePositions.shape[0]
    if Min_K > 1:
        Degree = np.bincount(Edges.flatten())
        EdgDegree = np.max(Degree[Edges], axis=1)
        nGraphs = np.where(EdgDegree >= Min_K)[0]
    else:
        nGraphs = np.array(range(Edges.shape[0]))

    # In case we have limits on the number of candidates
    if MaxNumberOfGraphCandidates < len(nGraphs):
        edge_lengths = np.sum(
            (NodePositions[Edges[:, 0], :] - NodePositions[Edges[:, 1], :]).T ** 2,
            axis=0,
        )
        inds = np.argsort(edge_lengths)[::-1]
        if Min_K > 1:
            nGraphs = inds[np.isin(inds, nGraphs)][:MaxNumberOfGraphCandidates]
            Edges = Edges[nGraphs]
        else:
            nGraphs = nGraphs[:MaxNumberOfGraphCandidates]
            Edges = Edges[inds[nGraphs]]

    # Create prototypes for new NodePositions, ElasticMatrix and inds
    npProt = np.vstack((NodePositions, np.zeros((1, NodePositions.shape[1]))))
    emProt = np.vstack(
        (
            np.hstack((ElasticMatrix, np.zeros((nNodes, 1)))),
            np.zeros((1, nNodes + 1)),
        )
    )
    niProt = np.arange(nNodes + 1)
    # niProt[nNodes] = 0

    # Allocate arrays and put prototypes in place
    NodePositionsArray = [npProt.copy() for i in range(len(nGraphs))]
    ElasticMatrices = [emProt.copy() for i in range(len(nGraphs))]
    NodeIndicesArray = np.repeat(niProt[:, np.newaxis], len(nGraphs), axis=1)
    AdjustVectArray = [AdjustVect + [False] for i in range(len(nGraphs))]

    for j, i in enumerate(nGraphs):
        NewNodePosition = (
            NodePositions[Edges[i, 0],] + NodePositions[Edges[i, 1],]
        ) / 2

        # Fill NodePosition
        NodePositionsArray[j][nNodes, :] = NewNodePosition
        # correct elastic matrix
        Lambda = ElasticMatrix[Edges[i, 0], Edges[i, 1]]
        # remove edge
        ElasticMatrices[j][Edges[i, 0], Edges[i, 1]] = 0
        ElasticMatrices[j][Edges[i, 1], Edges[i, 0]] = 0
        # add 2 edges
        ElasticMatrices[j][Edges[i, 0], nNodes] = Lambda
        ElasticMatrices[j][nNodes, Edges[i, 0]] = Lambda
        ElasticMatrices[j][nNodes, Edges[i, 1]] = Lambda
        ElasticMatrices[j][Edges[i, 1], nNodes] = Lambda
        # Define mus of edges
        mu1 = Mus[Edges[i, 0]]
        mu2 = Mus[Edges[i, 1]]
        if mu1 > 0 and mu2 > 0:
            ElasticMatrices[j][nNodes, nNodes] = (mu1 + mu2) / 2
        else:
            ElasticMatrices[j][nNodes, nNodes] = max(mu1, mu2)

    return (
        NodePositionsArray,
        ElasticMatrices,
        AdjustVectArray,
        NodeIndicesArray,
    )


def AddNode2Node(
    X,
    NodePositions,
    ElasticMatrix,
    partition,
    AdjustVect,
    FixNodesAtPoints,
    Max_K=float("inf"),
    MaxNumberOfGraphCandidates=float("inf"),
    PointWeights=None,
):
    """
    Adds a node to each graph node

    This grammar operation adds a node to each graph node. The position of the node
    is chosen as a linear extrapolation for a leaf node (in this case the elasticity of
    a newborn star is chosed as in BisectEdge operation), or as the data point giving
    the minimum local MSE for a star (without any optimization).

    X
    NodePositions
    ElasticMatrix
    @return


    @details



    @examples
    """
    nNodes = NodePositions.shape[0]
    Mus = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    indL = Lambda > 0
    Connectivities = indL.sum(axis=0)
    # add pointweights here if added
    if PointWeights is not None:
        assoc = np.bincount(
            partition[partition > -1].ravel(),
            weights=PointWeights[partition > -1].ravel(),
            minlength=nNodes,
        )
    else:
        assoc = np.bincount(partition[partition > -1].ravel(), minlength=nNodes)

    # Create prototypes for new NodePositions, ElasticMatrix and inds
    npProt = np.vstack((NodePositions, np.zeros((1, NodePositions.shape[1]))))
    emProt = np.vstack(
        (np.hstack((Lambda, np.zeros((nNodes, 1)))), np.zeros((1, nNodes + 1)))
    )
    niProt = np.arange(nNodes + 1, dtype=int)
    #     niProt[nNodes] = 0

    MuProt = np.zeros(nNodes + 1)
    MuProt[:-1] = Mus

    if not np.isinf(Max_K):
        Degree = np.sum(ElasticMatrix > 0, axis=1)
        Degree[Degree > 1] = Degree[Degree > 1] - 1

        if np.sum(Degree <= Max_K) > 1:
            idx_nodes = np.where(Degree <= Max_K)[0]
        else:
            raise ValueError("AddNode2Node impossible with the current parameters!")
    else:
        idx_nodes = np.array(range(nNodes))

    # In case we have limits on the number of candidates
    if MaxNumberOfGraphCandidates < len(idx_nodes) and np.isinf(Max_K):
        idx_nodes = np.argsort(assoc)[::-1][:MaxNumberOfGraphCandidates]

    elif MaxNumberOfGraphCandidates < len(idx_nodes) and not (np.isinf(Max_K)):
        nGraphs = [i for i in np.argsort(assoc)[::-1] if i in idx_nodes]
        idx_nodes = np.array(nGraphs)[:MaxNumberOfGraphCandidates]

    # In case we have fixed nodes
    if FixNodesAtPoints != []:
        idx_nodes = idx_nodes[~np.isin(idx_nodes, np.arange(len(FixNodesAtPoints)))]

    # Put prototypes to corresponding places
    NodePositionsArray = [npProt.copy() for i in range(len(idx_nodes))]
    ElasticMatrices = [emProt.copy() for i in range(len(idx_nodes))]
    NodeIndicesArray = np.repeat(niProt[:, np.newaxis], len(idx_nodes), axis=1)
    AdjustVectArray = [AdjustVect + [False] for i in range(len(idx_nodes))]

    for j, i in enumerate(idx_nodes):
        MuProt[-1] = 0
        # Compute mean edge elasticity for edges with node i
        meanL = Lambda[
            i,
            indL[i,],
        ].mean(axis=0)
        # Add edge to elasticity matrix
        ElasticMatrices[j][nNodes, i] = ElasticMatrices[j][i, nNodes] = meanL

        if Connectivities[i] == 1:
            # Add node to terminal node
            ineighbour = np.nonzero(indL[i,])[0]
            # Calculate new node position
            NewNodePosition = 2 * NodePositions[i,] - NodePositions[ineighbour,]
            # Complete Elasticity Matrix
            MuProt[i] = Mus[ineighbour]
        else:
            # Add node to a star
            # if 0 data points associated with this star
            if assoc[i] == 0:
                # then select mean of all leaves as new position
                NewNodePosition = NodePositions[indL[:, i]].mean(axis=0)

            else:
                # Otherwise take the mean of the points associated with the
                # central node
                NewNodePosition = X[(partition == i).ravel()].mean(axis=0)
        # fill node position
        NodePositionsArray[j][nNodes, :] = NewNodePosition
        np.fill_diagonal(ElasticMatrices[j], MuProt)

    return (
        NodePositionsArray,
        ElasticMatrices,
        AdjustVectArray,
        NodeIndicesArray,
    )


def ShrinkEdge(
    NodePositions,
    ElasticMatrix,
    AdjustVect,
    Min_K=1,
    MaxNumberOfGraphCandidates=float("inf"),
):
    """
    # %
    # % This grammar operation removes an edge from the graph
    # % If this is an edge connecting a leaf node then it is equivalent to
    # % RemoveNode. So we remove only internal edges.
    # % If this is an edge connecting two stars then their leaves are merged,
    # % and the star is placed in the middle of the shrinked edge.
    # % The elasticity of the new formed star is the average of two star
    # % elasticities.
    # %
    """
    ## Shrink edge
    Mus = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    Connectivities = (Lambda > 0).sum(axis=0)
    # get list of edges
    start, stop = np.triu(ElasticMatrix, 1).nonzero()
    # Edges = np.concatenate((start[None], stop[None]))
    # define size
    nNodes = NodePositions.shape[0]
    # identify edges with minimal connectivity > 1
    Degree = np.hstack((Connectivities[start[None].T], Connectivities[stop[None].T]))

    ind_sup1 = np.min(Degree, axis=1) > 1
    ind_min_K = np.max(Degree, axis=1) >= Min_K
    ind = np.where(ind_sup1 & ind_min_K)[0]

    # calculate nb of graphs
    nGraphs = len(ind)

    # In case we have limits on the number of candidates
    if MaxNumberOfGraphCandidates < nGraphs:
        edge_lengths = np.sum(
            (NodePositions[start, :] - NodePositions[stop, :]).T ** 2,
            axis=0,
        )
        nGraphs = MaxNumberOfGraphCandidates
        ind_l = np.argsort(edge_lengths)
        ind_l = ind_l[np.isin(ind_l, ind)][:nGraphs]
        start, stop = start[ind_l], stop[ind_l]
    else:
        start, stop = start[ind], stop[ind]

    # preallocate array
    NodePositionsArray = [
        np.zeros((nNodes - 1, NodePositions.shape[1])) for i in range(nGraphs)
    ]
    ElasticMatrices = [np.zeros((nNodes - 1, nNodes - 1)) for i in range(nGraphs)]
    NodeIndicesArray = np.zeros((nNodes - 1, nGraphs), dtype=int)
    AdjustVectArray = [[] for i in range(nGraphs)]

    for i in range(nGraphs):
        # create copy of elastic matrix
        em = ElasticMatrix.copy()
        # Reattaches all edges connected with stop[i] to start[i]
        # and make a new star with an elasticity average of two merged stars
        em[start[i],] = np.maximum(
            Lambda[start[i],],
            Lambda[stop[i],],
        )
        em[:, start[i]] = np.maximum(Lambda[:, start[i]], Lambda[:, stop[i]])
        em[start[i], start[i]] = (Mus[start[i]] + Mus[stop[i]]) / 2
        #         em[start[i], start[i]] = Mus[start[i]] + Mus[stop[i]] / 2  #### R bug ????
        # Create copy of node positions
        nodep = NodePositions.copy()
        # modify node start[i]
        nodep[start[i], :] = (nodep[start[i], :] + nodep[stop[i], :]) / 2
        # Form index for retained nodes and extract corresponding part of
        # node positions and elastic matrix
        newInds = np.concatenate(
            (np.arange(0, stop[i]), np.arange(stop[i] + 1, nNodes))
        )

        AdjustVectArray[i] = [AdjustVect[j] for j in newInds]

        NodePositionsArray[i] = nodep[newInds, :]
        ElasticMatrices[i] = em.take(newInds, axis=0).take(newInds, axis=1)
        NodeIndicesArray[:, i] = newInds

    return (
        NodePositionsArray,
        ElasticMatrices,
        AdjustVectArray,
        NodeIndicesArray,
    )


def ApplyOptimalGraphGrammarOperation(
    X,
    NodePositions,
    ElasticMatrix,
    opTypes,
    AdjustVect=None,
    SquaredX=None,
    verbose=False,
    MaxBlockSize=100000000,
    MaxNumberOfIterations=100,
    eps=0.01,
    TrimmingRadius=float("inf"),
    Mode=1,
    FinalEnergy="Base",
    alpha=1,
    beta=1,
    EmbPointProb=1,
    PointWeights=None,
    AvoidSolitary=False,
    AdjustElasticMatrix=None,
    DisplayWarnings=True,
    n_cores=1,
    MinParOp=20,
    multiproc_shared_variables=None,
    Xcp=None,
    SquaredXcp=None,
    FixNodesAtPoints=[],
    pseudotime=None,
    pseudotimeLambda=0.01,
    MaxNumberOfGraphCandidatesDict={
        "AddNode2Node": float("inf"),
        "BisectEdge": float("inf"),
        "RemoveNode": float("inf"),
        "ShrinkEdge": float("inf"),
    },
):
    """
    # Multiple grammar application --------------------------------------------
    Application of the grammar operation. This in an internal function that should not be used in by the end-user

    X numerical 2D matrix, the n-by-m matrix with the position of n m-dimensional points
    NodePositions numerical 2D matrix, the k-by-m matrix with the position of k m-dimensional points
    ElasticMatrix numerical 2D matrix, the k-by-k elastic matrix
    operationtypes string vector containing the operation to use
    SquaredX rowSums(X^2), if NULL it will be computed
    verbose boolean. Should addition information be displayed
    n.cores integer. How many cores to use. If EnvCl is not NULL, that cliuster setup will be used,
    otherwise a SOCK cluster willbe used
    EnvCl a cluster structure returned, e.g., by makeCluster.
    If a cluster structure is used, all the nodes must be able to access all the variable needed by PrimitiveElasticGraphEmbedment
    MaxNumberOfIterations is an integer number indicating the maximum number of iterations for the EM algorithm
    TrimmingRadius is a real value indicating the trimming radius, a parameter required for robust principal graphs
    (see https://github.com/auranic/Elastic-principal-graphs/wiki/Robust-principal-graphs)
    eps a real number indicating the minimal relative change in the nodenpositions
    to be considered the graph embedded (convergence criteria)
    Mode integer, the energy mode. It can be 1 (difference is computed using the position of the nodes) and
    2 (difference is computed using the changes in elestic energy of the configuraztions)
    FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    beta positive numeric, the value of the beta parameter of the penalized elastic energy
    gamma
    FastSolve boolean, should FastSolve be used when fitting the points to the data?
    AvoidSolitary boolean, should configurations with "solitary nodes", i.e., nodes without associted points be discarded?
    EmbPointProb numeric between 0 and 1. If less than 1 point will be sampled at each iteration. Prob indicate the probability of
    using each points. This is an *experimental* feature, which may helps speeding up the computation if a large number of points is present.
    AdjustVect
    AdjustElasticMatrix
    ...
    MinParOp integer, the minimum number of operations to use parallel computation

    @return

    @examples
    """

    NodePositionsArrayAll = []
    ElasticMatricesAll = []
    AdjustVectAll = []
    NodeIndicesArrayAll = []
    opTypesAll = []
    if Xcp is None:
        partition, _ = PartitionData(
            X, NodePositions, MaxBlockSize, SquaredX, TrimmingRadius
        )
    else:
        partition, _ = PartitionData_cp(
            Xcp, NodePositions, MaxBlockSize, SquaredXcp, TrimmingRadius
        )

    for i in range(len(opTypes)):
        if verbose:
            print(" Operation type : ", opTypes[i])
        (
            NodePositionsArray,
            ElasticMatrices,
            AdjustVectArray,
            NodeIndicesArray,
        ) = GraphGrammarOperation(
            X,
            NodePositions,
            ElasticMatrix,
            AdjustVect,
            opTypes[i],
            partition,
            FixNodesAtPoints,
            MaxNumberOfGraphCandidatesDict,
        )

        #         NodePositionsArrayAll = np.concatenate((NodePositionsArrayAll,
        #                                                NodePositionsArray), axis=2)
        #         ElasticMatricesAll = np.concatenate((ElasticMatricesAll,
        #                                              ElasticMatrices), axis=2)
        #     NodeIndicesArrayAll = np.concatenate((NodeIndicesArrayAll,
        #                                           NodeIndicesArray), axis=1)
        NodePositionsArrayAll.extend(NodePositionsArray)
        ElasticMatricesAll.extend(ElasticMatrices)
        AdjustVectAll.extend(AdjustVectArray)
        NodeIndicesArrayAll.extend(NodeIndicesArray.T)
        opTypesAll.extend([opTypes[i]] * len(NodePositionsArray))

    if verbose:
        print("Optimizing graphs")

    Valid_configurations = range(len(NodePositionsArrayAll))

    if AvoidSolitary:
        Valid_configurations = []
        if Xcp is None:
            for i in range(len(NodePositionsArrayAll)):
                partition = PartitionData(
                    X=X,
                    MaxBlockSize=MaxBlockSize,
                    NodePositions=NodePositionsArrayAll[i],
                    SquaredX=SquaredX,
                    TrimmingRadius=TrimmingRadius,
                )[0]
                if all(
                    np.isin(
                        np.array(range(NodePositionsArrayAll[i].shape[0])),
                        partition[partition > -1],
                    )
                ):
                    Valid_configurations.append(i)

        else:
            for i in range(len(NodePositionsArrayAll)):
                partition = PartitionData_cp(
                    Xcp,
                    MaxBlockSize,
                    NodePositionsArrayAll[i],
                    SquaredXcp,
                    TrimmingRadius=TrimmingRadius,
                )[0]
                if all(
                    np.isin(
                        np.array(range(NodePositionsArrayAll[i].shape[0])),
                        partition[partition > -1],
                    )
                ):
                    Valid_configurations.append(i)

        if verbose:
            print(
                len(Valid_configurations),
                "configurations out of ",
                len(NodePositionsArrayAll),
                "used",
            )
        if Valid_configurations == []:
            return "failed operation"
    #     NodePositionArrayAll = NodePositionArrayAll[...,Valid_configurations]
    #     ElasticMatricesAll = ElasticMatricesAll[...,Valid_configurations]
    #     AdjustVectAll = AdjustVectAll[Valid_configurations]

    if AdjustElasticMatrix:
        for i in Valid_configurations:
            ElasticMatricesAll[i], AdjustVectAll[i] = AdjustByConstant(
                ElasticMatricesAll[i], AdjustVectAll[i]
            )

    if n_cores > 1 and len(Valid_configurations) // (MinParOp + 1) > 1:
        #             X_remote, X_shape, SquaredX_remote, SquaredX_shape = multiproc_shared_variables
        #             ####### Multiprocessing
        #             with Pool(processes=n_cores,
        #                       initializer=init_worker,
        #                       initargs=(X_remote,
        #                                 X_shape,
        #                                 SquaredX_remote,
        #                                 SquaredX_shape,
        #                                 NodePositionsArrayAll,
        #                                 ElasticMatricesAll,
        #                                 MaxNumberOfIterations,
        #                                 TrimmingRadius,
        #                                 eps,
        #                                 Mode,
        #                                 FinalEnergy,
        #                                 alpha,
        #                                 beta,
        #                                 EmbPointProb,
        #                                 DisplayWarnings,
        #                                 None,
        #                                 MaxBlockSize,
        #                                 False)) as pool:

        #                 results = pool.map(proxy_multiproc,Valid_configurations)

        with mp.Pool(n_cores) as pool:
            if Xcp is None:
                results = pool.map(
                    proxy,
                    [
                        dict(
                            X=X,
                            NodePositions=NodePositionsArrayAll[i],
                            ElasticMatrix=ElasticMatricesAll[i],
                            MaxNumberOfIterations=MaxNumberOfIterations,
                            eps=eps,
                            Mode=Mode,
                            FinalEnergy=FinalEnergy,
                            alpha=alpha,
                            beta=beta,
                            prob=EmbPointProb,
                            DisplayWarnings=DisplayWarnings,
                            PointWeights=PointWeights,
                            MaxBlockSize=MaxBlockSize,
                            verbose=False,
                            TrimmingRadius=TrimmingRadius,
                            SquaredX=SquaredX,
                        )
                        for i in Valid_configurations
                    ],
                )
            else:
                results = pool.map(
                    proxy_cp,
                    [
                        dict(
                            X=X,
                            NodePositions=NodePositionsArrayAll[i],
                            ElasticMatrix=ElasticMatricesAll[i],
                            MaxNumberOfIterations=MaxNumberOfIterations,
                            eps=eps,
                            Mode=Mode,
                            FinalEnergy=FinalEnergy,
                            alpha=alpha,
                            beta=beta,
                            prob=EmbPointProb,
                            DisplayWarnings=DisplayWarnings,
                            PointWeights=PointWeights,
                            MaxBlockSize=MaxBlockSize,
                            verbose=False,
                            TrimmingRadius=TrimmingRadius,
                            SquaredX=SquaredX,
                            Xcp=Xcp,
                            SquaredXcp=SquaredXcp,
                        )
                        for i in Valid_configurations
                    ],
                )

        list_energies = [r[1] for r in results]
        idx = list_energies.index(min(list_energies))
        NewNodePositions, minEnergy, partition, Dist, MSE, EP, RP = results[idx]
        AdjustVect = AdjustVectAll[idx]
        NewElasticMatrix = ElasticMatricesAll[idx]

    else:
        minEnergy = np.inf
        StoreMeanPseudotime = None
        StoreMergedElasticMatrix = None
        StoreMergedNodePositions = None
        if Xcp is None:
            for i in Valid_configurations:
                # TODO add pointweights ?
                if pseudotime is not None and len(NodePositions) > 10:
                    (
                        MeanPseudotime,
                        MergedNodePositions,
                        MergedElasticMatrix,
                        MergedEdges,
                        nPseudoNodes,
                    ) = gen_pseudotime_augmented_graph_by_path(
                        X,
                        SquaredX,
                        NodePositionsArrayAll[i],
                        ElasticMatricesAll[i],
                        pseudotime,
                        root_node=0,
                        LinkMu=0,
                        LinkLambda=pseudotimeLambda,
                        PointWeights=PointWeights,
                        TrimmingRadius=TrimmingRadius,
                    )

                    FitNodePositions = MergedNodePositions
                    FitElasticMatrix = MergedElasticMatrix
                else:
                    FitNodePositions = NodePositionsArrayAll[i]
                    FitElasticMatrix = ElasticMatricesAll[i]
                    nPseudoNodes = 0
                (
                    nodep,
                    ElasticEnergy,
                    part,
                    dist,
                    mse,
                    ep,
                    rp,
                ) = PrimitiveElasticGraphEmbedment(
                    X,
                    FitNodePositions,
                    FitElasticMatrix,
                    MaxNumberOfIterations,
                    eps,
                    Mode=Mode,
                    FinalEnergy=FinalEnergy,
                    alpha=alpha,
                    beta=beta,
                    prob=EmbPointProb,
                    DisplayWarnings=DisplayWarnings,
                    PointWeights=PointWeights,
                    MaxBlockSize=MaxBlockSize,
                    verbose=False,
                    TrimmingRadius=TrimmingRadius,
                    SquaredX=SquaredX,
                    FixNodesAtPoints=[[] for i in range(nPseudoNodes)]
                    + FixNodesAtPoints,
                )

                if ElasticEnergy < minEnergy:
                    if pseudotime is not None and len(NodePositions) > 10:
                        StoreMeanPseudotime = MeanPseudotime
                        StoreMergedElasticMatrix = MergedElasticMatrix
                        StoreMergedNodePositions = nodep
                        NewNodePositions = nodep[nPseudoNodes:]
                    else:
                        StoreMeanPseudotime = None
                        StoreMergedElasticMatrix = None
                        StoreMergedNodePositions = None
                        NewNodePositions = nodep
                    NewElasticMatrix = ElasticMatricesAll[i]
                    partition = part
                    AdjustVect = AdjustVectAll[i]
                    minEnergy = ElasticEnergy
                    MSE = mse
                    EP = ep
                    RP = rp
                    Dist = dist

        else:
            for i in Valid_configurations:
                # TODO add pointweights ?
                if pseudotime is not None and len(NodePositions) > 10:
                    (
                        MeanPseudotime,
                        MergedNodePositions,
                        MergedElasticMatrix,
                        MergedEdges,
                        nPseudoNodes,
                    ) = gen_pseudotime_augmented_graph_by_path(
                        X,
                        SquaredX,
                        NodePositionsArrayAll[i],
                        ElasticMatricesAll[i],
                        pseudotime,
                        root_node=0,
                        LinkMu=0,
                        LinkLambda=pseudotimeLambda,
                        PointWeights=PointWeights,
                        TrimmingRadius=TrimmingRadius,
                    )

                    FitNodePositions = MergedNodePositions
                    FitElasticMatrix = MergedElasticMatrix
                else:
                    FitNodePositions = NodePositionsArrayAll[i]
                    FitElasticMatrix = ElasticMatricesAll[i]
                    nPseudoNodes = 0
                (
                    nodep,
                    ElasticEnergy,
                    part,
                    dist,
                    mse,
                    ep,
                    rp,
                ) = PrimitiveElasticGraphEmbedment_cp(
                    X,
                    FitNodePositions,
                    FitElasticMatrix,
                    MaxNumberOfIterations,
                    eps,
                    Mode=Mode,
                    FinalEnergy=FinalEnergy,
                    alpha=alpha,
                    beta=beta,
                    prob=EmbPointProb,
                    DisplayWarnings=DisplayWarnings,
                    PointWeights=PointWeights,
                    MaxBlockSize=MaxBlockSize,
                    verbose=False,
                    TrimmingRadius=TrimmingRadius,
                    SquaredX=SquaredX,
                    Xcp=Xcp,
                    SquaredXcp=SquaredXcp,
                    FixNodesAtPoints=[[] for i in range(nPseudoNodes)]
                    + FixNodesAtPoints,
                )

                if ElasticEnergy < minEnergy:
                    if pseudotime is not None and len(NodePositions) > 10:
                        StoreMeanPseudotime = MeanPseudotime
                        StoreMergedElasticMatrix = MergedElasticMatrix
                        StoreMergedNodePositions = nodep
                        NewNodePositions = nodep[nPseudoNodes:]
                    else:
                        StoreMeanPseudotime = None
                        StoreMergedElasticMatrix = None
                        StoreMergedNodePositions = None
                        NewNodePositions = nodep
                    NewElasticMatrix = ElasticMatricesAll[i]
                    partition = part
                    AdjustVect = AdjustVectAll[i]
                    minEnergy = ElasticEnergy
                    MSE = mse
                    EP = ep
                    RP = rp
                    Dist = dist

        return dict(
            StoreMeanPseudotime=StoreMeanPseudotime,
            StoreMergedElasticMatrix=StoreMergedElasticMatrix,
            StoreMergedNodePositions=StoreMergedNodePositions,
            NodePositions=NewNodePositions,
            ElasticMatrix=NewElasticMatrix,
            ElasticEnergy=minEnergy,
            MSE=MSE,
            EP=EP,
            RP=RP,
            AdjustVect=AdjustVect,
            Dist=Dist,
        )


def ApplyOptimalGraphGrammarOperation_v2(
    X,
    NodePositions,
    ElasticMatrix,
    opTypes,
    AdjustVect=None,
    SquaredX=None,
    verbose=False,
    MaxBlockSize=100000000,
    MaxNumberOfIterations=100,
    eps=0.01,
    TrimmingRadius=float("inf"),
    Mode=1,
    FinalEnergy="Base",
    alpha=1,
    beta=1,
    EmbPointProb=1,
    PointWeights=None,
    AvoidSolitary=False,
    AdjustElasticMatrix=None,
    DisplayWarnings=True,
    n_cores=1,
    MinParOp=20,
    multiproc_shared_variables=None,
    Xcp=None,
    SquaredXcp=None,
    FixNodesAtPoints=[],
    pseudotime=None,
    pseudotimeLambda=0.01,
    MaxNumberOfGraphCandidatesDict={
        "AddNode2Node": float("inf"),
        "BisectEdge": float("inf"),
        "RemoveNode": float("inf"),
        "ShrinkEdge": float("inf"),
    },
):
    """
    # Multiple grammar application --------------------------------------------
    Application of the grammar operation. This in an internal function that should not be used in by the end-user

    X numerical 2D matrix, the n-by-m matrix with the position of n m-dimensional points
    NodePositions numerical 2D matrix, the k-by-m matrix with the position of k m-dimensional points
    ElasticMatrix numerical 2D matrix, the k-by-k elastic matrix
    operationtypes string vector containing the operation to use
    SquaredX rowSums(X^2), if NULL it will be computed
    verbose boolean. Should addition information be displayed
    n.cores integer. How many cores to use. If EnvCl is not NULL, that cliuster setup will be used,
    otherwise a SOCK cluster willbe used
    EnvCl a cluster structure returned, e.g., by makeCluster.
    If a cluster structure is used, all the nodes must be able to access all the variable needed by PrimitiveElasticGraphEmbedment
    MaxNumberOfIterations is an integer number indicating the maximum number of iterations for the EM algorithm
    TrimmingRadius is a real value indicating the trimming radius, a parameter required for robust principal graphs
    (see https://github.com/auranic/Elastic-principal-graphs/wiki/Robust-principal-graphs)
    eps a real number indicating the minimal relative change in the nodenpositions
    to be considered the graph embedded (convergence criteria)
    Mode integer, the energy mode. It can be 1 (difference is computed using the position of the nodes) and
    2 (difference is computed using the changes in elestic energy of the configuraztions)
    FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    beta positive numeric, the value of the beta parameter of the penalized elastic energy
    gamma
    FastSolve boolean, should FastSolve be used when fitting the points to the data?
    AvoidSolitary boolean, should configurations with "solitary nodes", i.e., nodes without associted points be discarded?
    EmbPointProb numeric between 0 and 1. If less than 1 point will be sampled at each iteration. Prob indicate the probability of
    using each points. This is an *experimental* feature, which may helps speeding up the computation if a large number of points is present.
    AdjustVect
    AdjustElasticMatrix
    ...
    MinParOp integer, the minimum number of operations to use parallel computation

    @return

    @examples
    """

    NodePositionsArrayAll = []
    ElasticMatricesAll = []
    AdjustVectAll = []
    NodeIndicesArrayAll = []
    opTypesAll = []
    if Xcp is None:
        partition, dists, precomp_d = PartitionData(
            X,
            NodePositions,
            MaxBlockSize,
            SquaredX,
            TrimmingRadius,
            precomp=True,
        )
    else:
        partition, dists, precomp_d = PartitionData_cp(
            Xcp,
            NodePositions,
            MaxBlockSize,
            SquaredXcp,
            TrimmingRadius,
            precomp=True,
        )

    for i in range(len(opTypes)):
        if verbose:
            print(" Operation type : ", opTypes[i])
        (
            NodePositionsArray,
            ElasticMatrices,
            AdjustVectArray,
            NodeIndicesArray,
        ) = GraphGrammarOperation(
            X,
            NodePositions,
            ElasticMatrix,
            AdjustVect,
            opTypes[i],
            partition,
            FixNodesAtPoints,
            MaxNumberOfGraphCandidatesDict,
            PointWeights=PointWeights,
        )

        NodePositionsArrayAll.extend(NodePositionsArray)
        ElasticMatricesAll.extend(ElasticMatrices)
        AdjustVectAll.extend(AdjustVectArray)
        NodeIndicesArrayAll.extend(NodeIndicesArray.T)
        opTypesAll.extend([opTypes[i]] * len(NodePositionsArray))

    if verbose:
        print("Optimizing graphs")

    Valid_configurations = range(len(NodePositionsArrayAll))

    if AvoidSolitary:
        Valid_configurations = []
        if Xcp is None:
            for i in range(len(NodePositionsArrayAll)):
                partition = PartitionData(
                    X=X,
                    MaxBlockSize=MaxBlockSize,
                    NodePositions=NodePositionsArrayAll[i],
                    SquaredX=SquaredX,
                    TrimmingRadius=TrimmingRadius,
                )[0]
                if all(
                    np.isin(
                        np.array(range(NodePositionsArrayAll[i].shape[0])),
                        partition[partition > -1],
                    )
                ):
                    Valid_configurations.append(i)

        else:
            for i in range(len(NodePositionsArrayAll)):
                partition = PartitionData_cp(
                    Xcp,
                    MaxBlockSize,
                    NodePositionsArrayAll[i],
                    SquaredXcp,
                    TrimmingRadius=TrimmingRadius,
                )[0]
                if all(
                    np.isin(
                        np.array(range(NodePositionsArrayAll[i].shape[0])),
                        partition[partition > -1],
                    )
                ):
                    Valid_configurations.append(i)

        if verbose:
            print(
                len(Valid_configurations),
                "configurations out of ",
                len(NodePositionsArrayAll),
                "used",
            )
        if Valid_configurations == []:
            return "failed operation"
    #     NodePositionArrayAll = NodePositionArrayAll[...,Valid_configurations]
    #     ElasticMatricesAll = ElasticMatricesAll[...,Valid_configurations]
    #     AdjustVectAll = AdjustVectAll[Valid_configurations]

    if AdjustElasticMatrix:
        for i in Valid_configurations:
            ElasticMatricesAll[i], AdjustVectAll[i] = AdjustByConstant(
                ElasticMatricesAll[i], AdjustVectAll[i]
            )

    if n_cores > 1 and len(Valid_configurations) // (MinParOp + 1) > 1:
        #             X_remote, X_shape, SquaredX_remote, SquaredX_shape = multiproc_shared_variables
        #             ####### Multiprocessing
        #             with Pool(processes=n_cores,
        #                       initializer=init_worker,
        #                       initargs=(X_remote,
        #                                 X_shape,
        #                                 SquaredX_remote,
        #                                 SquaredX_shape,
        #                                 NodePositionsArrayAll,
        #                                 ElasticMatricesAll,
        #                                 MaxNumberOfIterations,
        #                                 TrimmingRadius,
        #                                 eps,
        #                                 Mode,
        #                                 FinalEnergy,
        #                                 alpha,
        #                                 beta,
        #                                 EmbPointProb,
        #                                 DisplayWarnings,
        #                                 None,
        #                                 MaxBlockSize,
        #                                 False)) as pool:

        #                 results = pool.map(proxy_multiproc,Valid_configurations)

        with mp.Pool(n_cores) as pool:
            results = pool.map(
                proxy,
                [
                    dict(
                        X=X,
                        NodePositions=NodePositionsArrayAll[i],
                        ElasticMatrix=ElasticMatricesAll[i],
                        MaxNumberOfIterations=MaxNumberOfIterations,
                        eps=eps,
                        Mode=Mode,
                        FinalEnergy=FinalEnergy,
                        alpha=alpha,
                        beta=beta,
                        prob=EmbPointProb,
                        DisplayWarnings=DisplayWarnings,
                        PointWeights=PointWeights,
                        MaxBlockSize=MaxBlockSize,
                        verbose=False,
                        TrimmingRadius=TrimmingRadius,
                        SquaredX=SquaredX,
                    )
                    for i in Valid_configurations
                ],
            )

        list_energies = [r[1] for r in results]
        idx = list_energies.index(min(list_energies))
        NewNodePositions, minEnergy, partition, Dist, MSE, EP, RP = results[idx]
        AdjustVect = AdjustVectAll[idx]
        NewElasticMatrix = ElasticMatricesAll[idx]

    else:
        minEnergy = np.inf
        StoreMergedElasticMatrix = None
        StoreMergedNodePositions = None
        cache_PseudotimeNodePositions = {}
        for i in Valid_configurations:
            newpartition, newdists, newprecomp_d = RePartitionData(
                X,
                NodePositions,
                NodePositionsArrayAll[i],
                NodeIndicesArrayAll[i],
                opTypesAll[i],
                precomp_d,
                SquaredX,
                TrimmingRadius=np.inf,
            )
            if pseudotime is not None and len(NodePositions) > 10:
                (
                    MergedNodePositions,
                    MergedElasticMatrix,
                    MergedEdges,
                    nPseudoNodes,
                    cache_PseudotimeNodePositions,
                ) = gen_pseudotime_augmented_graph_by_path(
                    X,
                    SquaredX,
                    NodePositionsArrayAll[i],
                    ElasticMatricesAll[i],
                    pseudotime,
                    root_node=0,
                    LinkMu=0,
                    LinkLambda=pseudotimeLambda,
                    PointWeights=PointWeights,
                    TrimmingRadius=TrimmingRadius,
                    partition=newpartition,
                    cache_PseudotimeNodePositions=cache_PseudotimeNodePositions,
                )

                # (
                #    MeanPseudotime,
                #    MergedNodePositions,
                #    MergedElasticMatrix,
                #    MergedEdges,
                #    nPseudoNodes,
                # ) = old_gen_pseudotime_augmented_graph_by_path(
                #    X,
                #    SquaredX,
                #    NodePositionsArrayAll[i],
                #    ElasticMatricesAll[i],
                #    pseudotime,
                #    root_node=0,
                #    LinkMu=0,
                #    LinkLambda=pseudotimeLambda,
                #    PointWeights=PointWeights,
                #    TrimmingRadius=TrimmingRadius,
                #    partition=newpartition,
                # )

                _split = int(len(MergedNodePositions) / 2)
                FitNodePositions = MergedNodePositions[_split:]
                PseudotimeNodePositions = MergedNodePositions[:_split]
                FitElasticMatrix = ElasticMatricesAll[i]

            else:
                FitNodePositions = NodePositionsArrayAll[i]
                FitElasticMatrix = ElasticMatricesAll[i]
                PseudotimeNodePositions = None

            if Xcp is None:
                (
                    nodep,
                    ElasticEnergy,
                    part,
                    dist,
                    mse,
                    ep,
                    rp,
                ) = PrimitiveElasticGraphEmbedment_v2(
                    X,
                    FitNodePositions,
                    FitElasticMatrix,
                    MaxNumberOfIterations,
                    eps,
                    Mode=Mode,
                    FinalEnergy=FinalEnergy,
                    alpha=alpha,
                    beta=beta,
                    prob=EmbPointProb,
                    DisplayWarnings=DisplayWarnings,
                    PointWeights=PointWeights,
                    MaxBlockSize=MaxBlockSize,
                    verbose=False,
                    TrimmingRadius=TrimmingRadius,
                    SquaredX=SquaredX,
                    FixNodesAtPoints=FixNodesAtPoints,
                    PseudotimeNodePositions=PseudotimeNodePositions,
                    PseudotimeLambda=pseudotimeLambda,
                    partition=newpartition,
                    dists=newdists,
                    precomp_d=newprecomp_d,
                )
            else:
                (
                    nodep,
                    ElasticEnergy,
                    part,
                    dist,
                    mse,
                    ep,
                    rp,
                ) = PrimitiveElasticGraphEmbedment_cp_v2(
                    X,
                    FitNodePositions,
                    FitElasticMatrix,
                    MaxNumberOfIterations,
                    eps,
                    Mode=Mode,
                    FinalEnergy=FinalEnergy,
                    alpha=alpha,
                    beta=beta,
                    prob=EmbPointProb,
                    DisplayWarnings=DisplayWarnings,
                    PointWeights=PointWeights,
                    MaxBlockSize=MaxBlockSize,
                    verbose=False,
                    TrimmingRadius=TrimmingRadius,
                    SquaredX=SquaredX,
                    Xcp=Xcp,
                    SquaredXcp=SquaredXcp,
                    FixNodesAtPoints=FixNodesAtPoints,
                    PseudotimeNodePositions=PseudotimeNodePositions,
                    PseudotimeLambda=pseudotimeLambda,
                    partition=newpartition,
                    dists=newdists,
                    precomp_d=newprecomp_d,
                )

            if ElasticEnergy < minEnergy:
                if pseudotime is not None and len(NodePositions) > 10:
                    StoreMergedElasticMatrix = MergedElasticMatrix
                    StoreMergedNodePositions = np.concatenate(
                        (PseudotimeNodePositions, nodep)
                    )
                    NewNodePositions = nodep
                else:
                    StoreMergedElasticMatrix = None
                    StoreMergedNodePositions = None
                    NewNodePositions = nodep
                NewElasticMatrix = ElasticMatricesAll[i]
                partition = part
                AdjustVect = AdjustVectAll[i]
                minEnergy = ElasticEnergy
                MSE = mse
                EP = ep
                RP = rp
                Dist = dist

    # if ~np.isfinite(minEnergy):
    #    return "failed operation"
    return dict(
        StoreMergedElasticMatrix=StoreMergedElasticMatrix,
        StoreMergedNodePositions=StoreMergedNodePositions,
        NodePositions=NewNodePositions,
        ElasticMatrix=NewElasticMatrix,
        ElasticEnergy=minEnergy,
        MSE=MSE,
        EP=EP,
        RP=RP,
        AdjustVect=AdjustVect,
        Dist=Dist,
    )


def ApplyOptimalGraphGrammarOperation_v3(
    X,
    NodePositions,
    ElasticMatrix,
    opTypes,
    AdjustVect=None,
    SquaredX=None,
    verbose=False,
    MaxBlockSize=100000000,
    MaxNumberOfIterations=100,
    eps=0.01,
    TrimmingRadius=float("inf"),
    Mode=1,
    FinalEnergy="Base",
    alpha=1,
    beta=1,
    EmbPointProb=1,
    PointWeights=None,
    AvoidSolitary=False,
    AdjustElasticMatrix=None,
    DisplayWarnings=True,
    n_cores=1,
    MinParOp=20,
    multiproc_shared_variables=None,
    Xcp=None,
    SquaredXcp=None,
    FixNodesAtPoints=[],
    pseudotime=None,
    pseudotimeLambda=0.01,
    label=None,
    labelLambda=0.01,
    MaxNumberOfGraphCandidatesDict={
        "AddNode2Node": float("inf"),
        "BisectEdge": float("inf"),
        "RemoveNode": float("inf"),
        "ShrinkEdge": float("inf"),
    },
):
    """
    # Multiple grammar application --------------------------------------------
    Application of the grammar operation. This in an internal function that should not be used in by the end-user

    X numerical 2D matrix, the n-by-m matrix with the position of n m-dimensional points
    NodePositions numerical 2D matrix, the k-by-m matrix with the position of k m-dimensional points
    ElasticMatrix numerical 2D matrix, the k-by-k elastic matrix
    operationtypes string vector containing the operation to use
    SquaredX rowSums(X^2), if NULL it will be computed
    verbose boolean. Should addition information be displayed
    n.cores integer. How many cores to use. If EnvCl is not NULL, that cliuster setup will be used,
    otherwise a SOCK cluster willbe used
    EnvCl a cluster structure returned, e.g., by makeCluster.
    If a cluster structure is used, all the nodes must be able to access all the variable needed by PrimitiveElasticGraphEmbedment
    MaxNumberOfIterations is an integer number indicating the maximum number of iterations for the EM algorithm
    TrimmingRadius is a real value indicating the trimming radius, a parameter required for robust principal graphs
    (see https://github.com/auranic/Elastic-principal-graphs/wiki/Robust-principal-graphs)
    eps a real number indicating the minimal relative change in the nodenpositions
    to be considered the graph embedded (convergence criteria)
    Mode integer, the energy mode. It can be 1 (difference is computed using the position of the nodes) and
    2 (difference is computed using the changes in elestic energy of the configuraztions)
    FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    beta positive numeric, the value of the beta parameter of the penalized elastic energy
    gamma
    FastSolve boolean, should FastSolve be used when fitting the points to the data?
    AvoidSolitary boolean, should configurations with "solitary nodes", i.e., nodes without associted points be discarded?
    EmbPointProb numeric between 0 and 1. If less than 1 point will be sampled at each iteration. Prob indicate the probability of
    using each points. This is an *experimental* feature, which may helps speeding up the computation if a large number of points is present.
    AdjustVect
    AdjustElasticMatrix
    ...
    MinParOp integer, the minimum number of operations to use parallel computation

    @return

    @examples
    """

    NodePositionsArrayAll = []
    ElasticMatricesAll = []
    AdjustVectAll = []
    NodeIndicesArrayAll = []
    opTypesAll = []

    if Xcp is None:
        partition, _ = PartitionData(
            X, NodePositions, MaxBlockSize, SquaredX, TrimmingRadius
        )
    else:
        partition, _ = PartitionData_cp(
            Xcp, NodePositions, MaxBlockSize, SquaredXcp, TrimmingRadius
        )

    for i in range(len(opTypes)):
        if verbose:
            print(" Operation type : ", opTypes[i])
        (
            NodePositionsArray,
            ElasticMatrices,
            AdjustVectArray,
            NodeIndicesArray,
        ) = GraphGrammarOperation(
            X,
            NodePositions,
            ElasticMatrix,
            AdjustVect,
            opTypes[i],
            partition,
            FixNodesAtPoints,
            MaxNumberOfGraphCandidatesDict,
        )

        NodePositionsArrayAll.extend(NodePositionsArray)
        ElasticMatricesAll.extend(ElasticMatrices)
        AdjustVectAll.extend(AdjustVectArray)
        NodeIndicesArrayAll.extend(NodeIndicesArray.T)
        opTypesAll.extend([opTypes[i]] * len(NodePositionsArray))

    if verbose:
        print("Optimizing graphs")

    Valid_configurations = range(len(NodePositionsArrayAll))

    if AvoidSolitary:
        Valid_configurations = []
        if Xcp is None:
            for i in range(len(NodePositionsArrayAll)):
                partition = PartitionData(
                    X=X,
                    MaxBlockSize=MaxBlockSize,
                    NodePositions=NodePositionsArrayAll[i],
                    SquaredX=SquaredX,
                    TrimmingRadius=TrimmingRadius,
                )[0]
                if all(
                    np.isin(
                        np.array(range(NodePositionsArrayAll[i].shape[0])),
                        partition[partition > -1],
                    )
                ):
                    Valid_configurations.append(i)

        else:
            for i in range(len(NodePositionsArrayAll)):
                partition = PartitionData_cp(
                    Xcp,
                    MaxBlockSize,
                    NodePositionsArrayAll[i],
                    SquaredXcp,
                    TrimmingRadius=TrimmingRadius,
                )[0]
                if all(
                    np.isin(
                        np.array(range(NodePositionsArrayAll[i].shape[0])),
                        partition[partition > -1],
                    )
                ):
                    Valid_configurations.append(i)

        if verbose:
            print(
                len(Valid_configurations),
                "configurations out of ",
                len(NodePositionsArrayAll),
                "used",
            )
        if Valid_configurations == []:
            return "failed operation"
    #     NodePositionArrayAll = NodePositionArrayAll[...,Valid_configurations]
    #     ElasticMatricesAll = ElasticMatricesAll[...,Valid_configurations]
    #     AdjustVectAll = AdjustVectAll[Valid_configurations]

    if AdjustElasticMatrix:
        for i in Valid_configurations:
            ElasticMatricesAll[i], AdjustVectAll[i] = AdjustByConstant(
                ElasticMatricesAll[i], AdjustVectAll[i]
            )

    if n_cores > 1 and len(Valid_configurations) // (MinParOp + 1) > 1:
        #             X_remote, X_shape, SquaredX_remote, SquaredX_shape = multiproc_shared_variables
        #             ####### Multiprocessing
        #             with Pool(processes=n_cores,
        #                       initializer=init_worker,
        #                       initargs=(X_remote,
        #                                 X_shape,
        #                                 SquaredX_remote,
        #                                 SquaredX_shape,
        #                                 NodePositionsArrayAll,
        #                                 ElasticMatricesAll,
        #                                 MaxNumberOfIterations,
        #                                 TrimmingRadius,
        #                                 eps,
        #                                 Mode,
        #                                 FinalEnergy,
        #                                 alpha,
        #                                 beta,
        #                                 EmbPointProb,
        #                                 DisplayWarnings,
        #                                 None,
        #                                 MaxBlockSize,
        #                                 False)) as pool:

        #                 results = pool.map(proxy_multiproc,Valid_configurations)

        with mp.Pool(n_cores) as pool:
            if Xcp is None:
                results = pool.map(
                    proxy,
                    [
                        dict(
                            X=X,
                            NodePositions=NodePositionsArrayAll[i],
                            ElasticMatrix=ElasticMatricesAll[i],
                            MaxNumberOfIterations=MaxNumberOfIterations,
                            eps=eps,
                            Mode=Mode,
                            FinalEnergy=FinalEnergy,
                            alpha=alpha,
                            beta=beta,
                            prob=EmbPointProb,
                            DisplayWarnings=DisplayWarnings,
                            PointWeights=PointWeights,
                            MaxBlockSize=MaxBlockSize,
                            verbose=False,
                            TrimmingRadius=TrimmingRadius,
                            SquaredX=SquaredX,
                        )
                        for i in Valid_configurations
                    ],
                )
            else:
                results = pool.map(
                    proxy_cp,
                    [
                        dict(
                            X=X,
                            NodePositions=NodePositionsArrayAll[i],
                            ElasticMatrix=ElasticMatricesAll[i],
                            MaxNumberOfIterations=MaxNumberOfIterations,
                            eps=eps,
                            Mode=Mode,
                            FinalEnergy=FinalEnergy,
                            alpha=alpha,
                            beta=beta,
                            prob=EmbPointProb,
                            DisplayWarnings=DisplayWarnings,
                            PointWeights=PointWeights,
                            MaxBlockSize=MaxBlockSize,
                            verbose=False,
                            TrimmingRadius=TrimmingRadius,
                            SquaredX=SquaredX,
                            Xcp=Xcp,
                            SquaredXcp=SquaredXcp,
                        )
                        for i in Valid_configurations
                    ],
                )

        list_energies = [r[1] for r in results]
        idx = list_energies.index(min(list_energies))
        NewNodePositions, minEnergy, partition, Dist, MSE, EP, RP = results[idx]
        AdjustVect = AdjustVectAll[idx]
        NewElasticMatrix = ElasticMatricesAll[idx]

    else:
        minEnergy = np.inf
        StoreMeanPseudotime = None
        StoreMergedElasticMatrix = None
        StoreMergedNodePositions = None
        if Xcp is None:
            for i in Valid_configurations:
                # TODO add pointweights ?
                if pseudotime is not None and len(NodePositions) > 10:
                    (
                        MeanPseudotime,
                        MergedNodePositions,
                        MergedElasticMatrix,
                        MergedEdges,
                        nPseudoNodes,
                    ) = gen_pseudotime_augmented_graph_by_path(
                        X,
                        SquaredX,
                        NodePositionsArrayAll[i],
                        ElasticMatricesAll[i],
                        pseudotime,
                        root_node=0,
                        LinkMu=0,
                        LinkLambda=pseudotimeLambda,
                        PointWeights=PointWeights,
                        TrimmingRadius=TrimmingRadius,
                    )

                    _split = int(len(MergedNodePositions) / 2)
                    FitNodePositions = MergedNodePositions[_split:]
                    PseudotimeNodePositions = MergedNodePositions[:_split]
                    FitElasticMatrix = ElasticMatricesAll[i]

                else:
                    FitNodePositions = NodePositionsArrayAll[i]
                    FitElasticMatrix = ElasticMatricesAll[i]
                    PseudotimeNodePositions = None
                    nPseudoNodes = 0
                (
                    nodep,
                    ElasticEnergy,
                    part,
                    dist,
                    mse,
                    ep,
                    rp,
                ) = PrimitiveElasticGraphEmbedment_v2(
                    X,
                    FitNodePositions,
                    FitElasticMatrix,
                    MaxNumberOfIterations,
                    eps,
                    Mode=Mode,
                    FinalEnergy=FinalEnergy,
                    alpha=alpha,
                    beta=beta,
                    prob=EmbPointProb,
                    DisplayWarnings=DisplayWarnings,
                    PointWeights=PointWeights,
                    MaxBlockSize=MaxBlockSize,
                    verbose=False,
                    TrimmingRadius=TrimmingRadius,
                    SquaredX=SquaredX,
                    FixNodesAtPoints=FixNodesAtPoints,
                    PseudotimeNodePositions=PseudotimeNodePositions,
                    PseudotimeLambda=pseudotimeLambda,
                    # Label=label,
                    # LabelLambda=labelLambda,
                )

                if ElasticEnergy < minEnergy:
                    if pseudotime is not None and len(NodePositions) > 10:
                        StoreMeanPseudotime = MeanPseudotime
                        StoreMergedElasticMatrix = MergedElasticMatrix
                        StoreMergedNodePositions = np.concatenate(
                            (PseudotimeNodePositions, nodep)
                        )
                        NewNodePositions = nodep
                    else:
                        StoreMeanPseudotime = None
                        StoreMergedElasticMatrix = None
                        StoreMergedNodePositions = None
                        NewNodePositions = nodep
                    NewElasticMatrix = ElasticMatricesAll[i]
                    partition = part
                    AdjustVect = AdjustVectAll[i]
                    minEnergy = ElasticEnergy
                    MSE = mse
                    EP = ep
                    RP = rp
                    Dist = dist

        else:
            for i in Valid_configurations:
                # TODO add pointweights ?
                if pseudotime is not None and len(NodePositions) > 10:
                    (
                        MeanPseudotime,
                        MergedNodePositions,
                        MergedElasticMatrix,
                        MergedEdges,
                        nPseudoNodes,
                    ) = gen_pseudotime_augmented_graph_by_path(
                        X,
                        SquaredX,
                        NodePositionsArrayAll[i],
                        ElasticMatricesAll[i],
                        pseudotime,
                        root_node=0,
                        LinkMu=0,
                        LinkLambda=pseudotimeLambda,
                        PointWeights=PointWeights,
                        TrimmingRadius=TrimmingRadius,
                    )
                    _split = int(len(MergedNodePositions) / 2)
                    FitNodePositions = MergedNodePositions[_split:]
                    PseudotimeNodePositions = MergedNodePositions[:_split]
                    FitElasticMatrix = ElasticMatricesAll[i]
                else:
                    FitNodePositions = NodePositionsArrayAll[i]
                    FitElasticMatrix = ElasticMatricesAll[i]
                    PseudotimeNodePositions = None
                    nPseudoNodes = 0
                (
                    nodep,
                    ElasticEnergy,
                    part,
                    dist,
                    mse,
                    ep,
                    rp,
                ) = PrimitiveElasticGraphEmbedment_cp_v2(
                    X,
                    FitNodePositions,
                    FitElasticMatrix,
                    MaxNumberOfIterations,
                    eps,
                    Mode=Mode,
                    FinalEnergy=FinalEnergy,
                    alpha=alpha,
                    beta=beta,
                    prob=EmbPointProb,
                    DisplayWarnings=DisplayWarnings,
                    PointWeights=PointWeights,
                    MaxBlockSize=MaxBlockSize,
                    verbose=False,
                    TrimmingRadius=TrimmingRadius,
                    SquaredX=SquaredX,
                    Xcp=Xcp,
                    SquaredXcp=SquaredXcp,
                    FixNodesAtPoints=FixNodesAtPoints,
                    PseudotimeNodePositions=PseudotimeNodePositions,
                    PseudotimeLambda=pseudotimeLambda,
                    # Label=label,
                    # LabelLambda=labelLambda,
                )

                if ElasticEnergy < minEnergy:
                    if pseudotime is not None and len(NodePositions) > 10:
                        StoreMeanPseudotime = MeanPseudotime
                        StoreMergedElasticMatrix = MergedElasticMatrix
                        StoreMergedNodePositions = np.concatenate(
                            (PseudotimeNodePositions, nodep)
                        )
                        NewNodePositions = nodep
                    else:
                        StoreMeanPseudotime = None
                        StoreMergedElasticMatrix = None
                        StoreMergedNodePositions = None
                        NewNodePositions = nodep
                    NewElasticMatrix = ElasticMatricesAll[i]
                    partition = part
                    AdjustVect = AdjustVectAll[i]
                    minEnergy = ElasticEnergy
                    MSE = mse
                    EP = ep
                    RP = rp
                    Dist = dist

    # if ~np.isfinite(minEnergy):
    #    return "failed operation"
    return dict(
        StoreMeanPseudotime=StoreMeanPseudotime,
        StoreMergedElasticMatrix=StoreMergedElasticMatrix,
        StoreMergedNodePositions=StoreMergedNodePositions,
        NodePositions=NewNodePositions,
        ElasticMatrix=NewElasticMatrix,
        ElasticEnergy=minEnergy,
        MSE=MSE,
        EP=EP,
        RP=RP,
        AdjustVect=AdjustVect,
        Dist=Dist,
    )
