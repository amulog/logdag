# coding: utf-8

"""
Use CausalDiscoveryToolbox
https://github.com/FenTechSolutions/CausalDiscoveryToolbox
"""


def estimate(data, category, algorithm, max_iter=2000, tolerance=0.0001,
             use_deconvolution=True, deconvolution_algorithm="aracne",
             init_graph=None):
    if init_graph is not None:
        raise Warning("init_graph is not available on cdt now")

    if category == "independence":
        return independence_graph(data, algorithm, max_iter, tolerance,
                                  use_deconvolution,
                                  deconvolution_algorithm)
    elif category == "causality":
        return causality_graph(data, algorithm)
    else:
        raise NotImplementedError


def independence_graph(data, algorithm, max_iter=2000, tolerance=0.0001,
                       use_deconvolution=True,
                       deconvolution_algorithm="aracne"):

    import cdt
    if algorithm == "glasso":
        glasso = cdt.independence.graph.Glasso()
        skeleton = glasso.predict(data, max_iter=max_iter,
                                  tol=tolerance, mode='lars')
    else:
        raise NotImplementedError

    if use_deconvolution:
        skeleton = cdt.utils.graph.remove_indirect_links(
            skeleton, alg=deconvolution_algorithm)

    l_remove = []
    for i, j in skeleton.edges():
        if i == j:
            l_remove.append((i, j))
    skeleton.remove_edges_from(l_remove)
        #weight = skeleton.get_edge_data(i, j)["weight"]
        #skeleton[i][j]["label"] = str(round(weight, 2))

    return skeleton


def causality_graph(data, algorithm):

    import cdt
    if algorithm == "ges":
        model = cdt.causality.graph.GES()
        output_graph = model.predict(data)
    else:
        raise NotImplementedError

    return output_graph

