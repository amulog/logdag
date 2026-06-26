import datetime
import numpy as np
import networkx as nx
import pandas as pd
import itertools
from collections import defaultdict

from . import variable


class Evaluator:

    def __init__(self, n_trials):
        self._n_trials = n_trials

        self._methods = []
        self._method_names = []
        self._score_funcs = []
        self._score_names = []

        self._results = []
        self._scores = None

    def _make_trial(self, model, defaults):
        df = variable.generate_all(model, defaults)
        for method_obj in self._methods:
            func = method_obj["func"]
            data_key = method_obj["data_key"]
            kwargs = method_obj["kwargs"]

            kwargs[data_key] = df
            if "skeleton_key" in method_obj:
                kwargs[method_obj["skeleton_key"]] = model.to_undirected()

            yield func(**kwargs)

    @classmethod
    def _generate_data(cls, model, defaults):
        return variable.generate_all(model, defaults)

    def add_method(self, func, data_key, kwargs=None,
                   skeleton_key=None, name=None):
        """
        Add a causal discovery method for evaluation and its arguments.

        Args:
            func (func): A causal discovery function. It must allow DataFrame data input.
            data_key (str): Argument key of input DataFrame.
            kwargs (dict): Other static arguments for func.
            skeleton_key (str, optional): If given, func uses
                skeleton structure (i.e., edge existense) of the original model.
                Note that it will be a partially LEAKED evaluation.
            name (str, optional): Method name. Used in output DataFrame of score measurements.
        """
        if kwargs is None:
            kwargs = {}
        method_obj = {"func": func,
                      "data_key": data_key,
                      "kwargs": kwargs}
        if skeleton_key:
            method_obj["skeleton_key"] = skeleton_key
        self._methods.append(method_obj)

        if name:
            self._method_names.append(name)
        else:
            self._method_names.append(str(len(self._methods) - 1))

    def add_score(self, func, kwargs=None, name=None):
        if kwargs is None:
            kwargs = {}
        self._score_funcs.append([func, kwargs])
        if name:
            self._score_names.append(name)
        else:
            self._score_names.append(str(len(self._score_funcs) - 1))

    def get_graph(self, trial_id, method_id=0, do_label=True,
                  weight_key="weight", label_func=str):
        g = self._results[trial_id][method_id]
        if do_label:
            return labeled_graph(g, weight_key, label_func)
        else:
            return g

    def _score_df(self, method_id):
        return pd.DataFrame(self._scores[method_id],
                            columns=self._score_names,
                            index=range(self._n_trials))

    def _calculate_scores(self, models):
        self._scores = defaultdict(list)
        for model, result in zip(models, self._results):
            for method_id, g in enumerate(result):
                l_sc = [sc_func(model, g, **sc_kwargs)
                        for sc_func, sc_kwargs in self._score_funcs]
                self._scores[method_id].append(l_sc)

    def _average_scores(self):
        table_src = []
        for method_id, method_scores in self._scores.items():
            l_trial_scores = [[] for _ in range(len(self._score_funcs))]
            for l_sc in method_scores:
                for sid, sc in enumerate(l_sc):
                    if not np.isnan(sc):
                        l_trial_scores[sid].append(sc)
            avg_scores = np.array([np.average(trial_scores)
                                   for trial_scores in l_trial_scores])
            table_src.append(avg_scores)

        return pd.DataFrame(table_src,
                            columns=self._score_names,
                            index=self._method_names)


class GraphEvaluator(Evaluator):

    def __init__(self, model, defaults, n_trials):
        super().__init__(n_trials)
        self._model = model
        self._defaults = defaults

    def make(self):
        for n in range(self._n_trials):
            trial_results = list(self._make_trial(self._model, self._defaults))
            self._results.append(trial_results)

    def generate_data(self):
        return self._generate_data(self._model, self._defaults)

    def get_model(self, label_func=str):
        return visual_model(self._model, self._defaults, label_func)

    def get_scores(self, method_id=0):
        if not self._scores:
            self._calculate_scores([self._model] * self._n_trials)
        return self._score_df(method_id)

    def get_average_scores(self):
        if not self._scores:
            self._calculate_scores([self._model] * self._n_trials)
        return self._average_scores()


class RandomGraphEvaluator(Evaluator):

    def __init__(self, defaults, n_nodes, n_trials,
                 prob_edge_exists=0.5, weight_range=(0.5, 1),
                 node_attributes=None):
        super().__init__(n_trials)
        self._defaults = defaults
        self._n_nodes = n_nodes
        self._prob_edge_exists = prob_edge_exists
        self._weight_range = weight_range
        self._node_attributes = node_attributes

        self._models = []

    def _random_edge_exists(self):
        return np.random.rand() > self._prob_edge_exists

    def _random_weight(self):
        wmin, wmax = self._weight_range
        return wmin + np.random.rand() * (wmax - wmin)

    def _random_graph(self):
        node_list = list(range(self._n_nodes))
        rand_node_list = np.random.permutation(node_list)

        g = nx.DiGraph()
        if self._node_attributes:
            assert len(self._node_attributes) == self._n_nodes
            for node_id, node_attr in enumerate(self._node_attributes):
                g.add_node(node_id, **node_attr)
        else:
            g.add_nodes_from(node_list)

        while g.number_of_edges() == 0:
            for n1, n2 in itertools.combinations(rand_node_list, 2):
                if self._random_edge_exists():
                    g.add_edge(n1, n2, weight=self._random_weight())

        assert nx.is_directed_acyclic_graph(g)
        return g

    def make(self):
        for n in range(self._n_trials):
            model = self._random_graph()
            trial_results = list(self._make_trial(model, self._defaults))
            self._models.append(model)
            self._results.append(trial_results)

    def generate_modeled_data(self):
        model = self._random_graph()
        return model, self._generate_data(model, self._defaults)

    def get_model(self, trial_id, label_func=str):
        return visual_model(self._models[trial_id], self._defaults, label_func)

    def get_scores(self, method_id=0):
        if not self._scores:
            self._calculate_scores(self._models)
        return self._score_df(method_id)

    def get_average_scores(self):
        if not self._scores:
            self._calculate_scores(self._models)
        return self._average_scores()


def skeleton_accuracy(g1, g2):
    cnt_all = 0
    cnt_correct = 0
    if g1.number_of_edges() == 0:
        return np.nan

    ng1 = g1.to_undirected()
    ng2 = g2.to_undirected()

    nodes = g1.nodes()
    for n1, n2 in itertools.combinations(nodes, 2):
        if ng1.has_edge(n1, n2) == ng2.has_edge(n1, n2):
            cnt_correct += 1
        cnt_all += 1
    return cnt_correct / cnt_all


#def direction_accuracy(g1, g2):
#    cnt_all = 0
#    cnt_correct = 0
#    nodes = g1.nodes()
#    for n1, n2 in itertools.combinations(nodes, 2):
#        n1_n2 = g1.has_edge(n1, n2) == g2.has_edge(n1, n2)
#        n2_n1 = g1.has_edge(n2, n1) == g2.has_edge(n2, n1)
#        if n1_n2 ^ n2_n1:  # XOR
#            cnt_correct += 1
#        cnt_all += 1
#    return cnt_correct / cnt_all


def skeleton_fmeasure(g1, g2):
    from sklearn.metrics import f1_score
    ng1 = g1.to_undirected()
    ng2 = g2.to_undirected()
    nodes = g1.nodes()
    node_pairs = list(itertools.combinations(nodes, 2))
    g1_arr = [ng1.has_edge(n1, n2) for n1, n2 in node_pairs]
    g2_arr = [ng2.has_edge(n1, n2) for n1, n2 in node_pairs]
    return f1_score(g1_arr, g2_arr)


def accurate_direction_ratio(g1, g2):
    ng1 = g1.to_undirected()
    ng2 = g2.to_undirected()

    # ignore bidirected edges in g2
    bidirected_edges = set()
    for n1, n2 in g2.edges():
        if g2.has_edge(n2, n1):
            bidirected_edges.add((n1, n2))
            bidirected_edges.add((n2, n1))

    nodes = g1.nodes()
    cnt_all = 0
    cnt_correct = 0
    for n1, n2 in itertools.combinations(nodes, 2):
        if ng1.has_edge(n1, n2) and ng2.has_edge(n1, n2):
            if (n1, n2) in bidirected_edges:
                pass
            elif g1.has_edge(n1, n2) and g2.has_edge(n1, n2):
                cnt_correct += 1
            elif g1.has_edge(n2, n1) and g2.has_edge(n2, n1):
                cnt_correct += 1
            else:
                pass
            cnt_all += 1

    if cnt_all == 0:
        return np.nan
    else:
        return cnt_correct / cnt_all


def weight_diff(g1, g2, weight_key="weight"):
    diffs = []
    nodes = g1.nodes()
    for n1, n2 in itertools.combinations_with_replacement(nodes, 2):
        if g1.has_edge(n1, n2) and g2.has_edge(n1, n2):
            d1 = g1.get_edge_data(n1, n2)
            d2 = g2.get_edge_data(n1, n2)
            if not (weight_key in d1 and weight_key in d2):
                return np.nan
            diffs.append(np.abs(d1[weight_key] - d2[weight_key]))
    return np.average(diffs)


def labeled_graph(g, weight_key="weight", func=str):
    """Generate labeled DiGraph with edge weights.

    Args:
        g (networkx.DiGraph): Estimated graph.
        weight_key (str, optional): Attribute name of causal weight value.
        func (function, optional): Function to generate label string
            from weight value.

    Returns:
        networkx.DiGraph
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(g.nodes())

    for (edge_src, edge_dst, edge_data) in g.edges(data=True):
        weight = edge_data[weight_key]
        label = func(weight)
        graph.add_edge(edge_src, edge_dst, label=label)

    return graph


def visual_model(g, defaults, func=str):
    """Generate labeled DiGraph of causaltestdata model.

    Args:
        g (networkx.DiGraph): Input graph model.
        defaults (dict): Default values of input model.
        func (function, optional): Function to generate label string
            from weight value.

    Returns:
        networkx.DiGraph
    """

    def _get_spec(data, key, default_value=None):
        if key in data:
            return data[key]
        elif key in defaults:
            return defaults[key]
        else:
            return default_value

    graph = nx.DiGraph()
    graph.add_nodes_from(g.nodes())

    for (edge_src, edge_dst, edge_data) in g.edges(data=True):
        weight = _get_spec(edge_data, "weight")
        delay = _get_spec(edge_data, "delay", datetime.timedelta(0))
        if delay == datetime.timedelta(0):
            label = func(weight)
        else:
            label = "{0} ({1})".format(func(weight), int(delay.total_seconds()))
        graph.add_edge(edge_src, edge_dst, label=label)

    return graph


def output_graph(g, defaults, output):
    graph = visual_model(g, defaults)
    ag = nx.nx_agraph.to_agraph(graph)
    ag.draw(output, prog='circo')
