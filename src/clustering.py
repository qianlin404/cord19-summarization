#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 4/28/20
# Description: 
# ========================================================

import numpy as np
import networkx as nx

from typing import Dict, List
from collections import namedtuple

ConceptCluster = namedtuple("ConceptCluster", ["HVS", "non_HVS", "num_concepts"])


class ConceptClustering(object):
    """ Retrieve concept cluster """
    def __init__(self, num_salience):
        """
        Set up parameters
        Args:
            num_salience: the number of salience concepts
        """
        self.num_salience = num_salience

    def get_salience_vertices(self, G: nx.Graph):
        """
        Compute the salience of vertices and return the highest n nodes
        Args:
            G: Input graph

        Returns:
            saliences: List of node

        """
        for cui, prop in G.nodes.data():
            salience = 0
            for cui2 in G[cui]:
                salience += G[cui][cui2]["weight"]
            prop["salience"] = salience

        node_cui = np.array([cui for cui, _ in G.nodes.data()])
        node_salience = np.array([prop["salience"] for _, prop in G.nodes.data()])

        sorted_indexes = np.argsort(node_salience)[::-1]
        sorted_indexes = sorted_indexes[:self.num_salience]

        return node_cui[sorted_indexes]

    def get_hvs(self, G: nx.Graph, salience_vertices: np.array):
        """
        Get hub vertice sets
        Args:
            G: input graph
            salience_vertices: salience vertices

        Returns:
            hvs: list of hvs

        """
        # First get most closely connected pairs of hub vertices
        hv_pairs = set()

        for cui1 in salience_vertices:
            max_weight = 0
            connected_node = None
            for cui2 in salience_vertices:
                if cui2 != cui1 and (cui1, cui2) in G.edges and G[cui1][cui2]["weight"] > max_weight:
                    connected_node = cui2

            if connected_node:
                pair = tuple(sorted([cui1, connected_node]))
                hv_pairs.add(pair)
            else:
                hv_pairs.add((cui1,))

        # Merge HVS
        parents = {hvs: hvs for hvs in hv_pairs}
        for i in hv_pairs:
            for j in hv_pairs:
                hvs1 = find_root(i, parents)
                hvs2 = find_root(j, parents)

                if hvs1 != hvs2:  # when they are not the same
                    hvs1_intra_conn = self._get_intra_connectivity(G, hvs1)
                    hvs2_intra_conn = self._get_intra_connectivity(G, hvs2)
                    intra_score = min(hvs1_intra_conn, hvs2_intra_conn)
                    inter_score = self._get_inter_connectivity(G, hvs1, hvs2)

                    # Merge if inter connectivity is higher than intra connectivity
                    if intra_score < inter_score:
                        new_hvs = set(hvs1).union(set(hvs2))
                        new_hvs = tuple(sorted(list(new_hvs)))

                        parents[new_hvs] = new_hvs
                        parents[hvs1] = new_hvs
                        parents[hvs2] = new_hvs

        # Find merged sets
        hvs = [k for k, v in parents.items() if k == v]
        return hvs

    def assign_cluster(self, G: nx.Graph, hvs: List[tuple]):
        """
        assign each node in the graph to a cluster
        Args:
            G: input graph
            hvs: cluster centroids

        Returns:
            concept_cluster: List of ConceptCluser

        """
        concept_cluster = []
        graph_nodes = G.nodes.data()
        # assign centriod nodes
        for i, cluster in enumerate(hvs):
            for node in cluster:
                graph_nodes[node]["cluster"] = i

        for node in G.nodes:
            if "cluster" not in graph_nodes[node]:
                scores = [0] * len(hvs)
                for i, cluster in enumerate(hvs):
                    for hv in cluster:
                        if (node, hv) in G.edges:
                            scores[i] += G[node][hv]["weight"]

                if np.max(scores) > 0:
                    graph_nodes[node]["cluster"] = np.argmax(scores)
                else:
                    graph_nodes[node]["cluster"] = None

        # retrieve cluster
        for i, cluster in enumerate(hvs):
            centroid = set(cluster)
            all_nodes = set([node for node, prop in graph_nodes if prop["cluster"] == i])
            non_centroid =  all_nodes - centroid
            concept_cluster.append(ConceptCluster(centroid, non_centroid, len(all_nodes)))

        return concept_cluster

    def _get_intra_connectivity(self, G: nx.Graph, hvs: tuple):
        """
        Compute intra connectivity
        Args:
            G: input graph
            hvs: hub vertices set

        Returns:
            conn: real number

        """
        nodes = list(hvs)

        conn = 0
        for i in range(len(nodes)-1):
            for j in range(i+1, len(nodes)):
                if (nodes[i], nodes[j]) in G.edges:
                    conn += G[nodes[i]][nodes[j]]["weight"]

        return conn

    def _get_inter_connectivity(self, G: nx.Graph, hvs1: tuple, hvs2: tuple):
        """
        Compute inter connectivity between two hsv
        Args:
            G: input graph
            hvs1: first hsv
            hvs2: second hsv

        Returns:
            conn: real number

        """
        conn = 0

        for node1 in hvs1:
            for node2 in hvs2:
                if (node1, node2) in G.edges:
                    conn += G[node1][node2]["weight"]

        return conn


def sentence_cluster_similarity(sentence_nodes: List[str], cluster_nodes: ConceptCluster):
    """
    Compute similarity score given sentence and cluster
    Args:
        sentence_nodes: List of nodes in sentence graph
        cluster_nodes: List of nodes in cluster graph

    Returns:
        score: number

    """
    score = 0.0
    for node in sentence_nodes:
        if node in cluster_nodes.HVS:
            score += 1
        elif node in cluster_nodes.non_HVS:
            score += 0.5

    return score


def find_root(target: tuple, parents: Dict[tuple, tuple]):
    """ disjoint set find """
    assert target in parents
    t = target
    while t != parents[t]:
        parents[t] = parents[parents[t]]
        t = parents[t]

    return t

