#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 4/23/20
# Description: 
# ========================================================

import spacy
import json
import pymedtermino
import pymedtermino.umls as umls
import pymysql
import pymysql.cursors
import networkx as nx
import numpy as np
import pymetamap

from subprocess import Popen, PIPE
from collections import namedtuple
from typing import Dict

pymedtermino.LANGUAGE = "en"
pymedtermino.REMOVE_SUPPRESSED_CONCEPTS = True

Concept = namedtuple("Concept", ["CUI", "name", "semantic_type"])
ConceptNode = namedtuple("ConceptNode", ["concept", "parents", "related_concepts", "associated_sty"])

METAMAP_HOME = "./public_mm/bin/metamap18"
CONCEPT_THRESHOLD = 10.0


class MetaMap(object):
    # Map text to concepts
    def __init__(self, sty_mapping_filename, cmd="metamap18 -y -K --JSONf 4"):
        self._cmd = cmd.split()

        # Map metamap senmantic type abbre to full name
        with open(sty_mapping_filename, 'r') as f:
            lines = f.readlines()

        self.sty_mappings = {}
        for line in lines:
            abbre, tui, sty = line.strip().split('|')
            self.sty_mappings[abbre] = sty

        self.ignore_sty = set(["Qualitative Concept", "Quantitative Concept", "Temporal Concept",
                               "Functional Concept", "Idea or Concept", "Intellectual Product",
                               "Mental Process", "Spatial Concept", "Language"])
        self.buffer_dir = "/tmp/metamap/"

    def __call__(self, text, filename):
        """
        Map input text to concepts
        Args:
            text: Input text
            filename: buffer filename

        Returns:
            concepts: set, a Dict of Concept

        """

        # mm = pymetamap.MetaMap.get_instance(METAMAP_HOME)
        # res, err = mm.extract_concepts([text], word_sense_disambiguation=True)
        #
        # for concept in res:
        #     semantic_type = concept.semtypes[1:-1].split(',')
        #     semantic_type = self.sty_mappings[semantic_type[0]]
        #     if float(concept.score) > CONCEPT_THRESHOLD and semantic_type not in self.ignore_sty:
        #         concepts[concept.cui] = Concept(concept.cui, concept.preferred_name, semantic_type)
        input_buffer_filename = self.buffer_dir + filename + '.txt'
        output_buffer_filename = self.buffer_dir + filename + '.json'

        with open(input_buffer_filename, 'w') as f:
            f.write(text)

        cmd = self._cmd + [input_buffer_filename, output_buffer_filename]
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, universal_newlines=True, bufsize=1)
        res, _ = p.communicate()

        with open(output_buffer_filename, 'r') as f:
            res = json.loads(f.read())

        sents = []
        sents_concepts = []
        for doc in res["AllDocuments"][0]["Document"]["Utterances"]:
            sents.append(doc["UttText"])
            concepts = {}

            for phrase in doc["Phrases"]:
                if phrase["Mappings"]:
                    mapping = phrase["Mappings"][0]     # Pick the first mapping
                    for candidate in mapping["MappingCandidates"]:
                        semantic_type = self.sty_mappings[candidate["SemTypes"][0]]

                        # Ignore general types
                        if semantic_type not in self.ignore_sty and candidate["CandidateCUI"] not in concepts:
                            concept = Concept(candidate["CandidateCUI"], candidate["CandidatePreferred"], semantic_type)
                            concepts[candidate["CandidateCUI"]] = concept
            sents_concepts.append(concepts)

        return sents, sents_concepts


class UMLSMetathesaurus(object):
    """ Retrieve Metathesaurus information """
    def __init__(self, db_host, db_user, db_password, db_name="umls"):
        self._db_host = db_host
        self._db_user = db_user
        self._db_password = db_password
        self._db_name = db_name

        umls.connect_to_umls_db(self._db_host, self._db_user, self._db_password, self._db_name, encoding='utf8')
        self.connection = pymysql.connect(self._db_host, self._db_user, self._db_password, db=self._db_name)

    def create_concept_node(self, concepts: Dict[str, Concept]):
        """
        Map concepts to concept nodes
        Args:
            concepts: Dict, where keys are CUIs and values are Concept(s)

        Returns:
            concept_nodes: Dict, keys are the same as input and values are result ConceptNode(s)

        """
        concepts_nodes = {}
        for cui, concept in concepts.items():
            parents = self._get_parents(concept.CUI)
            related_concepts = self._get_related_concepts(concept.CUI)
            associated_sty = self._get_associated_sty(concept.semantic_type)

            node = ConceptNode(concept, parents, related_concepts, associated_sty)
            concepts_nodes[cui] = node

        return concepts_nodes

    def _get_parents(self, cui: str):
        """
        Return parents of input concept
        Args:
            cui: input concept cui

        Returns:
            parents: a set of CUIs of parents

        """
        try:
            term = umls.UMLS_CUI[cui]
        except ValueError as err:
            # print(err)
            return set()

        if "RN" in term.relations:
            parents = [t.code for t in term.RN]
        else:
            parents = []

        return set(parents)

    def _get_related_concepts(self, cui: str):
        """
        Return concepts have a 'related_to' relation with input concept
        Args:
            cui: input concept CUI

        Returns:
            related_concepts; set of CUIs

        """
        try:
            term = umls.UMLS_CUI[cui]
        except ValueError as err:
            # print(err)
            return set()

        if "related_to" in term.relations:
            related_concepts = [t.code for t in term.related_to]
        else:
            related_concepts = []

        return set(related_concepts)

    def _get_associated_sty(self, semantic_type: str):
        """
        Return semantic types with a 'associated_with' relation with semantic type of input concept
        Args:
            concept: input concept

        Returns:
            ass_sty: set of associated semantic types

        """
        sql = """SELECT STY2 FROM SRSTRE2
                 WHERE STY1='{semantic_type}' AND RL='associated_with'""".format(semantic_type=semantic_type)

        with self.connection.cursor() as cursor:
            cursor.execute(sql)

        data = cursor.fetchall()
        ass_sty = set([sty[0] for sty in data])

        return ass_sty

    def create_sentence_representation(self, concept_nodes: Dict[str, ConceptNode]):
        """
        Given a set of concept nodes, construct a graph to represent its hypernymy relations
        Args:
            concept_nodes: input concept nodes

        Returns:
            G: the hypernymy graph

        """
        G = nx.Graph()

        for cui, node in concept_nodes.items():
            self._hypernymy_dfs(G, cui, set(), concept_nodes)

        return G

    def create_doc_representation(self, concept_nodes: Dict[str, ConceptNode]):
        """
        Create graph-based document representation
        Args:
            concept_nodes: input concept nodes

        Returns:
            G: document representation graph

        """
        G = self.create_sentence_representation(concept_nodes)
        G = self._add_associated_relations(G, concept_nodes)
        G = self._add_related_to_relations(G, concept_nodes)

        return G

    def _hypernymy_dfs(self, G: nx.Graph, node_cui: str, visited: set, leaf: Dict[str, ConceptNode]) -> int:
        """ Depth first search helper function """
        visited.add(node_cui)
        parents = self._get_parents(node_cui)
        parents = [p for p in parents if p not in visited]
        parent_depth = []

        if parents:
            for parent in parents:
                parent_depth.append(self._hypernymy_dfs(G, parent, visited, leaf))

            current_depth = np.max(parent_depth) + 1

            # Add node to graph
            if current_depth > 0 or node_cui in leaf:
                if node_cui in leaf:
                    attr = "leaf"
                else:
                    attr = "hypernymy"

                G.add_node(node_cui, type=attr)

                # Create edge to hypernymy
                for p_depth, parent in zip(parent_depth, parents):
                    if parent in G and p_depth > 0:
                        G.add_edge(node_cui, parent, weight=p_depth/(p_depth+1), relation="hypernymy")

            return current_depth
        else:
            return -1  # define root depth as -1, ignore the top-2 node

    def _add_related_to_relations(self, G: nx.Graph, concept_nodes: Dict[str, ConceptNode]):
        """ Helper function to add 'related_to' relation to the graph """
        for node_cui, concept in concept_nodes.items():
            if node_cui in G:
                for related_concept_cui, related_concept in concept_nodes.items():
                    if related_concept_cui != node_cui \
                            and related_concept_cui in concept.related_concepts \
                            and related_concept_cui in G:
                        G.add_edge(node_cui, related_concept_cui, weight=1.0, relation="related")

        return G

    def _add_associated_relations(self, G: nx.Graph, concept_nodes: Dict[str, ConceptNode]):
        """ Helper function to add 'associated_with' relations to the graph """
        for node_cui, concept in concept_nodes.items():
            if node_cui in G:
                for ass_node_cui, ass_concept in concept_nodes.items():
                    if node_cui != ass_node_cui \
                            and ass_concept.concept.semantic_type in concept.associated_sty \
                            and ass_node_cui in G:
                        G.add_edge(node_cui, ass_node_cui, weight=1.0, relation="associated")

        return G

