#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 4/23/20
# Description: 
# ========================================================

import numpy as np
import spacy
import json
import clustering

from umls import MetaMap, UMLSMetathesaurus
from typing import Callable

# MySQL configuration
HOST = "localhost"
USER = "root"
PASSWORD = "mypassword"
DB_NAME = "umls"

SUMMARY_FRAC = 0.3


class Document(object):
    # Read articles from JSON file and perform preprocess
    def __init__(self, filename):
        """
        Read articles from JSON file and remove cite and reference spans
        Args:
            filename: str, name of article file
        """
        nlp = spacy.load("en_core_web_sm")

        with open(filename, 'r') as f:
            doc_json = json.loads(f.read())

        self.id = doc_json["paper_id"]
        self.title = nlp(doc_json["metadata"]["title"])
        self.abstract = [nlp(self._read_text(sentence)) for sentence in doc_json["abstract"]]

        self.body_text = []
        for sentence in doc_json["body_text"]:
            text = nlp(self._read_text(sentence))
            self.body_text.extend(list(text.sents))

        # self.body_text = [nlp(self._read_text(sentence)) for sentence in doc_json["body_text"]]

        sentence_length = [len(sentence) for sentence in self.body_text]
        self.avg_sentence_length = np.mean(sentence_length)
        self.length = np.sum(sentence_length) + len(self.title) + np.sum([len(sentence) for sentence in self.abstract])

        self.sentence_concepts = None
        self.overall_concepts = None
        self.sentence_graph = None
        self.document_graph = None
        self.concept_cluster = None
        self.cluster_setence_scores = None

    def _read_text(self, text_dict):
        """
        Read text from dictionary with cite and reference spans removed
        Args:
            text_dict: Dict, contains text and spans information

        Returns:
            text: pure text with spans removed

        """
        text = list(text_dict["text"])

        for cite in text_dict["cite_spans"]:
            for i in range(cite["start"], cite["end"] + 1):
                text[i] = ''

        for ref in text_dict["ref_spans"]:
            for i in range(ref["start"], ref["end"] + 1):
                text[i] = ''

        text = "".join(text)

        return text

    def summarize_by_main_concept(self):
        """
        Extract salient sentences based on main concept
        Returns:
            salient_sentences: List of sentences

        """
        if not self.concept_cluster:
            print("No concept cluster found")
            return []

        num_sentences = int(self.length * SUMMARY_FRAC/self.avg_sentence_length)

        num_concepts = [c.num_concepts for c in self.concept_cluster]
        main_concept_index = np.argmax(num_concepts)

        salient_sentence_indexes = np.argsort(self.cluster_setence_scores[main_concept_index])[::-1]
        salient_sentence_indexes = salient_sentence_indexes[:num_sentences]

        return [self.body_text[i] for i in salient_sentence_indexes]

    def summarize_by_fraction(self):
        """
        Extract salient sentences with size propotional to cluster size
        Returns:
            salient sentence: List of sentences

        """
        if not self.concept_cluster:
            print("No concept cluster found")
            return []

        num_sentences = int(self.length * SUMMARY_FRAC/self.avg_sentence_length)
        cluster_size = np.array([len(c) for c in self.concept_cluster])
        cluster_size = cluster_size / np.sum(cluster_size)
        cluster_size = np.round(cluster_size * num_sentences).astype(int)

        salient_indexes = set()

        for i, size in enumerate(cluster_size):
            setence_rank = np.argsort(self.cluster_setence_scores[i])[::-1]
            count = 0
            j = 0
            while count < size:
                if setence_rank[j] not in salient_indexes:
                    salient_indexes.add(setence_rank[j])
                    count += 1
                j += 1

        return [self.body_text[i] for i in salient_indexes]

    def summarize_by_scores(self):
        """
        Extract salient sentences by the semantic similarity scores
        Returns:
            salient sentence: List of sentences

        """
        if not self.concept_cluster:
            print("No concept cluster found")
            return []

        num_sentences = int(self.length * SUMMARY_FRAC/self.avg_sentence_length)

        scores = []
        for i in range(len(self.body_text)):
            score = 0
            for j in range(len(self.concept_cluster)):
                cluster_size = len(self.concept_cluster[j])
                score += self.cluster_setence_scores[j][i] / cluster_size

            scores.append(score)

        salient_indexes = np.argsort(scores)[::-1]
        salient_indexes = salient_indexes[:num_sentences]

        return [self.body_text[i] for i in salient_indexes]

    def summarize_random(self):
        """
        Randomly select sentences for summarization
        Returns:
            salient sentence: List of sentences

        """
        num_sentences = int(self.length * SUMMARY_FRAC / self.avg_sentence_length)
        indexes = np.random.permutation(np.arange(len(self.body_text)))[:num_sentences]

        return [self.body_text[i] for i in indexes]

    def rouge(self, salient_sentences: list, n_gram=1):
        """
        Compute n-gram rouge score
        Args:
            salient_sentences: salient sentences
            n_gram: n gram

        Returns:
            rouge_score: number

        """
        eps = 1e-6
        abstract_ngram = set(get_ngram(self.abstract, n_gram))
        salient_ngram = set(get_ngram(salient_sentences, n_gram))

        overlap = abstract_ngram.intersection(salient_ngram)
        return len(overlap) / (len(abstract_ngram) + eps)

    def bleu(self, salient_sentences: list, n_gram=1):
        """
        Compute n-gram bleu score
        Args:
            salient_sentences:  salient sentences
            n_gram: n gram

        Returns:
            bleu_score: number

        """
        eps = 1e-6
        abstract_ngram = set(get_ngram(self.abstract, n_gram))
        salient_ngram = set(get_ngram(salient_sentences, n_gram))

        overlap = abstract_ngram.intersection(salient_ngram)
        return len(overlap) / (len(salient_ngram) + eps)


def get_ngram(sentences: list, n_gram: int=1):
    """
    get ngram of input sentences
    Args:
        sentences: List of sentences
        n_gram: n gram

    Returns:
        res: n gram tokens

    """
    ngram_list = []
    for sentence in sentences:
        for word in sentence:
            if not (word.is_stop or word.is_digit or word.is_punct or not word.is_ascii):
                ngram_list.append(word.text)

    res = [' '.join(ngram_list[i:i+n_gram]) for i in range(len(ngram_list)-n_gram)]
    return res


def summary_document_umls(doc: Document, sty_mapping_filename, hv_fraction=0.1):
    """
    Extractive salient setences from document
    Args:
        doc: input document
        sty_mapping_filename: semantic style mapping filename
        hv_fraction: hub vertices fraction

    Returns:
        doc: assign attributes to origin document

    """

    # Map setence to concepts
    # print("Retrieving concepts from MetaMap...")

    metamap = MetaMap(sty_mapping_filename)
    text = []
    for sent in doc.body_text:
        tokens = [word.text for word in sent if word.is_ascii]
        text.append(' '.join(tokens))

    nlp = spacy.load("en_core_web_sm")

    text = ' '.join(text) + "\n"
    sents, concepts = metamap(text, doc.id)
    doc.body_text = [nlp(s) for s in sents]

    sentence_length = [len(sentence) for sentence in doc.body_text]
    doc.avg_sentence_length = np.mean(sentence_length)

    # concepts = []
    # for i, sentence in enumerate(doc.body_text):
    #     print("Mapping %d/%d sentences" % (i, len(doc.body_text)), end='\r')
    #     text = ' '.join([w.text for w in sentence if not (w.is_stop or w.is_punct or not w.is_ascii)])
    #     concepts.append(metamap(text))


    # Map concepts to concept nodes
    # print("Retrieving concept relations from UMLS Metathesaurus...")
    mm = UMLSMetathesaurus(HOST, USER, PASSWORD, DB_NAME)
    concepts = [mm.create_concept_node(c) for c in concepts]
    concept_union = {}
    for c in concepts:
        concept_union.update(c)

    doc.sentence_concepts = concepts
    doc.overall_concepts = concept_union

    # Create sentence and document graphs
    # print("Creating setence and document graph...")
    sentence_graph = []
    for setence_concept in doc.sentence_concepts:
        sentence_graph.append(mm.create_sentence_representation(setence_concept))

    doc_graph = mm.create_doc_representation(doc.overall_concepts)

    doc.sentence_graph = sentence_graph
    doc.document_graph = doc_graph

    # Clustering
    # print("Clustering concepts...")
    num_salience = int(hv_fraction * len(doc.overall_concepts)) + 1
    cluster = clustering.ConceptClustering(num_salience)
    salience = cluster.get_salience_vertices(doc.document_graph)
    hvs = cluster.get_hvs(doc.document_graph, salience)

    doc.concept_cluster = cluster.assign_cluster(doc.document_graph, hvs)

    cluster_sentence_scores = []
    for c in doc.concept_cluster:
        sentence_scores = []
        for s in doc.sentence_graph:
            sentence_scores.append(clustering.sentence_cluster_similarity(s.nodes, c))

        cluster_sentence_scores.append(sentence_scores)

    doc.cluster_setence_scores = cluster_sentence_scores
    # print("Done")