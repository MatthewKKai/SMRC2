import pandas as pd
import numpy as np
import json
import scispacy, spacy
import pubmed_parser as pp

def get_pmid():

    return pmid

def get_abs():

    return abstraction

def get_intro():

    return intro

def get_citanace():

    return citance

def get_citation():

    return citation

def dump_data(address):
    pmid = get_pmid()
    abstraction = get_abs()
    introduction = get_intro()
    citance = get_citanace()
    citation = get_citation()
    data = dict()
    with open(address, "w") as f:
        json.dump(data, address)
    return data

