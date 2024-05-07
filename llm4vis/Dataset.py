import pandas as pd
import numpy as np
import os
import re
import json
import openai
import pycode_similar
from bs4 import BeautifulSoup
import subprocess
from difflib import SequenceMatcher

class Dataset:
    batchsize = 5
    datasets_available = [
        "nvBench",
        "Vis30K"
    ]
    nvBench = {
        "name": "nvBench", 
        "path": "./data/raw/NvBench/nvBench_VegaLite/",
        "path_intermediate": "path_to_intermediate_dataset", 
        "path_final": "path_to_final_dataset",
        "path_json" : "./data/raw/NvBench/NVBench.json",
        "schema": {
            "type": "csv",
            "columns": ["nl_query", "x_name", "y_name", "x_data", "y_data"]
        }
    }
    visText = {
        "name": "VisText", 
        "path": "./data/raw/vistext/vistext.csv",
        "path_intermediate": "./data/raw/vistext/vistext.csv", 
        "path_final": "./data/raw/vistext/vistext.csv",
        "path_json" : "",
        "schema": {
            "type": "csv",
            "columns": ["caption_id", "img_id", "scenegraph", "datatable", "caption_L1", "caption_L1L2", "L1_properties"]
        }
    }
    codeGenVega = {
        "name": "CodeGenVega", 
        "path": "./data/raw/codegen_vega/codegen_vega.csv",
        "path_intermediate": "", 
        "path_final": "",
        "path_json" : "",
        "schema": {
            "type": "csv",
            "columns": ["caption_id", "img_id", "scenegraph", "datatable", "caption_L1", "caption_L1L2", "L1_properties"]
        }
    }

    def show_available_datasets(self):
        print("The available Datasets are \n")
        for dataset in self.datasets_available:
            print(dataset)

    def setBatchSize(self, batchsize):
        self.batchsize = batchsize

    def load(self, dataset):
        try:
            df = pd.read_json(dataset['path_json'], orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'ref'}, inplace=True)
            return df
        except FileNotFoundError:
            print("File not found! Please check path of the dataset in llm4vis.Dataset.py")

        

