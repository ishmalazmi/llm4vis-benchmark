# LLM4VIS Benchmarking

This repository contains code for benchmarking various tasks in the field of visualization using the LLM4VIS (Language Models for Visualization) framework. LLM4VIS utilizes large language models to perform tasks such as query generation, visualization generation, captioning, and code generation related to visualization tasks.

## Requirements

- Datasets (NvBench, VisText, CodeGenVega)
- Deep Infra API Token (Mandatory)
- OpenAI API Key (Mandatory)
- codeinterpreterapi
- vega
- vega-lite
- vega-embed
- altair
- langchain[all]
- BeautifulSoup
- openai
- pandas

## Setup

1. Clone this repository to your local machine.
2. Make sure you have added the datasets to be used in the right structure
3. For this, you can refer to the llm4vis.Dataset.py Class for better guidance of the right location for datasets.
4. Install all the requirements above and in requirements.txt

## Running the benchmark
Firstly, load the library and handle all the imports neccessary for the proccesses to run smoothly:
```
from llm4vis.LLM import LLM
from llm4vis.Dataset import Dataset
from llm4vis.Task import Task
llm = LLM()
dataset = Dataset()
task = Task()
```
It is crucial to provide only the supported datasets to the tasks as shown in the snippets below.
##### Visualization Query Generation
To start benchmarking an LLM for this task of generating a Vega-Lite schema for a given data and user query, use the following code snippet as an example.
```
llm.benchmark(
    llm.meta_llama_2_70b_chat_hf, 
    dataset.nvBench, 
    batchsize=50, 
    task=task.vizQueryGeneration, 
    deep_infra_api_token="YOUR_DEEP_INFRA_TOKEN",
    openai_key=None 
)
```
##### Visualization Generation
Extending from the previous task, the LLMs are now prompted to generate valid Vega-Lite specs and Altair is used to generate the visualizations.
```
llm.benchmark(
    llm.meta_llama_2_70b_chat_hf, 
    dataset.nvBench, 
    batchsize=50, 
    task=task.vizGeneration, 
    deep_infra_api_token="YOUR_DEEP_INFRA_TOKEN",
    openai_key="YOUR_OPENAI_KEY"
)
```

##### Captioning
It is essential to provide VisText dataset to this task as support for other datasets has not been added yet.
```
llm.benchmark(
    llm=llm.openai_gpt_3_5_turbo,
    dataset=dataset.visText,
    batchsize=50,
    task=task.captioning,
    deep_infra_api_token="YOUR_DEEP_INFRA_TOKEN",
    openai_key="YOUR_OPENAI_KEY"
)
```
##### Code Generation
For this specific task, a custom dataset was curated found [here](https://docs.google.com/spreadsheets/d/1fNwwHaAx5xPYg1WdCE7qOo3pAb1V5FnT1soVp7UZqYE/edit#gid=1370007887). 
```
llm.benchmark(
    llm=llm.mistral_7b_instruct,
    dataset=dataset.codeGenVega,
    batchsize=50,
    task=task.codeGeneration,
    deep_infra_api_token="YOUR_DEEP_INFRA_TOKEN",
    openai_key="YOUR_OPENAI_KEY"
)
```


