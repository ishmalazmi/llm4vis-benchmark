from llm4vis.LLM import LLM
from llm4vis.Dataset import Dataset
from llm4vis.Task import Task


llm = LLM()
dataset = Dataset()
task = Task()

# Below is an example piece of code that starts the benchmark of NvBench dataset for the task of Viz Query Generation
# Must provide deep infra api token

llm.benchmark(
    llm.meta_llama_2_70b_chat_hf, 
    dataset.nvBench, 
    batchsize=50, 
    task=task.vizQueryGeneration, 
    deep_infra_api_token="YOUR_DEEP_INFRA_TOKEN",
    openai_key=None 
)

# Below is an example piece of code that starts the benchmark of NvBench dataset for the task of Viz Generation
# This function utilizes OpenAI's ChatGPT for some comparisons of the Vegalite schemas. It is ESSENTIAL you provide an openai key also

llm.benchmark(
    llm.meta_llama_2_70b_chat_hf, 
    dataset.nvBench, 
    batchsize=50, 
    task=task.vizGeneration, 
    deep_infra_api_token="YOUR_DEEP_INFRA_TOKEN",
    openai_key="YOUR_OPENAI_KEY"
)

# Below is an example piece of code that starts the benchmark of caption generation task

llm.benchmark(
    llm=llm.openai_gpt_3_5_turbo,
    dataset=dataset.visText,
    batchsize=50,
    task=task.captioning,
    deep_infra_api_token="YOUR_DEEP_INFRA_TOKEN",
    openai_key="YOUR_OPENAI_KEY"
)

# Below is an example piece of code that starts the benchmark of code generation task

llm.benchmark(
    llm=llm.mistral_7b_instruct,
    dataset=dataset.codeGenVega,
    batchsize=50,
    task=task.codeGeneration,
    deep_infra_api_token="YOUR_DEEP_INFRA_TOKEN",
    openai_key="YOUR_OPENAI_KEY"
)