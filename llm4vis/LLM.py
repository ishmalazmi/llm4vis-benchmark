from llm4vis.Task import Task
from llm4vis.Benchmark import Benchmark

class LLM:
    llms_available = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "meta-llama/Llama-2-70b-chat-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "jondurbin/airoboros-l2-70b-gpt4-1.4.1",
        "Gryphe/MythoMax-L2-13b",
        "Salesforce/codegen-16B-mono",
        "bigcode/starcoder",
        "databricks/dolly-v2-12b",
        "meta-llama/Llama-2-13b-hf"
        "meta-llama/Llama-2-7b-hf",
        "tiiuae/falcon-7b"
    ]
    meta_llama_2_7b_chat_hf = {
        "name" : "Llama-2-7b-chat-hf",
        "model_id" : "meta-llama/Llama-2-7b-chat-hf",
        "configuration" : {
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 2000,
            "top_p": 0.9,
        },
        "provider" : "deepinfra",
        "short_name" : "meta_llama_7b"
    }
    meta_llama_2_13b_chat_hf = {
        "name" : "Llama-2-13b-chat-hf",
        "model_id" : "meta-llama/Llama-2-13b-chat-hf",
        "configuration" : {
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 2000,
            "top_p": 0.9,
        },
        "provider" : "deepinfra",
        "short_name" : "meta_llama_13b"
    }
    meta_llama_2_70b_chat_hf = {
        "name" : "meta-llama-2-70b-chat-hf",
        "model_id" : "meta-llama/Llama-2-70b-chat-hf",
        "configuration" : {
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 2000,
            "top_p": 0.9,
        },
        "provider" : "deepinfra",
        "short_name" : "meta_llama_70b"
    }
    codellama_34b_instruct_hf = {
        "name" : "CodeLlaMA 34B",
        "model_id" : "codellama/CodeLlama-34b-Instruct-hf",
        "configuration" : {
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 2000,
            "top_p": 0.9,
        },
        "provider" : "deepinfra",
        "short_name" : "codellama_34b"
    }
    mistral_7b_instruct = {
        "name" : "Mistral-7B-Instruct",
        "model_id" : "mistralai/Mistral-7B-Instruct-v0.1",
        "configuration" : {
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 2000,
            "top_p": 0.9,
        },
        "provider" : "deepinfra",
        "short_name" : "mistral_7b"
    }
    deepinfra_airoboros_70b = {
        "name" : "DeepInfra-Airoboros-70B",
        "model_id" : "deepinfra/airoboros-70b",
        "configuration" : {
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 2000,
            "top_p": 0.9,
        },
        "provider" : "deepinfra",
        "short_name" : "deepinfra_airoboros_70b"
    }
    salesforce_codegen_16b_mono = {
        "name" : "Salesforce-CodeGen-16B-Mono",
        "model_id" : "Salesforce/codegen-16B-mono",
        "configuration" : {
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 2000,
            "top_p": 0.9,
        },
        "provider" : "deepinfra",
        "short_name" : "salesforce_codegen_16b_mono"
    }


    phind_code_llama_v2 = {
        "name" : "Phind/Phind-CodeLlama",
        "model_id" : "Phind/Phind-CodeLlama-34B-v2",
        "configuration" : {
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 2000,
            "top_p": 0.9,
        },
        "provider" : "deepinfra",
        "short_name" : "phind_code_llama_v2"
    }

    openai_gpt_3_5_turbo = {
        "name" : "OpenAI-GPT-3.5-Turbo",
        "model_id" : "gpt-3.5-turbo",
        "provider" : "openai",
        "short_name" : "gpt_3_5_turbo"
    }
    
    def show_available_llms(self):
        print("The available LLMs are \n")
        for llm in self.llms_available:
            print(llm)

    def benchmark(self, llm, dataset, batchsize, task, deep_infra_api_token, openai_key=None):
        print(f"Starting benchmark of the LLM {llm['name']} against a batchsize of {batchsize} from the dataset {dataset['name']} for the task of {task['name']}")
        print("Hold on tight! This may take a while :)")
        
        if (task == Task.vizQueryGeneration):
            benchmark = Benchmark()
            benchmark.viz_query_gen(llm=llm, dataset=dataset, batchsize=batchsize, deep_infra_api_token=deep_infra_api_token)

        if(task == Task.vizGeneration):
            benchmark = Benchmark()
            benchmark.viz_generation(llm=llm, dataset=dataset, batchsize=batchsize, deep_infra_api_token=deep_infra_api_token, openai_key=openai_key)

        if(task == Task.captioning):
            benchmark = Benchmark()
            benchmark.caption_generation(llm=llm, dataset=dataset, batchsize=batchsize, deep_infra_api_token=deep_infra_api_token, openai_key=openai_key)

        if(task == Task.codeGeneration):
            benchmark = Benchmark()
            benchmark.code_generation(llm=llm, dataset=dataset, batchsize=batchsize, deep_infra_api_token=deep_infra_api_token, openai_key=openai_key)