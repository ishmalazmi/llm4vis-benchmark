from llm4vis.VisHelper import VisHelper
from llm4vis.Dataset import Dataset
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
from langchain.llms import DeepInfra
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import jsonschema
import altair as alt
from vega import Vega
from vega import VegaLite
import datetime
import re
from langchain.llms import OpenAI
import csv
from langchain.chat_models import ChatOpenAI
import difflib
import matplotlib
import os
import matplotlib
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from lpips import lpips



class Benchmark:
    dataset = ""
    def viz_query_gen(self, llm, dataset, batchsize, deep_infra_api_token):
        self.dataset = Dataset()
        df = self.dataset.load(dataset)
        file_names = []
        try:
            with os.scandir(dataset['path']) as entries:
                elements = [entry.name for entry in entries]
            file_names.append(elements)
        except FileNotFoundError:
            print(f"The folder '{dataset['path']}' does not exist.")
            file_names = []

        i=0
        # Instantiate VisHelper class
        vis_helper = VisHelper()

        # Initialize LLM to benchmark
        llm_to_benchmark = vis_helper.setupLLM(llm_to_benchmark=llm['model_id'], api_token=deep_infra_api_token)
        # print(llm_to_benchmark)

        # Initialize dataframe to store the results
        columns = ["vis_file","nl_query","x_name","y_name","x_data","y_data","formatted_params","prompt","is_viz_query_generated","viz_query_generated","is_viz_executable","vega_similarity"]
        results_df = pd.DataFrame(columns=columns)
    


        for html in file_names[0]:
            
            path = os.path.join(dataset['path'],  html)
            print(f"Dataset Instance {html} ({i+1} of {batchsize})")
            print("===============================================================================")
            try:
                i+=1
                nl_vis_id = vis_helper.return_id_html_file(path)
                print(f"(NL_VIS_ID) => {nl_vis_id}")


                # Evaluation Metrics Start
                viz_query_generated = None
                is_viz_query_generated = False
                is_viz_compilable = False
                # Evaluation Metrics End

                

                vega_lite_str = vis_helper.return_vega_lite_html_file(path, 'vlSpec1')
                vega_lite_str = vega_lite_str.replace("'", "\"")
                ground_truth = vis_helper.return_vega_lite_html_file(path, 'vlSpec1')

                row = df[df['ref'] == nl_vis_id]
                # print(row)
                try:
                    
                    x_name = row['vis_obj'].values[0]['x_name']
                    # print(x_name)
                    y_name = row['vis_obj'].values[0]['y_name']
                    # print(y_name)

                    x_data = row['vis_obj'].values[0]['x_data']
                    # print(x_data)
                    y_data = row['vis_obj'].values[0]['y_data']
                    # print(y_data)

                    nl_query = row['nl_queries'].values[0][0]
                    # print(nl_query)
                    # prompt = build_prompt(nl_query, x_name,y_name,x_data[0],y_data[0])
                    
                    formatted_params = vis_helper.lists_to_formatted_string(x_data[0],y_data[0])
                    
                    # print(formatted_params)
                    # template = f""" 
                    #         You are a data visualization specialist. Your task is given an instruction and data to return only the most appropriate Vega-Lite specification confined in backticks.
                    #         #instruction: 
                    #         {{nl_query}}
                    #         #dataset:
                    #         x_data as {{x_name}} | y_data as {{y_name}} \n
                    #         {{formatted_params}} \n
                    #         Write only the Vegalite JSON specification confined only between backticks like ```Vegalite Spec goes here```.
                    #         """
                    # template = f""" 
                    #         #instruction: 
                    #         {{nl_query}}
                    #         #dataset:
                    #         x_data as {{x_name}} | y_data as {{y_name}} \n
                    #         {{formatted_params}} \n
                    #         Write only the Vegalite JSON specification for above confined between three backticks ```.
                    #         """
                    template = f"""
                                Act as a data visualization specialist. 
                                Your task is given an question and a dataset to return the most appropriate vega lite specification in ```.
                                #instruction: 
                                {{nl_query}}
                                #data:
                                x_data as {{x_name}} | y_data as {{y_name}}\n
                                {{formatted_params}}\n 
                                Return the visualization in vega lite specification confined between three backticks```.
                               """
                    # print(template)

                    prompt = PromptTemplate(template=template, input_variables=["nl_query","x_name","y_name","formatted_params"])
                    # print(prompt)
                    
                    llm_chain = LLMChain(prompt=prompt, llm=llm_to_benchmark)

                    # print(llm_chain)

                    result = llm_chain.run({"nl_query": nl_query, "x_name" : x_name,"y_name" : y_name, "formatted_params": formatted_params})

                    # print(f"ID # {html} => {result}")

                    # Extracting JSON from the string "result"
                    vega_lite_spec = None
                    try:
                        vega_lite_spec = vis_helper.extract_text_between_backticks(result)
                        # print(f"Extracted Vegalite Spec => {vega_lite_spec}")
                        # vega_lite_spec = vis_helper.extract_json_between_backticks(result)
                        viz_query_generated = vega_lite_spec
                    except:
                        print("Error in extracting json schema from LLM response")    
                    # print(f"LLM Response => {result}")

                    print(f"Ground truth => {type(ground_truth)} => {ground_truth}")
                    print(f"LLM Vegalite => {type(vega_lite_spec)} => {vega_lite_spec}")
                    vega_similarity = 0
                    
                    


                    checkcode_result = None
                    
                    try:
                        llm_vegalite = json.loads(vega_lite_spec)
                        viz_query_generated = llm_vegalite
                        
                        is_viz_query_generated=True
                        # checkcode_result = vis_helper.checkcode(ground_truth, viz_query_generated, x_name, y_name, nl_query, id, row['hardness'].values[0])
                        vega_similarity = vis_helper.vega_lite_similarity_score(ground_truth, vega_lite_spec) 

                        try:    
                            alt_chart = alt.Chart.from_dict(llm_vegalite)
                            is_viz_compilable = True
                            
                        except:
                            # print("Something went wrong")
                            is_viz_compilable = False  

                    except json.decoder.JSONDecodeError:
                        print("Error in parsing json from the LLM response")  
                    except TypeError:
                        print("TypeError in json load")       
                    
                    print(f"Has Viz. Query been generated => {is_viz_query_generated}")
                    # print(f"Viz. Query Compilable => {is_viz_compilable}")
                    # print(f"Viz Query Generated => {viz_query_generated}")
                    # print(f"Viz Query GT => {vega_lite_str}")
                    # print(f"Comparison => {checkcode_result}")
                    print(f"Vega Similarity => {vega_similarity}")

                    # Create Result Object to add into results dataframe
                    results= [
                        html,
                        nl_query,
                        x_name,
                        y_name,
                        x_data,
                        y_data,
                        formatted_params,
                        prompt,
                        is_viz_query_generated,
                        viz_query_generated,
                        is_viz_compilable,
                        vega_similarity
                    ]
                    
                    # print(results)

                    # results_df = results_df.concat([results_df, pd.DataFrame(result_element)], ignore_index=True)    
                    results_df.loc[len(results_df)] = results
                    
                    # Write result to dataframe

                except KeyError:
                    print(f"ID {nl_vis_id} not found in the DataFrame keyerror.")
            except IndexError:
                print(f"ID {nl_vis_id} not found in the DataFrame.")
            
            
            print("===============================================================================")
            
            if(i==batchsize):
                break  

        
        # Ensure that the 'results' folder exists
        results_folder = 'results'
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Get the current datetime
        current_datetime = datetime.datetime.now()

        # Format the datetime object as a custom string
        formatted_datetime = current_datetime.strftime("%d-%m-%Y-%H-%M-%S")    

        # Define the file_name
        file_name = f"./{results_folder}/{llm['name']}-viz-query-gen-{dataset['name']}-{formatted_datetime}.csv"

        # Write results_df to CSV
        results_df.to_csv(file_name)



    def viz_generation(self, llm, dataset, batchsize, deep_infra_api_token, openai_key):
        self.dataset = Dataset()
        if(dataset != self.dataset.nvBench):
            print("Only NvBench dataset is supported for this task right now. Please wait for other integrations :)")
            return
        
        os.environ["OPENAI_API_KEY"] = openai_key
        df = self.dataset.load(dataset)
        file_names = []
        try:
            with os.scandir(dataset['path']) as entries:
                elements = [entry.name for entry in entries]
            file_names.append(elements)
        except FileNotFoundError:
            print(f"The folder '{dataset['path']}' does not exist.")
            file_names = []

        i=0
        # Instantiate VisHelper class
        vis_helper = VisHelper()

        # Initialize LLM to benchmark
        llm_to_benchmark = vis_helper.setupLLM(llm_to_benchmark=llm['model_id'], api_token=deep_infra_api_token)
        # print(llm_to_benchmark)

        # Initialize dataframe to store the results
        columns = ["vis_file","nl_query","x_name","y_name","x_data","y_data","formatted_params","prompt","is_viz_query_generated","viz_query_generated","viz_accuracy"]
        results_df = pd.DataFrame(columns=columns)
    


        for html in file_names[0]:
            
            path = os.path.join(dataset['path'],  html)
            print(f"Dataset Instance {html} ({i+1} of {batchsize})")
            print("===============================================================================")
            try:
                i+=1
                nl_vis_id = vis_helper.return_id_html_file(path)
                # print(f"(NL_VIS_ID) => {nl_vis_id}")


                # Evaluation Metrics Start
                viz_query_generated = None
                is_viz_query_generated = False
                is_viz_compilable = False
                # Evaluation Metrics End

                

                vega_lite_str = vis_helper.return_vega_lite_html_file(path, 'vlSpec1')
                vega_lite_str = vega_lite_str.replace("'", "\"")
                ground_truth = vis_helper.return_vega_lite_html_file(path, 'vlSpec1')

                row = df[df['ref'] == nl_vis_id]
                # print(row)
                try:
                    
                    x_name = row['vis_obj'].values[0]['x_name']
                    # print(x_name)
                    y_name = row['vis_obj'].values[0]['y_name']
                    # print(y_name)

                    x_data = row['vis_obj'].values[0]['x_data']
                    # print(x_data)
                    y_data = row['vis_obj'].values[0]['y_data']
                    # print(y_data)

                    nl_query = row['nl_queries'].values[0][0]
                    # print(nl_query)
                    # prompt = build_prompt(nl_query, x_name,y_name,x_data[0],y_data[0])
                    
                    formatted_params = vis_helper.lists_to_formatted_string(x_data[0],y_data[0])
                    
                    # print(formatted_params)
                    # template = f""" 
                    #         You are a data visualization specialist. Your task is given an instruction and data to return only the most appropriate Vega-Lite specification confined in backticks.
                    #         #instruction: 
                    #         {{nl_query}}
                    #         #dataset:
                    #         x_data as {{x_name}} | y_data as {{y_name}} \n
                    #         {{formatted_params}} \n
                    #         Write only the Vegalite JSON specification confined only between backticks like ```Vegalite Spec goes here```.
                    #         """
                    # template = f""" 
                    #         #instruction: 
                    #         {{nl_query}}
                    #         #dataset:
                    #         x_data as {{x_name}} | y_data as {{y_name}} \n
                    #         {{formatted_params}} \n
                    #         Write only the Vegalite JSON specification for above confined between three backticks ```.
                    #         """
                    # template = f"""
                    #             Act as a data visualization specialist. 
                    #             Your task is given an instruction and a dataset to return the most appropriate vega lite specification.
                                
                    #             #instruction: 
                                
                    #             {{nl_query}}
                                
                    #             #data:
                    #             x_data as {{x_name}} | y_data as {{y_name}} \n

                    #             {{formatted_params}} \n

                    #             YOUR OUTPUT MUST ONLY BE THE VEGALITE SPECIFICATION IN JSON FORMAT CONFINED IN BACKTICKS ```.
                    #            """
                    # template = f"""
                    #             YOU ARE A DATA VISUALIZATION EXPERT AND YOUR TASK IS TO CREATE THE MOST APPROPRIATE AND ACCURATE VEGALITE SPECIFICATION IN JSON FORMAT FOR BELOW QUERY AND DATA. YOU MUST RETURN ONLY THE VEGALITE SPECIFICATION IN JSON FORMAT.
                    #             #instruction: 
                                
                    #             {{nl_query}}
                                
                    #             #data:
                    #             x_data as {{x_name}} | y_data as {{y_name}}

                    #             {{formatted_params}}
                    #            """
                    # print(template)

                    result = None
                    if(llm['provider'] == "deepinfra"):
                        template = f"""
                                    You are a helpful assistant highly skilled in writing PERFECT code for visualizations. Your task is to create the most appropriate Vega-Lite source for a given instruction and data. The specification must have data, mark and encoding properties only. \n
                                    #Instruction:\n
                                    {{nl_query}}\n

                                    #Data
                                    x_data as {{x_name}} | y_data as {{y_name}} \n
                                    
                                    {{formatted_params}}
                                    \n

                                    DONT EXPLAIN ANYTHING. JUST GIVE VEGA-LITE SPECIFICATION. PLEASE CONFINE THE VEGALITE SPECIFICATION IN ```.
                                    """
                        

                        vegalite_data = vis_helper.create_vegalite_data(x_data[0], y_data[0])
                        # print(dataset)
                        vegalite_to_complete = {
                            "data" : vegalite_data,
                            "mark" : "Judge this according to given data and NL Query",
                            "encoding" : {
                                "x" : {
                                    "title" : x_name,
                                    ".." : "Judge other properties based on data and query"
                                },
                                "y" : {
                                    "title" : y_name,
                                    ".." : "Judge other properties based on data and query"
                                }
                            },
                            
                        }

                        # template = f"""YOU ARE A VISUALIZATION EXPERT IN VEGALITE SPECIFICATIONS. YOUR TASK IS TO COMPLETE THE GIVEN VEGALITE SPECIFICATION BY FILLING IN THE MARK AND ENCODING PROPERTIES ACCORDING TO NL QUERY AND DATA GIVEN BELOW.
                        
                        # #instruction: 
                        # {{nl_query}}
        
                        # #Vegalite to complete:
                        # {{vegalite_to_complete}}

                        # YOUR OUTPUT MUST ONLY BE THE VEGALITE SPECIFICATION IN FULL."""

                        # print(f"NLQ => {nl_query}")

                        prompt = PromptTemplate(template=template, input_variables=["nl_query","x_name","y_name","formatted_params"])
                        # prompt = PromptTemplate(template=template, input_variables=["nl_query","vegalite_to_complete"])
                        # print(prompt)
                        
                        llm_chain = LLMChain(prompt=prompt, llm=llm_to_benchmark)

                        # print(llm_chain)

                        result = llm_chain.run({
                            "nl_query": nl_query, 
                            "x_name" : x_name,
                            "y_name" : y_name, 
                            "formatted_params": formatted_params,
                            })
                        
                    if(llm['provider'] == 'openai'):
                        print("Provider is OpenAI. Using GPT-3.5 Turbo")
                        os.environ["OPENAI_API_KEY"] = openai_key
                        # OpenAI Provider    
                        # result = llm_chain.run({
                        #     "nl_query": nl_query,
                        #     "vegalite_to_complete" : vegalite_to_complete
                        # })
                        template = f"""You are a helpful assistant highly skilled in writing PERFECT code for visualizations. Your task is to create the most appropriate vegalite specification.
                                        #Goal:
                                        {{nl_query}}

                                        #Data

                                        x_data as {{x_name}}  ,  y_data as {{y_name}}

                                        {{formatted_params}}

                                        PLEASE CONFINE THE VEGALITE JSON IN ```.
                                    """
                        prompt = PromptTemplate(template=template, input_variables=["nl_query","x_name","y_name","formatted_params"])
                                        # prompt = PromptTemplate(template=template, input_variables=["nl_query","vegalite_to_complete"])
                                        # print(prompt)
                                        
                        llm_chagpt = ChatOpenAI(model=llm['model_id'])

                        llm_chain_chatgpt = LLMChain(prompt=prompt, llm=llm_chagpt)

                  

                        result = llm_chain_chatgpt.run({
                            "nl_query": nl_query, 
                            "x_name" : x_name,
                            "y_name" : y_name, 
                            "formatted_params": formatted_params,
                        })
                    
                    # Define evaluation metrics
                    viz_accuracies = {
                        "json" : 0,
                        "lpips_alex" : 0,
                        "lpips_vgg" : 0,
                        "ssi" : 0
                    }

                    print(f"ID # {html} => {result}")

                    # extracted_json = vis_helper.get_json_from_llm_response(result)
                    extracted_json = vis_helper.extract_text_with_braces(result)

                    print(f"Extracted JSON => {extracted_json}")

                    is_chart_gen_generated = False  # This is for checking if a chart has been generated or not for viz
                    is_chart_gt_generated = False  # This is for checking if a chart has been generated or not for gt
                    directory = f'./././data/intermediate/NvBench/{html}/'
                    # Check if extracted_json was successfull
                    if extracted_json:
                        
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        try:
                            try:
                                print("Trying to load extracted_json as json.loads()")
                                extracted_json = json.loads(extracted_json)
                            except:
                                print("[Extracted JSON Load Error] : Could not parse Extracted JSON as json dict")   

                            # Create Altair chart from the loaded JSON (dictionary format)
                            chart_gen = alt.Chart.from_dict(extracted_json)
                            chart_gen.save(f'{directory}gen.png')
                            is_chart_gen_generated = True
                        except:
                            is_chart_gen_generated = False
                            print("Failed to create visualization from Generated Vegalite")
                            # Here we can make use of GPT to extract the JSON and correct the quotes where needed
                            template_chatgpt = f"""
                                        You are given a string below. Your task is to extract only vegalite specification (json) from the given input below. YOU HAVE TO CORRECT QUOTES/COMMAS AS WELL.

                                        {{result}}

                                        YOUR OUTPUT MUST ONLY BE A VEGALITE SPECIFICATION CONFINED IN ```.
                                        """

                            prompt_viz_accuracy = PromptTemplate(template=template_chatgpt, input_variables=["result"])

                            llm_chagpt = ChatOpenAI(model="gpt-3.5-turbo")

                            llm_chain_chatgpt = LLMChain(prompt=prompt_viz_accuracy, llm=llm_chagpt)

                            # accuracy_result = llm_chain_chatgpt.run({"json_schema" : json_schema,"ground_truth":ground_truth, "generated_vegalite":result})
                            vegalite_from_gpt = llm_chain_chatgpt.run({"result":result})
                            vegalite_gpt = vis_helper.extract_text_with_braces(vegalite_from_gpt)

                            print(f"GPT-Extracted and Correct JSON => {vegalite_gpt}")
                            try:
                                # Create Altair chart from the loaded JSON (dictionary format)
                                chart_gen = alt.Chart.from_dict(json.loads(vegalite_gpt))
                                chart_gen.save(f'{directory}gen.png')
                                is_chart_gen_generated = True
                            except:
                                print("GPT Extracted JSON could not generate visualization for Altair")        

                        try:
                            # Try now to save ground truth visualization
                            chart_gt = alt.Chart.from_dict(json.loads(ground_truth))
                            chart_gt.save(f'{directory}gt.png')
                            is_chart_gt_generated = True
                        except:
                            print("Failed to create visualization from Ground Truth") 

                    else:
                        
                        print("Failed to load a Vega-Lite specification from the LLM Result.")
                        

                    try:
                        viz_accuracies['json'] = vis_helper.get_grammar_similarity_score(json.loads(ground_truth),extracted_json)
                    except:
                        print("Failed to calculate Grammar similarity")


                    if is_chart_gen_generated and is_chart_gt_generated:
                        # It means the LLM generated visualization has been rendered successfully
                        # Now we calculate the Viz Accuracies with Computer Vision
                        image_1_path = f"{directory}gt.png"         
                        image_2_path = f"{directory}gen.png"         
                        viz_accuracies['ssi'] = vis_helper.get_ssi_scores(image_1_path, image_2_path)
                        viz_accuracies['lpips_alex'] = vis_helper.get_lpips_scores(image_1_path, image_2_path,'alex')
                        viz_accuracies['lpips_vgg'] = vis_helper.get_lpips_scores(image_1_path, image_2_path,'vgg')

                    
                    json_schema = [
                                    {"dimension":  "data",  "score": "0-10"}, 
                                    {"dimension":  "type",  "score": "0-10"},
                                    {"dimension":  "schema",  "score": "0-10"},
                                    {"dimension":  "encoding",  "score": "0-10"},
                                    {"dimension":  "rating",  "score": "0-10"},
                                ]

                    # Code here was exported out to MassCode     
                    
                    # Create Result Object to add into results dataframe
                    results= [
                        llm['name'],
                        dataset['name'],
                        html,
                        nl_query,
                        x_name,
                        y_name,
                        x_data,
                        y_data,
                        formatted_params,
                        prompt,
                        ground_truth,
                        result,
                        extracted_json,
                        viz_accuracies['json'],
                        viz_accuracies['lpips_alex'],
                        viz_accuracies['lpips_vgg'],
                        viz_accuracies['ssi']
                        
                    ]
                    
                    csv_file_path = './results/viz-gen.csv'
                    # Append 'results' list to the CSV file
                    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(results)


                    
                    
                    # print(results)

                    # results_df = results_df.concat([results_df, pd.DataFrame(result_element)], ignore_index=True)    
                    # results_df.loc[len(results_df)] = results
                    
                    # Write result to dataframe

                except KeyError:
                    print(f"ID {nl_vis_id} not found in the DataFrame keyerror.")
            except IndexError:
                print(f"ID {nl_vis_id} not found in the DataFrame.")
            
            
            
            print("===============================================================================")
            
            if(i==batchsize):
                break  




    def caption_generation(self, llm, dataset, batchsize, deep_infra_api_token, openai_key):    
        self.dataset = Dataset()
        if(dataset == self.dataset.nvBench):
            print("nvBench dataset is not yet supported. Please use Vistext")
            return
        captions_df = None
        # captions_df = pd.read_csv(dataset['path'])
        try:
            captions_df = pd.read_csv(dataset['path'])
        except FileNotFoundError:
            print("File not found!")
            return
        

       
        

        

        # Initialize dataframe to store the results
        # columns = ["llm","dataset","caption_id","img_id","datatable","L1_caption","L2L3_caption","predicted_caption","cosine_similarity","jaccard_similarity"]
        # results_df = pd.DataFrame(columns=columns)

        # Instantiate VisHelper class
        vis_helper = VisHelper()

        # Initialize LLM to benchmark
        llm_to_benchmark = vis_helper.setupLLM(llm_to_benchmark=llm['model_id'], api_token=deep_infra_api_token)

        # Set up the prompt for LLM to generate caption
        template = f"""
                    {{datatable}}

                    Return only the most accurate caption that effectively communicates key insights from the information above. Write nothing else.
                    """
        # print(template)

        prompt = PromptTemplate(template=template, input_variables=["datatable"])
        # print(prompt)
        llm_chain = None            
        if llm['provider'] == "deepinfra":
            llm_chain = LLMChain(prompt=prompt, llm=llm_to_benchmark)

        if llm['provider'] == "openai":
            os.environ["OPENAI_API_KEY"] = openai_key
            llm_chagpt = ChatOpenAI(model=llm['model_id'])

            llm_chain = LLMChain(prompt=prompt, llm=llm_chagpt)    

        # print(llm_chain)

        

        # Iterate through rows of the DataFrame
        for index, row in captions_df.iterrows():
            if index >= batchsize:
                break  # Exit the loop after batchsize is reached

            print(f"Dataset Instance # {index+1} of {batchsize}")
            print("=============================================")
            # Access and print specific columns if needed
            result = llm_chain.run({"datatable": row['datatable']})
            
            print(f"Image ID # {row['img_id']} => {result}")
            bleu_score = 0
            rouge1_score = 0
            rouge2_score = 0
            rougel_score = 0
            cosine_similarity_score = 0
            jaccard_similarity_score = 0
            if(result.strip() != ""):
                # print("Calculating BLEU and ROUGE Scores")
                # bleu_score, rouge_score = vis_helper.get_evaluation_scores(row['caption_L2L3'], result.strip(), True, True)
                # print(f"BLEU Score : {bleu_score}")
                # print(f"ROUGE-1 (f) : {rouge_score['rouge-1']['f']}")
                # print(f"ROUGE-2 (f) : {rouge_score['rouge-2']['f']}")
                # print(f"ROUGE-L (f) : {rouge_score['rouge-l']['f']}")
                # print(f"ROUGE SCORES : {rouge_score}")

                # rouge1_score = rouge_score['rouge-1']['f']
                # rouge2_score = rouge_score['rouge-2']['f']
                # rougel_score = rouge_score['rouge-l']['f']
                print("Calculating Cosine Similarity Score...")
                cosine_similarity_score = vis_helper.get_cosine_similarity(row['caption_L2L3'], result.strip())
                print(f"Cosine Similarity Score: {cosine_similarity_score:.4f}")
                print("Calculating Jaccard Similarity Score...")
                jaccard_similarity_score = vis_helper.get_jaccard_similarity(row['caption_L2L3'], result.strip())
                print(f"Jaccard Similarity Score: {jaccard_similarity_score:.4f}")


            # Create Result Object to add into results dataframe
            results= [
                llm['name'],
                dataset['name'],
                row['caption_id'],
                row['img_id'],
                row['datatable'],
                row["caption_L1"],
                row["caption_L2L3"],
                result.strip(),
                cosine_similarity_score,
                jaccard_similarity_score
            ]
                    

            # results_df = results_df.concat([results_df, pd.DataFrame(result_element)], ignore_index=True)    
            # results_df.loc[len(results_df)] = results
            csv_file_path = './results/captioning.csv'
            # Append 'results' list to the CSV file
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(results)

    

            print("=============================================")

    def code_generation(self, llm, dataset, batchsize, deep_infra_api_token, openai_key):
        import shutil
        import time

        matplotlib.use('Agg')  # Set non-interactive backend    
        self.dataset =  Dataset()
        os.environ["OPENAI_API_KEY"] = openai_key
        if(dataset != self.dataset.codeGenVega):
            print("Only CodeGenVega dataset is supported for this task right now. Please wait for other integrations :)")
            return
        
        nvbench_df = pd.read_json("./data/raw/NvBench/NVBench.json", orient='index')
        nvbench_df.reset_index(inplace=True)
        nvbench_df.rename(columns={'index': 'ref'}, inplace=True)
    
        if os.path.exists("./data/intermediate/CodeGen/"+llm['short_name']):
            shutil.rmtree("./data/intermediate/CodeGen/"+llm['short_name'])

             # print(prompt)
           
        
        codes_df = pd.read_csv(dataset['path'],encoding='cp1252')
        for index, row in codes_df.iterrows():
            
            #Evaluation metrics variables
            ssi_score = 0
            lpips_score = 0
            result = False
            is_executable = False
            pycode_similarity = 0
        
            # Api related variables
            api_tries = 1
            max_api_tries = 5


            print("================================")
            print(f"========{row['vis_file']}==========")
            print("================================")
            nvbench_row = nvbench_df[nvbench_df['ref'] == row['nl_vis_id']]
            x_name = nvbench_row['vis_obj'].values[0]['x_name']
            y_name = nvbench_row['vis_obj'].values[0]['y_name']
            x_data = nvbench_row['vis_obj'].values[0]['x_data']
            y_data = nvbench_row['vis_obj'].values[0]['y_data']
            order = nvbench_row['vis_obj'].values[0]['describe']
            nl_query = nvbench_row['nl_queries'].values[0][0]

            template_print = f""" Below you have some data and instruction on generating a visualization. YOU HAVE TO WRITE PYTHON CODE USING MATPLOTLIB TO GENERATE THE VISUALIZATION DESIRED.\n
                            {x_name} = {x_data[0]} \n
                            {y_name} = {y_data[0]} \n
                            {order} \n

                            {nl_query} \n

                            CONFINE ONLY THE PYTHON CODE IN ```.
                        """
            template = f""" Below you have some data and instruction on generating a visualization. YOU HAVE TO WRITE PYTHON CODE USING MATPLOTLIB TO GENERATE THE VISUALIZATION DESIRED.\n
                            {{x_name}} = {{x_data}} \n
                            {{y_name}} = {{y_data}} \n
                            {{order}} \n

                            {{nl_query}}

                            WRITE ONLY THE PYTHON CODE. PLEASE CONFINE THE CODE IN ```.
                        """
            prompt = PromptTemplate(template=template, input_variables=["x_name","y_name","x_data","y_data","order","nl_query"])

            # Instantiate VisHelper class
            vis_helper = VisHelper()

            # Initialize LLM to benchmark
            llm_to_benchmark = vis_helper.setupLLM(llm_to_benchmark=llm['model_id'], api_token=deep_infra_api_token)

            llm_chain = None            
            if llm['provider'] == "deepinfra":
                llm_chain = LLMChain(prompt=prompt, llm=llm_to_benchmark)

            if llm['provider'] == "openai":
                os.environ["OPENAI_API_KEY"] = openai_key
                llm_chagpt = ChatOpenAI(model=llm['model_id'])

                llm_chain = LLMChain(prompt=prompt, llm=llm_chagpt) 
            
            while api_tries < max_api_tries:
                try:
                    result = llm_chain.run({
                        "x_name": x_name, 
                        "y_name": y_name, 
                        "x_data": x_data, 
                        "y_data": y_data, 
                        "order": order, 
                        "nl_query": nl_query, 
                    })
                    break  # If successful, break out of the loop
                except:
                    print(f"[DeepInfra Exception] : Connection timed out. Attempt {api_tries}/{max_api_tries}")
                    api_tries = api_tries + 1    

            if result == False or result == '':
                is_executable = False
                print("No code was generated")
                continue

            # print(f"{row['vis_file']} => {result}")
            # Find the Python code between the first import statement and plt.show(), including plt.show()
            match_import = re.search(r'import.*\n', result)
            match_show = re.search(r'plt\.show\(\)', result)

            start_index = match_import.start() if match_import else 0
            end_index = match_show.end() if match_show else len(result)

            extracted_code = result[start_index:end_index].strip()

            print(f"LLM Response => {result}")
            print(f"Extracted Code => {extracted_code}")
            
            
            try:
                exec(extracted_code)
                is_executable = True   
                # pass 
            except:
                is_executable = False
                code_similarity = 0
                print("[Code Exception] : Generated Code is not executable and has errors")
                # pass

            if is_executable:
                print("Generated code is executable!")
                try:
                    # Calculate similarity ratio
                    code_similarity = (difflib.SequenceMatcher(None, row['python_code'], extracted_code).ratio()) * 100
                    print(f"Code Similarity => {code_similarity}")
                except:
                    code_similarity = 0
                    print("[Code Similarity Exception] : Error in calculating code similarity")
                    # pass

                
                # codes_df.at[index, 'ssi_score'] = ssi_score
                # codes_df.at[index, 'lpips_score'] = lpips_score
                flag = 0
                # Define the line to be added
                directory = "./data/intermediate/CodeGen/" + llm['short_name'] + "/"+row['vis_file']+"/"
                if not os.path.exists(directory):
                    os.makedirs(directory)

                line_to_add_gt = f"plt.savefig('{directory}gt.png')\n"
                line_to_add_gen = f"plt.savefig('{directory}gen.png')\n"

                try:
                    # Split the code string into lines
                    # lines = row['python_code'].split('\n')

                    # # Add the line before plt.show() and reassemble the code
                    # modified_code = '\n'.join(lines[:-2] + [line_to_add_gt] + lines[-2:])
                    lines = row['python_code'].split('\n')
                    show_index = next((i for i, line in enumerate(lines) if "plt.show()" in line), None)
                    if show_index is not None:
                        lines.insert(show_index, line_to_add_gt)
                        modified_code = '\n'.join(lines)
                    
                    # Write the modified code to a Python script
                    with open(directory+"gt.py", "w") as file:
                        file.write(modified_code)
                    
                    # Execute the script
                    exec(open(directory+"gt.py").read())
                    flag += 1
                    time.sleep(3)
                except:
                    print("[Execution Code Error] : Error while executing Ground Truth Code")
                    ssi_score = 0
                    lpips_score = 0
                    # pass    

                # Do this for generated code now
                try:
                    # Split the code string into lines
                    # lines = extracted_code.split('\n')

                    # # Add the line before plt.show() and reassemble the code
                    # modified_code_gen = '\n'.join(lines[:-2] + [line_to_add_gen] + lines[-2:])
                    lines = extracted_code.split('\n')
                    show_index = next((i for i, line in enumerate(lines) if "plt.show()" in line), None)
                    if show_index is not None:
                        lines.insert(show_index, line_to_add_gen)
                        modified_code_gen = '\n'.join(lines)
                    
                    # Write the modified code to a Python script
                    with open(directory+"gen.py", "w") as file:
                        file.write(modified_code_gen)
                    
                    # Execute the script
                    exec(open(directory+"gen.py").read())
                    flag += 1
                    time.sleep(3)
                except:
                    ssi_score = 0
                    lpips_score = 0
                    print("[Execution Code Error] : Error while executing Generated Code")
                    # pass

                line_to_add_gen = None
                line_to_add_gt = None     


                if flag == 2:
                    image1 = cv2.imread(directory+"gt.png")
                    image2 = cv2.imread(directory+"gen.png")
                    if image1 is None and image2 is None:
                        continue
                    # # Resize images to have the same dimensions
                    try:
                        width = min(image1.shape[1], image2.shape[1])
                        height = min(image1.shape[0], image2.shape[0])
                        image1 = cv2.resize(image1, (width, height))
                        image2 = cv2.resize(image2, (width, height))
                    except:
                        print("[Image Error] : Could not parse Image/Find image")
                        continue    

                    # Convert the images to grayscale
                    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)



                    # Compute the Structural Similarity Index (SSI)
                    ssi_index, _ = ssim(gray_image1, gray_image2, full=True)

                    # Compute the Mean Squared Error (MSE)
                    mse = ((gray_image1 - gray_image2) ** 2).mean()

                    # Display the images
                    # plt.subplot(131), plt.imshow(gray_image1, cmap='gray'), plt.title('Image 1')
                    # plt.subplot(132), plt.imshow(gray_image2, cmap='gray'), plt.title('Image 2')
                    # plt.subplot(133), plt.imshow(gray_image1 - gray_image2, cmap='gray'), plt.title('Difference')

                    # plt.show()

                    # Display the comparison metrics
                    print(f'Structural Similarity Index: {ssi_index:.2f}')
                    print(f'Mean Squared Error: {mse:.2f}')

                    ssi_score = ssi_index * 100
                    # row['ssi_score'] = ssi_score
                    # codes_df.at[index, 'ssi_score'] = ssi_score * 100

                    # LPIPS Code
                    # Function to load and preprocess an image
                    def load_and_preprocess_image(image_path, target_size=(256, 256)):
                        # Load the image using PIL
                        image = Image.open(image_path).convert('RGB')

                        # Resize the image to the target size
                        resize_transform = transforms.Resize(target_size)
                        image = resize_transform(image)

                        # Define a transformation to convert the PIL image to a PyTorch tensor
                        transform = transforms.Compose([
                            transforms.ToTensor(),  # Converts the image to a PyTorch tensor
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image to the range [-1, 1]
                        ])

                        # Apply the transformation to the image
                        image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

                        return image_tensor

                    # Replace 'path_to_image1.jpg' and 'path_to_image2.jpg' with the paths to your images
                    image1_path = directory+'gen.png'
                    image2_path = directory+'gt.png'

                    # Load and preprocess the images as tensors
                    image1_tensor = load_and_preprocess_image(image1_path)
                    image2_tensor = load_and_preprocess_image(image2_path)



                    # Continue with the LPIPS calculation code from the previous example
                    lpips_model = lpips.LPIPS(net='vgg')
                    lpips_distance = lpips_model(image1_tensor, image2_tensor).item()

                    print(f"LPIPS distance between the images (vgg): {lpips_distance}")
                    lpips_score = (1 - lpips_distance)* 100
                    # row['lpips_score'] = lpips_score
                    # codes_df.at[index, 'lpips_score'] = lpips_score * 100

                    # Code Similarity Measure
                    

            try:
                referenced_code_str = "def fun1(): " + modified_code
                candidate_code_str1 = "def fun2(): " + modified_code_gen

                result = pycode_similar.detect([referenced_code_str, candidate_code_str1], diff_method=pycode_similar.UnifiedDiff, keep_prints=True, module_level=True)
                summary = pycode_similar.summarize(result[0][1])
                pycode_similarity = summary[0]
            except:
                print("Could not calculate Code Similarity (PyCode Similar)")
                pycode_similarity = 0   
                    

            # Create Result Object to add into results dataframe
            results= [
                llm['name'],
                dataset['name'],
                row['vis_file'],
                row['nl_vis_id'],
                row['python_code'],
                extracted_code,
                is_executable,
                pycode_similarity,
                ssi_score,
                lpips_score
            ]
                    

            # results_df = results_df.concat([results_df, pd.DataFrame(result_element)], ignore_index=True)    
            # results_df.loc[len(results_df)] = results
            csv_file_path = './results/code-gen.csv'
            # Append 'results' list to the CSV file
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(results)
            print("================================")    

            if(index+1 == batchsize):
                break   
        
        
        
    
       

