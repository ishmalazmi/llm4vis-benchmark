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
import altair as alt
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge import Rouge
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from lpips import lpips
import difflib
from contextlib import contextmanager
import sys

class VisHelper:
    deep_infra_api_token = ""

    # Custom context manager to suppress output
    @contextmanager
    def suppress_stdout():
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

                
    def set_deep_infra_api_token(self, token):
        self.deep_infra_api_token = token

    def return_vega_lite_html_file(self, html_file, variable_name):
        with open(html_file, 'r') as file:
            soup = BeautifulSoup(file, 'html.parser',)
            script_tags = soup.find_all('script')
            # print(script_tags)

            for script_tag in script_tags:
                script_content = script_tag.string
                if script_content:
                    match = re.search(rf'var\s+{variable_name}\s*=\s*(.*?);', script_content)
                    if match:
                        return self.convert_single_quotes_to_double_quotes(match.group(1))

        return None
    
    def return_id_html_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')
        p_tags = soup.find_all('p')

        for p_tag in p_tags:
            # Check if the p_tag contains the specific pattern "<p><b>(NL, VIS) ID: </b>772@y_name@DESC</p>"
            if p_tag.b and "(NL, VIS) ID: " in p_tag.b.text:
                # Extract the value after "(NL, VIS) ID: " and before the "@" symbol
                value = p_tag.text.split("(NL, VIS) ID: ")[1]
                return value.strip()

        return None 
    
    def extract_text_between_backticks(self, text):
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].replace('json', '')
    
    def extract_text_with_braces(self, text):
        start_index = text.find('{')  # Find the index of the first '{'
        if start_index == -1:
            return None  # If '{' is not found, return None or handle the case accordingly
        
        open_braces = 1
        end_index = -1
        
        for i in range(start_index + 1, len(text)):
            if text[i] == '{':
                open_braces += 1
            elif text[i] == '}':
                open_braces -= 1
            
            if open_braces == 0:
                end_index = i
                break
        
        if end_index != -1:
            return text[start_index:end_index + 1]  # Return the text enclosed in '{' and '}'
        else:
            return None  # If '}' is not found, return None or handle the case accordingly
    
    def extract_json_between_backticks(input_string):
        # Use a regular expression to find and extract JSON between backticks
        pattern = r'`(.*?)`'
        match = re.search(pattern, input_string, re.DOTALL)

        if match:
            extracted_json = match.group(1)
            return extracted_json
        else:
            return None  # Return None if no JSON between backticks is found
        
    def extract_json_from_string(input_string):
        # Use a regular expression to find and extract JSON
        pattern = r'({.*?})'  # Matches JSON objects within curly braces
        matches = re.findall(pattern, input_string, re.DOTALL)

        extracted_json = []
        for match in matches:
            try:
                json_obj = json.loads(match)
                extracted_json.append(json_obj)
            except json.JSONDecodeError:
                pass  # Ignore invalid JSON objects

        return extracted_json    

    def compare_vega_lite_specifications(self, specification1, specification2):
        similarity_score = 0
        for key, value in specification1.items():
            if key in specification2:
                if value == specification2[key]:
                    similarity_score += 1
        return similarity_score

    def extract_text(self, input_string):
        pattern = r'[A-Za-z\s]+'
        matches = re.findall(pattern, input_string)
        return ''.join(matches)
    
    def lists_to_formatted_string(self, list1, list2):
        formatted_list = [f"{val1}, {val2}" for val1, val2 in zip(list1, list2)]
        formatted_string = '\n'.join(formatted_list)
        return formatted_string
    
    def build_prompt(self, nl_query, x_name, y_name, x_data, y_data):
        formatted_params = self.lists_to_formatted_string(x_data,y_data)
        
        template = f""" 
                    Act as a data visualization specialist. Your task is given an question and a dataset to create the most appropriate vega lite specification.
                    #instruction: 
                    {nl_query}
                    #dataset
                    x_data as {x_name} | y_data as {y_name} \n
                    {formatted_params} \n
                """
        # print(template)

        prompt = PromptTemplate(template=template)
        
        return prompt
    
    def setupLLM(self, llm_to_benchmark, api_token):
        llm = DeepInfra(model_id=llm_to_benchmark, deepinfra_api_token=api_token)
        llm.model_kwargs = {
            "temperature": 0.7,
            "repetition_penalty": 1.2,
            "max_new_tokens": 2000,
            "top_p": 0.9,
        }
        return llm
    

    def extract_json_from_string(self, json_string):
        try:
            # Use json.loads to parse the JSON object from the string
            json_object = json.loads(json_string)
            return json_object
        except json.JSONDecodeError as e:
            # Handle any JSON parsing errors, e.g., invalid JSON string
            print(f"Error parsing JSON: {e}")
            return None

        
        
    def checkcode(self, ref, gen, x_name, y_name, command, id, hardness="none"):
        """
        Check the code similarity and other metrics between reference and generated VegaLite specification.

        Parameters
        ----------
        ref : str
            The reference VegaLite to compare against.

        gen : str
            The generated VegaLite to be evaluated.

        x_name : str
            The name of the x-axis variable for plotting.

        y_name : str
            The name of the y-axis variable for plotting.

        command : str
            The nl query asked by the user to generate the VegaLite.

        id: int
            VegaLite groundtruth identifier from nvBench

        hardness : str, optional
            The hardness level VegaLite from nvBench (default is "none").

        Returns
        -------
        accuracy_x_axes_type : int
            An integer representing the type of x-axis accuracy.

        accuracy_x_axes : int
            An integer representing the accuracy of x-axis.

        accuracy_y_axes_type : int
            An integer representing the type of y-axis accuracy.

        accuracy_y_axes : int
            An integer representing the accuracy of y-axis.

        accuracy_marker : int
            An integer representing the accuracy of marker.

        code_similarity : int
            An integer representing the code similarity between ref and gen.

        is_valid : int
            An integer indicating whether the generated code is valid (1 for compiled, 0 for invalid).

        as_text_similarity : int
            An integer representing the similarity of the generated code when treated as text.

        as_vega_similarity : int
            An integer representing the similarity of the generated code when treated as Vega-Lite JSON.

        Notes
        -----
        This function calculates various metrics to evaluate the similarity and validity of
        generated code in comparison to a reference code. The metrics include accuracy measurements
        for x-axis and y-axis, marker accuracy, code similarity, validity, and similarity as text
        and Vega-Lite JSON.

        The `hardness` parameter can be used to specify the level of difficulty for the code
        comparison, which may affect the evaluation metrics.

        """
   
        accuracy_x_axes_type = 0
        accuracy_x_axes = -1
        accuracy_y_axes_type = 0
        accuracy_y_axes = -1
        accuracy_marker = 0
        code_similarity = 0
        is_valid = 0
        as_text_similarity = 0
        as_vega_similarity = 0

        #compute code similarity
        ref_str = ref.replace('\n', '').replace('  ', '')
        gen_str = gen.replace('\n', '').replace('  ', '')

        referenced_code_str = "def test(): return" + ref_str
        candidate_code_str1 = "def test(): return" + gen_str

        #measure the similarity as code plagiarism
        try:
            pycode_similar_res = pycode_similar.detect([referenced_code_str, candidate_code_str1], diff_method=pycode_similar.UnifiedDiff, keep_prints=False, module_level=False)
            code_similarity, _, _ = pycode_similar.summarize(pycode_similar_res[0][1])
        except Exception:
            code_similarity = 0

        try:

            ref = json.loads(ref)
            gen = json.loads(gen)

            #check the mark type of the genereted vis w.r.t. the ground truth
            try:
                type_gen = ""
                type_ground = ""
                if("type" in gen['mark']):
                    type_gen =  gen['mark']['type']
                else:
                    type_gen =  gen['mark']
                if("type" in ref['mark']):
                    type_ground =  ref['mark']['type']
                else:
                    type_ground =  ref['mark']
                if (type_ground == type_gen):
                    accuracy_marker = 1
                else:
                    accuracy_marker = 0
            except KeyError as er:
                print(f"Ecco l'error {er}")
                accuracy_marker = -1

            #check the x_axes of the genereted vis w.r.t. the ground truth
            for key_1 in ['field', 'title']:
                for key_2 in ['field', 'title']:
                    try:
                        if (self.extract_text(ref['encoding']['x'][key_1]).lower() == self.extract_text(gen['encoding']['x'][key_2]).lower()):
                            accuracy_x_axes += 1
                        else:
                            accuracy_x_axes = 0
                    except KeyError as er:
                        pass

            #check the y_axes of the genereted vis w.r.t. the ground truth
            for key_1 in ['field', 'title']:
                for key_2 in ['field', 'title']:
                    try:
                        if (self.extract_text(ref['encoding']['y'][key_1]).lower() == self.extract_text(gen['encoding']['y'][key_2]).lower()):
                            accuracy_y_axes = 1
                        else:
                            accuracy_y_axes = 0
                    except KeyError as er:
                        pass
            
            #check the y_axes_type of the genereted vis w.r.t. the ground truth
            try:
                if (ref['encoding']['y']['type'] == gen['encoding']['y']['type']):
                    accuracy_y_axes_type = 1
                else:
                    accuracy_y_axes_type = 0
            except KeyError as er:
                accuracy_y_axes_type = -1 
            
            #check the x_axes_type of the genereted vis w.r.t. the ground truth
            try:
                if (ref['encoding']['x']['type'] == gen['encoding']['x']['type']):
                    accuracy_x_axes_type = 1
                else:
                    accuracy_x_axes_type = 0
            except KeyError as er:
                accuracy_x_axes_type = -1

            #check if the vega lite compiles or not
            try:
                chart = alt.Chart().from_dict(gen)
                is_valid = 1
            except Exception as e:
                is_valid = 0

            #compute code similarity as text
            as_text_similarity = SequenceMatcher(None, json.dumps(gen), json.dumps(ref)).ratio()

            #compute code similarity as vega considering the number of key which are equals
            as_vega_similarity = self.compare_vega_lite_specifications(ref, gen)

            return {
                'accuracy_x_axes_type': accuracy_x_axes_type, 
                'accuracy_x_axes': accuracy_x_axes, 
                'accuracy_y_axes_type': accuracy_y_axes_type, 
                'accuracy_y_axes': accuracy_y_axes, 
                'accuracy_marker': accuracy_marker, 
                'code_similarity': code_similarity,
                'is_valid': is_valid,
                'as_text_similarity': as_text_similarity,
                'as_vega_similarity': as_vega_similarity,
                'hardness': hardness,
                'groundtruth': ref,
                'prediction': gen,
                'x_name': x_name,
                'y_name': y_name,
                'command': command,
                'id': id
                }
        except Exception:
            return False

    def extract_text(self, input_string):
        pattern = r'[A-Za-z\s]+'
        matches = re.findall(pattern, input_string)
        return ''.join(matches) 
       
    def compare_vega_lite_specifications(self, specification1, specification2):
        similarity_score = 0
        for key, value in specification1.items():
            if key in specification2:
                if value == specification2[key]:
                    similarity_score += 1
        return similarity_score

    def get_grammar_similarity_score(self, vega_lite_spec1, vega_lite_spec2):
        keys1 = set(vega_lite_spec1.keys())
        keys2 = set(vega_lite_spec2.keys())
        
        common_keys = keys1.intersection(keys2)
        
        accuracy_score = len(common_keys) / len(keys1) if len(keys1) != 0 else 0

        return accuracy_score * 100 

    def vega_lite_similarity_score(self, schema1, schema2):
        print("Checking vegalite similarity score")
        # Parse the JSON schemas
        
        try:
            schema1 = json.loads(schema1)
            print("Parsed JSON schema of Ground truth")
        except:
            print("Failed parsing json schema of the Ground truth")
            schema1 = self.convert_single_quotes_to_double_quotes(schema1) 
            schema1 = json.loads(schema1)
            print(schema1)  
        
        try:
            schema2 = json.loads(schema2)
            print("Parsed JSON schema of the LLM Vegalite")
        except:
            print("Failed parsing json schema of the LLM Vegalite")
            schema2 = self.convert_single_quotes_to_double_quotes(schema2)
            schema2 = json.loads(schema2)
            print(schema2)  

        # Extract the properties from each schema
        try:
            properties1 = set(schema1.keys())
            properties2 = set(schema2.keys())
            

            print(properties1)
            print(properties2)
        except:
            print("Keys invalid")    
        try:
            # Calculate the intersection and union of properties
            intersection = len(properties1.intersection(properties2))
            union = len(properties1.union(properties2))

            # Calculate the Jaccard similarity coefficient
            similarity_score = intersection / union
            return similarity_score
        except:
            print("Could not calculate vega similarity score")

    def convert_single_quotes_to_double_quotes(self, json_string):
        # Use str.replace to replace all single quotes with double quotes for property names and values
        json_string = json_string.replace("'", "\"")
        return json_string


    def get_evaluation_scores(self, reference, hypothesis, measure_bleu=False, measure_rouge=False):
        bleu_score = None
        rouge_score = None

        if measure_bleu:
            bleu_score = sentence_bleu(reference, hypothesis)
        if measure_rouge:
            rouge = Rouge()
            scores = rouge.get_scores(hypothesis, reference)
            rouge_score = scores[0]  # Assuming one reference

        return bleu_score, rouge_score

    def get_cosine_similarity(self, reference, hypothesis):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Combine reference and hypothesis into a list
        documents = [reference, hypothesis]

        # Create a CountVectorizer to convert text to vectors
        vectorizer = CountVectorizer().fit_transform(documents)

        # Calculate cosine similarity
        cosine_sim = cosine_similarity(vectorizer)

        # Extract the similarity score
        similarity_score = cosine_sim[0, 1]

        # print(f"Cosine Similarity Score: {similarity_score:.4f}")
        return similarity_score
    
    def get_jaccard_similarity(self, str1, str2):
        # Tokenize the strings
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())

        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        # Avoid division by zero
        jaccard_similarity = intersection / union if union != 0 else 0.0

        return jaccard_similarity


    def get_json_from_llm_response(self, result):
        extracted_json = False
        try:
            extracted_json = self.extract_text_between_backticks(result)
            try:
                # Try to parse json
                extracted_json = json.loads(extracted_json)
                return extracted_json
            except Exception as e:
                return False
        except Exception as e:
            return False

        return extracted_json
    
    def calculate_data_accuracy(self, generated_spec, ground_truth_spec):
        generated_data = generated_spec.get('data', {})
        ground_truth_data = ground_truth_spec.get('data', {})
            
        if generated_data == ground_truth_data:
            return 1.0
        else:
            matched_data_count = sum(1 for key in generated_data if key in ground_truth_data and generated_data[key] == ground_truth_data[key])
            total_data_keys = max(len(generated_data), len(ground_truth_data))
            return matched_data_count / total_data_keys
        
    def calculate_type_accuracy(self, generated_spec, ground_truth_spec):
        generated_mark = generated_spec.get('mark', '')
        ground_truth_mark = ground_truth_spec.get('mark', '')
            
        return 1.0 if generated_mark == ground_truth_mark else 0.0

    def calculate_encoding_accuracy(self, generated_spec, ground_truth_spec):
        generated_encoding = generated_spec.get('encoding', {})
        ground_truth_encoding = ground_truth_spec.get('encoding', {})
            
        matched_encoding_count = sum(1 for key in generated_encoding if key in ground_truth_encoding and generated_encoding[key] == ground_truth_encoding[key])
        total_encoding_keys = max(len(generated_encoding), len(ground_truth_encoding))
        return matched_encoding_count / total_encoding_keys if total_encoding_keys != 0 else 0.0    

    def openai_compare_vegalite_specs(self, generated_spec, ground_truth_spec):

        data_accuracy = self.calculate_data_accuracy(generated_spec, ground_truth_spec)
        type_accuracy = self.calculate_type_accuracy(generated_spec, ground_truth_spec)
        encoding_accuracy = self.calculate_encoding_accuracy(generated_spec, ground_truth_spec)
        overall_accuracy = (data_accuracy + type_accuracy + encoding_accuracy) / 2 * 100

        result = [
            {"data_accuracy": data_accuracy},
            {"type_accuracy": type_accuracy},
            {"overall_accuracy": overall_accuracy}
        ]
        return overall_accuracy

        return json.dumps(result, indent=2)

    def create_vegalite_data(self, values_x, values_y):
        if len(values_x) != len(values_y):
            raise ValueError("Input arrays should have the same length")

        data_values = []
        for x, y in zip(values_x, values_y):
            data_values.append({"x_data": str(x), "y_data": str(y)})

        return {"values": data_values}

    def get_ssi_scores(self, path_image_1, path_image_2):
        
        image1 = cv2.imread(path_image_1)
        image2 = cv2.imread(path_image_2)

        # Resize images to have the same dimensions
        width = min(image1.shape[1], image2.shape[1])
        height = min(image1.shape[0], image2.shape[0])
        image1 = cv2.resize(image1, (width, height))
        image2 = cv2.resize(image2, (width, height))

        # Convert the images to grayscale
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)



        # Compute the Structural Similarity Index (SSI)
        ssi_index, _ = ssim(gray_image1, gray_image2, full=True)

        # Compute the Mean Squared Error (MSE)
        mse = ((gray_image1 - gray_image2) ** 2).mean()

        return ssi_index * 100 

    def load_and_preprocess_image(self, image_path, target_size=(256, 256)):
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

    def get_lpips_scores(self,image1_path,image2_path,net="alex"):
        # Load and preprocess the images as tensors
        image1_tensor = self.load_and_preprocess_image(image1_path)
        image2_tensor = self.load_and_preprocess_image(image2_path)
        # Continue with the LPIPS calculation code from the previous example
        lpips_model = lpips.LPIPS(net=net)
        lpips_distance = lpips_model(image1_tensor, image2_tensor).item()
        return 100-lpips_distance    
