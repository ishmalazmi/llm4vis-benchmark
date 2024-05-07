class Task:
    vizQueryGeneration = {
        "name" : "Visualization Query Generation",
        "short_name" : "Viz. Query Gen.",
        "description" : "This task prompts the LLM to generate Vega-lite/Vega-zero Schema/JSON Specification for the NL Query"
    }

    vizGeneration = {
        "name" : "Visualization Generation",
        "short_name" : "Viz. Gen.",
        "description" : "This task prompts the LLM to generate Vega-lite/Vega-zero python code to generate visualization with Vega parser"
    }

    captioning = {
        "name" : "Captioning",
        "short_name" : "Captioning",
        "description" : "This task prompts the LLM to generate a caption for the visualization (Viz. Query or Image)"
    }

    codeGeneration = {
        "name" : "Code Generation",
        "short_name" : "Code Generation",
        "description" : "This task prompts the LLM to generate Python code to generate a visualization from the given dataset"
    }