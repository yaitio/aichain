from _client import BaseClient, YandexClient, OpenAIClient, StabilityAIClient, AnthropicClient, GoogleAIClient, MistralAIClient
from copy import deepcopy
import base64
import mimetypes

class Role:
    name: str
    description: str
    model: str
    instructions: dict
    options: dict
    api_key: str
    variables: list

    def __new__(cls, name: str, description: str, modelname: str, instructions:list):
        match modelname:            
            case "yandexgpt" | "yandexgpt-lite" | "summarization" as model:
                instance = Role_YandexGPT
            case "gpt-3.5-turbo" | "gpt-4" | "gpt-4-turbo" | "dall-e-3" | "gpt-4o" | "gpt-4o-mini" as model:
                instance = Role_OpenAI
            case "core" | "sd3" | "sd3-turbo":
                instance = Role_StabilityAI
            case "claude-3-opus-20240229" | "claude-3-sonnet-20240229" | "claude-3-haiku-20240307" | "claude-3-5-sonnet-20240620" as model:
                instance = Role_AnthropicAI
            case "gemini-pro" | "gemini-pro-vision" as model:
                instance = Role_GoogleAI
            case "mistral-large-latest" | "mistral-medium-latest" | "mistral-small-latest" as model:
                instance = Role_MistralAI                   
            case _:
                instance = cls
        return super().__new__(instance)
        
    def __init__(self, name: str, description: str, modelname: str, instructions:list ) -> None:
        self.model = modelname
        self.instructions = deepcopy(instructions)
        self.name = name
        self.description = description
        self.variables = []

        for instruction in self.instructions:
            self.variables = self.variables + self.__getVariables(instruction['content'])

    def __getVariables(self, content:str|list) -> list:
        variables = []        
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    for variable in item['text'].split():
                        if '{' in variable and '}' in variable:
                            variable_name = variable.strip('{},.?!')
                            variables.append(variable_name)
        else:
            for variable in content.split():
                if '{' in variable and '}' in variable:
                    variable_name = variable.strip('{},.?!')
                    variables.append(variable_name)        

        return variables

    def castInstructions(self, options: dict):
        for instruction in self.instructions:
            if isinstance(instruction["content"], list) and instruction["role"] == "user":
                updated = instruction.pop('content')
                for item in updated:                   
                    for key in item.keys():
                        item[key] = self.__setVariables(item[key], options)
                instruction['content'] = updated
            else:
                updated = self.__setVariables(instruction.pop('content'), options)
                instruction['content'] = updated

    def __setVariables(self, input_string, variables):
        for key in variables:
            input_string = input_string.replace("{" + key + "}", str(variables[key]))
        return input_string

    def run(self, api_key = None, options = None):
        print("Running Role")

class Role_OpenAI(Role):
    def visionInstructions(self, instruction:list):
        for item in instruction:
            if "text" in item: item["type"] = "text"
            if "image" in item: 

                item["type"] = "image_url"
                source = item.pop('image')

                if 'data' in source and 'media_type' in source:
                    item["image_url"] = {"url":f"data:{source['media_type']};base64,{source['data']}"}
                elif 'file' in source:                    
                    try:
                        mimetypes.init()
                        with open(source['file'], "rb") as image_file:                    
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            image_media_type = mimetypes.guess_type(source['file'])[0]
                            item["image_url"] = {"url":f"data:{image_media_type};base64,{image_data}"}

                    except Exception as e:
                        print("An error occurred:", str(e))
                        return None
                    
                elif 'url' in source:
                    item["image_url"] = {"url":source['url']}
                else:
                    print(f"An error occurred: Unknown source type for image, file, data or url expected.")
                    return None

    def run(self, api_key: str,  options: dict = None):    
        #Apply variables to instructions
        if options: self.castInstructions(options)

        match self.model:
            case "gpt-3.5-turbo"as model:                
                openaiChatCompletion = OpenAIClient(api_key = api_key, model = self.model).completion(messages = self.instructions)
                return openaiChatCompletion['choices'][0]['message']['content'] if openaiChatCompletion is not None else None

            case "gpt-4" | "gpt-4-turbo" | "gpt-4o" | "gpt-4o-mini" as model: #Extend instructions for Vision models
                for instruction in self.instructions:
                    if isinstance(instruction["content"], list) and instruction["role"] == "user":
                        self.visionInstructions(instruction["content"]);
               
                openaiChatCompletion = OpenAIClient(api_key = api_key, model = self.model).completion(messages = self.instructions)
                return openaiChatCompletion['choices'][0]['message']['content'] if openaiChatCompletion is not None else None
                     
            case "dall-e-3":
                openaiImageGeneration = OpenAIClient(api_key = api_key, model = self.model).generation(messages = self.instructions, options = {"response_format" : "b64_json"})
                return openaiImageGeneration if openaiImageGeneration is not None else None
            case _:
                return None

class Role_MistralAI(Role):
    def run(self, api_key: str, options: dict = None):

        for instruction in self.instructions: #Extend instructions for Vision models
            if isinstance(instruction["content"], list) and instruction["role"] == "user":
                print("An error occurred: this model doesn't support vision functions")
                return None

        #Apply variables to instructions
        if options: self.castInstructions(options)

        mistralChatCompletion = MistralAIClient(api_key = api_key, model = self.model).completion(messages = self.instructions)
        return mistralChatCompletion['choices'][0]['message']['content'] if mistralChatCompletion is not None else None

class Role_AnthropicAI(Role):
    def visionInstructions(self, instruction:list):
        for item in instruction:
            if "text" in item: item["type"] = "text"
            if "image" in item:

                source = item.pop('image')
                item["type"] = "image"

                if 'data' in source and 'media_type' in source:
                    item["source"] = {
                        "type": "base64",
                        "media_type": source['media_type'],
                        "data": source['data']
                    }
                elif 'file' in source:                    
                    try:
                        mimetypes.init()
                        with open(source['file'], "rb") as image_file:                    
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            image_media_type = mimetypes.guess_type(source['file'])[0]
                            
                            item["source"] = {
                                "data":image_data,
                                "type": "base64",
                                "media_type": image_media_type
                            }

                    except Exception as e:
                        print("An error occurred:", str(e))
                        return None
                    
                elif 'url' in source:
                    image_client = BaseClient(source['url'])
                    image = image_client.download()

                    if image is not None:
                        item["source"] = {
                            "data": base64.b64encode(image['data']).decode('utf-8'),
                            "media_type": image['media_type'],
                            "type": "base64",
                        }

                else:
                    print(f"An error occurred: Unknown source type for image, file, data or url expected.")
                    return None              

    def run(self, api_key: str, options: dict = None):
        #Apply variables to instructions
        if options: self.castInstructions(options)

        #Extend instructions for Vision models
        for instruction in self.instructions:
            if isinstance(instruction["content"], list) and instruction["role"] == "user":
                self.visionInstructions(instruction["content"]);
    
        AnthropicAICompletion = AnthropicClient(api_key = api_key, model = self.model).completion(messages = self.instructions)
        return AnthropicAICompletion['content'][0]['text'] if AnthropicAICompletion is not None else None

class Role_GoogleAI(Role):

    def visionInstructions(self, instruction:list):
        for item in instruction:
            if "image" in item:
                source = item.pop('image')
                if 'data' in source and 'media_type' in source:
                    item["inline_data"] = {
                        "mime_type": source['media_type'],
                        "data": source['data']
                    }
                elif 'file' in source:                    
                    try:
                        mimetypes.init()
                        with open(source['file'], "rb") as image_file:                    
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            image_media_type = mimetypes.guess_type(source['file'])[0]
                            
                            item["inline_data"] = {
                                "data":image_data,
                                "mime_type": image_media_type
                            }

                    except Exception as e:
                        print("An error occurred:", str(e))
                        return None                    
                elif 'url' in source:
                    image_client = BaseClient(source['url'])
                    image = image_client.download()

                    if image is not None:
                        item["inline_data"] = {
                            "data": base64.b64encode(image['data']).decode('utf-8'),
                            "mime_type": image['media_type']
                        }
                else:
                    print(f"An error occurred: Unknown source type for image, file, data or url expected.")
                    return None 
                
    def run(self, api_key: str, options: dict = None):
        #Apply variables to instructions
        if options: self.castInstructions(options)

        #Extend instructions for Vision models
        for instruction in self.instructions:
            if isinstance(instruction["content"], list) and instruction["role"] == "user":
                self.visionInstructions(instruction["content"]);

        GoogleAICompletion = GoogleAIClient(api_key = api_key, model = self.model).completion(messages = self.instructions)
        return GoogleAICompletion['candidates'][0]['content']['parts'][0]['text'] if GoogleAICompletion is not None else None

class Role_YandexGPT(Role):
    def run(self, api_key: str, options: dict):
        if options is None or 'YaFolderID' not in options: 
            print(f"An error occurred: You must provide a YaFolderID in the options for ant Yandex model.")
            return None
        if options: self.castInstructions(options)
        
        yandexChatCompletion = YandexClient(api_key = api_key, catalogId = options['YaFolderID'], model = self.model).completion(messages = self.instructions)
        return yandexChatCompletion['result']['alternatives'][0]['message']['text'] if yandexChatCompletion is not None else None

class Role_StabilityAI(Role):
    def run(self, api_key: str, options: dict):
        if options: self.castInstructions(options)
        
        stabilityaiImageGeneration = StabilityAIClient(api_key = api_key, model = self.model).generation(messages = self.instructions)
        return stabilityaiImageGeneration if stabilityaiImageGeneration is not None else None