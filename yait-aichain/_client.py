import urllib3
import json
import base64

from _constants import (
    DEFAULT_TIMEOUT,
    MAX_RETRY_DELAY,
    DEFAULT_MAX_RETRIES,
    INITIAL_RETRY_DELAY,
    DEFAULT_CHATMODEL_TEMPERATURE,
    DEFAULT_MAX_TOKENS
)

class BaseClient:
    _client: urllib3.PoolManager
    _base_url: str
    _tiemout: urllib3.Timeout
    _retries: urllib3.Retry

    def __init__(self, base_url: str, timeout: urllib3.Timeout = DEFAULT_TIMEOUT, retries: urllib3.Retry = DEFAULT_MAX_RETRIES):
        self._base_url = base_url
        self._client = urllib3.PoolManager(timeout=timeout, retries=retries)

    def get(self, path: str, headers: dict = None):
        response = self._client.request('GET', self._base_url + path, headers = headers)
        if response.status == 200:
            return response.data.decode('utf-8')
        else:
            return None


    def download(self, headers: dict = None):
        response = self._client.request('GET', self._base_url, headers = headers)

        if response.status == 200:
            return {
                "data": response.data,
                "media_type": response.headers['Content-Type']
            }

        else:
            return None

    def post(self, path: str, data: dict = None, formdata: dict = None, headers: dict = None):            
            try:
                if data:
                    response = self._client.request('POST', self._base_url + path, body = json.dumps(data), headers = headers)
                elif formdata:
                    response = self._client.request('POST', self._base_url + path, fields = formdata, headers = headers)
                else:
                    return None

                if response.status == 200:
                    return response.data
                else:
                    e = json.loads(response.data.decode('utf-8'))
                    print(f"An error occurred: {str(e)}")                    
                    return None
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                return None
    
class OpenAIClient(BaseClient):
    _model: str

    def __init__(self, api_key: str, model: str):
        super().__init__(
            base_url = 'https://api.openai.com',
            timeout = DEFAULT_TIMEOUT,
            retries = DEFAULT_MAX_RETRIES
        )
        self._api_key = api_key
        self._model = model

    def completion(self, messages: list):

        body = {
            "model": self._model,
            "messages": messages,
            "temperature": DEFAULT_CHATMODEL_TEMPERATURE
        }

        responce = self.post(
            '/v1/chat/completions',
            data = body,
            headers = {
                'Authorization': 'Bearer ' + self._api_key,
                'Content-Type': 'application/json'
            }
        )

        return json.loads(responce.decode('utf-8')) if responce is not None else None

    def generation(self, messages: object, options: dict = None):

        response_format =  options['response_format'] if (options is not None) and ('response_format' in options) else "url"

        body = {
            "prompt" : messages[0]['content'],
            "n" : 1,
            "model" : self._model,
            "size" : "1024x1024",
            "response_format" : response_format
        }

        responce = self.post(
            '/v1/images/generations',
            data = body,
            headers = {
                'Authorization': 'Bearer ' + self._api_key,
                'Content-Type': 'application/json'
            }
        )

        if responce is not None: 
            if response_format == "b64_json":                
                return {"type" : "image", "data" : base64.b64decode(json.loads(responce)["data"][0]["b64_json"])}
            else:
                return {"type" : "url", "data" : json.loads(responce.decode('utf-8'))["data"][0]["url"]}                
        else: 
            None    

    def embeddings(self, input: str):

        response = self.post(
            '/v1/embeddings',
            data = { "model": self._model, "input": input},
            headers = {
                'Authorization': 'Bearer ' + self._api_key,
                'Content-Type': 'application/json'
            }
        )

        return json.loads(response.decode('utf-8'))['data'][0]['embedding'] if response is not None else None
    
class MistralAIClient(BaseClient):
    _model: str

    def __init__(self, api_key: str, model: str):
        super().__init__(
            base_url = 'https://api.mistral.ai',
            timeout = DEFAULT_TIMEOUT,
            retries = DEFAULT_MAX_RETRIES
        )
        self._api_key = api_key
        self._model = model

    def completion(self, messages: list):

        body = {
            "model": self._model,
            "messages": messages,
            "temperature": DEFAULT_CHATMODEL_TEMPERATURE
        }

        responce = self.post(
            '/v1/chat/completions',
            data = body,
            headers = {
                'Authorization': 'Bearer ' + self._api_key,
                'Content-Type': 'application/json'
            }
        )

        return json.loads(responce.decode('utf-8')) if responce is not None else None

class YandexClient(BaseClient):
    _model: str

    def __init__(self, api_key: str, catalogId: str, model: str):
        super().__init__(
            base_url = 'https://llm.api.cloud.yandex.net',
            timeout = DEFAULT_TIMEOUT,
            retries = DEFAULT_MAX_RETRIES
        )
        self._api_key = api_key
        self._model = model
        self.catalogId = catalogId

    def __prepareBody(self, messages):
        for message in messages:
            message['text'] = message.pop('content')

        return {
            "modelUri": f"gpt://{self.catalogId}/{self._model}/latest",
            "completionOptions": {
                "stream": False,
                "temperature": DEFAULT_CHATMODEL_TEMPERATURE,
                "maxTokens": DEFAULT_MAX_TOKENS
            },
            "messages": messages
        }

    def completion(self, messages: list):        

        responce = self.post(
            '/foundationModels/v1/completion',
            data = self.__prepareBody(messages),
            headers = {
                'Authorization': 'Api-Key ' + self._api_key,
                'Content-Type': 'application/json'
            }
        )
        
        return json.loads(responce.decode('utf-8')) if responce is not None else None

class StabilityAIClient(BaseClient):
    _model: str

    def __init__(self, api_key: str, model: str):
        super().__init__(
            base_url = 'https://api.stability.ai/v2beta',
            timeout = DEFAULT_TIMEOUT,
            retries = DEFAULT_MAX_RETRIES
        )
        self._api_key = api_key
        self._model = model

    def generation(self, messages: object, options: dict = None):

        fileds = {
            "prompt": messages[0]['content']
        }

        if self._model == "core":
            url = '/stable-image/generate/core'
        else:
            url = '/stable-image/generate/sd3'
            fileds['model'] = self._model

        
        if options is not None:
            if 'output_format' in options: fileds['output_format'] = options['output_format']
            if 'seed' in options: fileds['seed'] = options['seed']
            if 'style' in options: fileds['style_preset'] = options['style']
            if 'output_format' in options: fileds['output_format'] = options['output_format']
            if 'aspect_ratio' in options: fileds['aspect_ratio'] = options['aspect_ratio']
       
        responce = self.post(
            url,
            formdata = fileds,
            headers = {
                'Authorization': 'Bearer ' + self._api_key,
                'accept': 'image/*'
            }
        )

        if responce is not None:
            return {"type" : "image", "data" : responce}
        else:
            return None

class AnthropicClient(BaseClient):
    _model: str

    def __init__(self, api_key: str, model: str):
        super().__init__(
            base_url = 'https://api.anthropic.com',
            timeout = DEFAULT_TIMEOUT,
            retries = DEFAULT_MAX_RETRIES
        )
        self._api_key = api_key
        self._model = model

    def __prepareBody(self, messages):
                
        systemPrompt = None
        for idx, message in enumerate(messages):
            if message['role'] == 'system':
                systemPrompt = message['content']
                del messages[idx]
                break

        body = {
            "model": self._model,
            "messages": messages,
            "max_tokens": DEFAULT_MAX_TOKENS
        }
        if systemPrompt is not None: body["system"] = systemPrompt                    
        return body

    def completion(self, messages: list):
        responce = self.post(
            '/v1/messages',
            data = self.__prepareBody(messages),
            headers = {
                'x-api-key': self._api_key,
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01'
            }
        )

        return json.loads(responce.decode('utf-8')) if responce is not None else None
    
class GoogleAIClient(BaseClient):
    _model: str

    def __init__(self, api_key: str, model: str):
        super().__init__(
            base_url = 'https://generativelanguage.googleapis.com/v1beta',
            timeout = DEFAULT_TIMEOUT,
            retries = DEFAULT_MAX_RETRIES
        )
        self._api_key = api_key
        self._model = model

    def __prepareBody(self, messages):
        
        systemPrompt = None
        for idx, message in enumerate(messages):
            if message['role'] == 'system':
                systemPrompt = message['content']
                del messages[idx]
                break

        isSystemPromptApplicable = True
        for message in messages:
            if message['role'] == 'user':
                message['parts'] = [] if systemPrompt is None and isSystemPromptApplicable else [{"text": systemPrompt}]
                isSystemPromptApplicable = False
                if message['content'] is not None and type(message['content']) is list:
                    message['parts'] = message.pop('content')                                                           
                else:
                    message['parts'].append({"text": message.pop('content')})
            
            if message['role'] == 'assistant':
                message['role'] = 'model'
                message['parts'] = [{"text": message.pop('content')}]

        return {            
            "contents": message
        }

    def completion(self, messages: list):
        
        responce = self.post(
            f'/models/{self._model}:generateContent?key={self._api_key}',
            data = self.__prepareBody(messages),
            headers = {
                'Content-Type': 'application/json',                
            }
        )

        return json.loads(responce.decode('utf-8')) if responce is not None else None
    
class VoyageAIClient(BaseClient):
    _model: str

    def __init__(self, api_key: str, model: str):
        super().__init__(
            base_url = 'https://api.voyageai.com/v1',
            timeout = DEFAULT_TIMEOUT,
            retries = DEFAULT_MAX_RETRIES
        )
        self._api_key = api_key
        self._model = model

    def embeddings(self, input: str | list, options: dict = None):
        response = self.post(
            '/embeddings',
            data = { 
                "model": self._model, 
                "input": input, 
                "input_type": options['type'] if (options is not None) and ('type' in options) else None
            },
            headers = {
                'Authorization': 'Bearer ' + self._api_key,
                'Content-Type': 'application/json'
            }
        )

        return json.loads(response.decode('utf-8'))['data'][0]['embedding'] if response is not None else None