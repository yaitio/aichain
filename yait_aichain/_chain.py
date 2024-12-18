from ._client import BaseClient, YandexClient, OpenAIClient, StabilityAIClient, AnthropicClient, GoogleAIClient, MistralAIClient
from ._role import Role

class Chain:
    name: str
    description: str
    model: str    
    api_key: str
    options: dict
    variables: list

    def __init__(self, name: str, modelname: str, pitch: list, context: bool = False, description: str = None) -> None:
        self.name = name
        self.pitch = []
        self.model = modelname
        self.description = description
        self.variables = []
        self.context = None
        self.options = {}

        for p in pitch:
            a = Role('','', self.model, p['messages'])
            if 'output' in p:
                self.pitch.append({"role":a, "output":p['output']})
            else:
                self.pitch.append({"role":a, "output":None})
            self.variables = self.variables + a.variables

        print(self.variables)

    def run(self, api_key:str, options: dict) -> None:
        
        for key in options.keys():
            a = options[key]
            print(a)
            self.options[key] = a

        for pitch in self.pitch:
            if pitch['output'] is not None:
                self.options[pitch['output']] = pitch['role'].run(api_key = api_key, options = self.options)
                print(self.options[pitch['output']])
            else:
                print(pitch['role'].run(api_key = api_key, options = self.options))