from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import Tool
from langchain.agents import create_openai_tools_agent
import os
from estudantes import DadosDeEstudante

class AgenteOpenAIFunctions:
    def __init__(self):
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"))
        dados_de_estudante = DadosDeEstudante()

        self.tools = [
            Tool(name = dados_de_estudante.name,
                 func = dados_de_estudante.run,
                 description = dados_de_estudante.description)
        ]

        # criar agente
        prompt = hub.pull("hwchase17/openai-functions-agent")
        print(prompt)
        self.agente = create_openai_tools_agent(llm, self.tools, prompt)
