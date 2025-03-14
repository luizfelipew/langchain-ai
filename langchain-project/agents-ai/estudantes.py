from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import BaseTool
import pandas as pd
import json
import os
from dotenv import load_dotenv

load_dotenv()

def buscar_dados_estudante(estudante):
    dados = pd.read_csv("agents-ai/documentos/estudantes.csv")
    dados_com_esse_estudante = dados[dados["USUARIO"] == estudante]
    if dados_com_esse_estudante.empty:
        return {}
    return dados_com_esse_estudante.iloc[:1].to_dict()

class ExtratorDeEstudante(BaseModel):
    estudante:str = Field("Nome do estudante informado, sempre em letras minúsculas. Exemplo: joão, carlos, joana, carla")


class DadosDeEstudante(BaseTool):
    name = "DadosDeEstudante"
    description = """Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico"""

    def _run(self, input: str) -> str:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"))
        
        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)

        template = PromptTemplate(template="""Você deve analisar a {input} e extrair o nome de usuário informado.
                       Formato de saída 
                       {formato_saida}""",
                       input_variables=["input"],
                       partial_variables={"formato_saida" : parser.get_format_instructions()})
        
        cadeia = template | llm | parser
        resposta = cadeia.invoke({"input": input})
        estudante = resposta['estudante']
        estudante = estudante.lower()
        dados = buscar_dados_estudante(estudante)
        return json.dumps(dados)
