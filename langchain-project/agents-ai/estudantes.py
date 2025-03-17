from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import BaseTool
import pandas as pd
from typing import List
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

class Nota(BaseModel):
    area:str = Field("nome da área de conhecimento")
    nota:float = Field("nota na área de conhecimento")
class PerfilAcademicoDeEstudante(BaseModel):
    nome:str = Field("nome do estudante")
    ano_de_conclusao:int = Field("ano de conclusão")
    notas:List[Nota] = Field("Lista de notas das disciplinas e áreas de conhecimento.")
    resumo:str = Field("resumo das principais características desse estudante de forma a torná-lo único e um ótimo potencial estudante para faculdades. Exemplo: só esse estudante tem bla bla bla")
class PerfilAcademico(BaseTool):
    name = "PerfilAcademico"
    description = """Cria um perfil acadêmico de um estudante.
    Esta ferramenta requer como entrada todos os dados do estudante."""

    def _run(self, input: str) -> str:
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"))
        
        parser = JsonOutputParser(pydantic_object=PerfilAcademicoDeEstudante)
        template = PromptTemplate(template="""- Formate o estudando para o seru perfil acadêmico
                                  - Com os dados, identifique as opções de universidade e sugeridas com nome da univerdade e cursos compatíveis com interesse do aluno
                                  - Destaqui o perfil do aluno dando enfase principalmente naquilo que faz sentido paras instituições de interesse do aluno
                                
                                  Persona: você é uma consultora de carreira e precisa indicar com detalhes, riqueza, 
                                                        mas direta ao ponto para estudante as opções e consequências possiveis.
                                  Informações atuais:
                                  
                                  {dados_do_estudante}
                                  {formato_de_saida}                       
                                  """, input_variables=["dados_do_estudante"],
                                  partial_variables={"formato_de_saida": parser.get_format_instructions()})
        
        cadeia = template | llm | parser
        resposta = cadeia.invoke({"dados_do_estudante": input})
        return resposta