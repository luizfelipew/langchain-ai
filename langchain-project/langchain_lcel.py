from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
set_debug(True)

# Extends da classe BaseModel
class Destino(BaseModel):
    cidade = Field("cidade a visitar")
    motivo = Field("motivo pelo qual é interessante visitar essa cidade")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY"))

parseador = JsonOutputParser(pydantic_object=Destino)

modelo_cidade = PromptTemplate(
    template = """Sugira uma cidade dado meu interesse por {interesse}.
    {formatacao_de_saida}
    """,

    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": parseador.get_format_instructions()}
)

modelo_restaurantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}"
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades em locais culturais em {cidade}"
)

parte1 = modelo_cidade | llm | parseador
parte2 = modelo_restaurantes | llm | StrOutputParser()
parte3 = modelo_cultural | llm | StrOutputParser()

cadeia = (parte1 | {
    "restaurante" : parte2, 
    "locais_culturais" : parte3 
})

resultado = cadeia.invoke({"interesse" : "praias"})
print(resultado)