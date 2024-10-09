from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser
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

cadeia_cidade = LLMChain(prompt=modelo_cidade, llm=llm)
cadeia_restaurantes = LLMChain(prompt=modelo_restaurantes, llm=llm)
cadeia_cultural = LLMChain(prompt=modelo_cultural, llm=llm)

cadeia = SimpleSequentialChain(chains = [cadeia_cidade, cadeia_restaurantes, cadeia_cultural],
                               verbose=True)

resultado = cadeia.invoke("praias")
print(resultado)