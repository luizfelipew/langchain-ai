from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY"))

parte1 = PromptTemplate.from_template("Analisar a queixa: {queixa}") | llm | StrOutputParser()
parte2 = PromptTemplate.from_template("Avaliar sentimento da queixa: {resultado_analise}") | llm | StrOutputParser()
parte3 = PromptTemplate.from_template("Formular resposta: {sentimento}") | llm | StrOutputParser()

cadeia = (
    {"queixa": RunnablePassthrough()}
    | RunnablePassthrough.assign(resultado_analise=parte1)
    | RunnablePassthrough.assign(sentimento=parte2)
    | parte3
)

queixa_texto = "Hoje comprei um telefone novo, modelo X com 256 GB e flip. No entanto, o produto apresentou defeito na dobradiça e não permanece fechado. O suporte não me atende e estou super arrependido."
resultado = cadeia.invoke({"queixa": queixa_texto})

print(resultado)