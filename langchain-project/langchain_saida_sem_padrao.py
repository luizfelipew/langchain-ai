from langchain_openai import ChatOpenAI
from langchain.output_parsers import DatetimeOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY"))

parseador_saida = DatetimeOutputParser()
modelo_data = """Responda a pergunta do usuário: 
    {pergunta}

    {formato_saida}
"""

prompt = PromptTemplate.from_template(
    modelo_data,
    partial_variables={"formato_saida": parseador_saida.get_format_instructions()},
)

PromptTemplate(
    input_variables=['question'], 
    partial_variables={'formato_saida': "Write a datetime string that matches the following pattern: '%Y-%m-%dT%H:%M:%S.%fZ'.\n\nExamples: 0668-08-09T12:56:32.732651Z, 1213-06-23T21:01:36.868629Z, 0713-07-06T18:19:02.257488Z\n\nReturn ONLY this string, no other words!"}, template='Answer the users question:\n\n{question}\n\n{format_instructions}')

chain = prompt | llm | parseador_saida

resposta = chain.invoke({"pergunta": "Quando a bitcoin foi fundada?"})

print(resposta)