from agente import AgenteOpenAIFunctions
from langchain.agents import AgentExecutor
from dotenv import load_dotenv

load_dotenv()

pergunta = "Quais os dados da Ana?"

agente = AgenteOpenAIFunctions()
executor = AgentExecutor(agent=agente.agente,
                         tools=agente.tools,
                         verbose=True)

resposta = executor.invoke({"input": pergunta})
print(resposta)