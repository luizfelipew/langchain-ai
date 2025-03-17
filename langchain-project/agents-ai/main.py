from agente import AgenteOpenAIFunctions
from langchain.agents import AgentExecutor
from dotenv import load_dotenv

load_dotenv()

pergunta = "Quais os dados da Ana?"
pergunta = "Quais os dados da Bianca?"
pergunta = "Quais os dados da Ana e Bianca?"
# pergunta = "Crie um perfil acadêmico para Ana!"
pergunta = "Compare o perfil acadêmico da Ana com o da Bianca!"
pergunta = "Tenho sentido Ana desanimada com o cursos de Matemática. Seria uma boa parear ela com a Bianca?"
pergunta = "Tenho sentido Ana desanimada com o cursos de Matemática. Seria uma boa parear ela com o Marcos?"

agente = AgenteOpenAIFunctions()
executor = AgentExecutor(agent=agente.agente,
                         tools=agente.tools,
                         verbose=True)

resposta = executor.invoke({"input": pergunta})
print(resposta)