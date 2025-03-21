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
pergunta = "Quais os dados da USP?"
pergunta = "Quais os dados da uniCAMp?"
pergunta = "Dentre USP e UFRJ, qual você recomenda para a acadêmica Ana?"
pergunta = "Dentre uni camp e USP, qual você recomenda para a Ana?"
pergunta = "Quais faculdades com melhroes chances para a Ana entrar?"
pergunta = "Dentro todas as faculdades existentes, quais Ana possui mais chances de entrar?"
pergunta = "Além das faculdades favoritas da Ana existem outras faculdades. Considere elas também. Quais Ana possui mais chance de entrar?"

agente = AgenteOpenAIFunctions()
executor = AgentExecutor(agent=agente.agente,
                         tools=agente.tools,
                         handle_parsing_errors=True,
                         verbose=True)

resposta = executor.invoke({"input": pergunta})
print(resposta)