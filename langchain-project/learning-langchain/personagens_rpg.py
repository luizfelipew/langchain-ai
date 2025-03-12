from langchain.prompts import PromptTemplate

# Criando um template para descrição de personagens de RPG
prompt_template = PromptTemplate.from_template(
    "Crie um personagem de RPG. Classe: {classe}, Raça: {raca}, Habilidade principal: {habilidade}."
)

# Usando o template para gerar a descrição de um mago elfo com magia elemental
descricao_personagem = prompt_template.format(classe="Mago", raca="Elfo", habilidade="Magia Elemental")
print(descricao_personagem)