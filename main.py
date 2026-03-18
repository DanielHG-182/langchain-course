import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Hello from langchain-course!")

    information ="""Claudia Sheinbaum Pardo (Ciudad de México, 24 de junio de 1962) es una política, científica y académica mexicana. Es la Presidenta de México desde el 1 de octubre de 2024, siendo la primera mujer en la historia de su país en ejercer el cargo.[6]​[7]​[8]​

    Antes de asumir la presidencia de México, Sheinbaum participó activamente en el servicio público, ocupando diversos cargos de liderazgo. Fue jefa de Gobierno de la Ciudad de México de 2018 a 2023, convirtiéndose en la primera alcaldesa electa de la ciudad.[9]​[10]​ Su administración priorizó la seguridad urbana, las iniciativas ambientales y los programas sociales, incluidas notables expansiones en el transporte público y los sistemas de apoyo educativo. Su mandato también incluyó respuestas a desafíos complejos, como el colapso de la Línea 12 del Metro de la Ciudad de México y la gestión de la pandemia de COVID-19, que le valieron tanto reconocimiento como críticas.

    La trayectoria política de Sheinbaum comenzó en el Partido de la Revolución Democrática (PRD), pero más tarde se alineó con el Movimiento Regeneración Nacional (Morena), uniéndose al partido de Andrés Manuel López Obrador en 2014.[11]​

    En 2024, tras ganar la nominación de Morena, se aseguró la presidencia con una victoria arrolladora sobre la candidata opositora Xóchitl Gálvez.[12]​ A fines de ese año, Forbes la situó como la cuarta mujer más poderosa del mundo,[13]​ mientras que Time, la reconoció como una de las 100 líderes mundiales más influyentes a favor del medio ambiente.[14]​ En abril de 2025, la misma publicación la incluyó en su lista anual de las 100 personas más influyentes del mundo.[15]​

    Durante su gobierno, la presidenta Sheinbaum promulgó una serie de reformas constitucionales con el apoyo de su supermayoría legislativa, incluyendo la consagración de programas sociales en la Constitución,[16]​ la reversión de aspectos clave de la reforma energética de 2013 para fortalecer el control estatal sobre el sector energético[17]​ y el mandato de que el salario mínimo aumente por encima de la tasa de inflación.[18]​ """

    summary_template=""" Given the information {information} about this person. I want you to create: 
    1.A short summary
    2.Three interesting facts bout this person.
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"],template=summary_template)

    llm = ChatOpenAI(temperature=0,model="gpt-4.1-nano")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information":information})

    print(response.content)



if __name__ == "__main__":
    main()
