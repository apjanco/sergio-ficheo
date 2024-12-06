from langchain_ollama.chat_models import ChatOllama
from langchain.schema import HumanMessage
import os

def summarize_text(prompt, text):
    llm_model = "mistral:instruct"  # Replace with your model name
    llm = ChatOllama(model=llm_model, format="json", num_ctx=1000, temperature=0)
    
    full_prompt = f"{prompt}: {text}"
    
    print(f"Input text:\n{full_prompt}")
    
    print("Invoking LLM...")
    try:
        response = llm.invoke([HumanMessage(content=full_prompt)])
        if response:
            return response.content
        else:
            return "No valid response from LLM."
    except Exception as e:
        return f"Error invoking LLM: {e}"

if __name__ == "__main__":
    prompt = "Please summarize the following text in Spanish"
    text = """Venta - Esclavo. En el pueblo de San Francisco de Quibdo, capital de la V Provincia Decitora, a siete de marzo de mil eochocien tos ocho, ante mi, Don Carlos de Siguizte, Capitolo de Rusengo, entre un Bon Cartu de Eladite, capitán de infantería, gobernador político y militar de estas provinciados, y de los testigos con quienes actuo sin impedimentos único. Esno pareció l y Agustin Dozo, vecino de esta province, a quien certifico conozco y | vendé y do en real y perpetua enagenación desde ahora y para siem- pre jamas petrónaferla | de este mismo veandario que seg | para la dicha y sus herederos y | que ca d'aprece va.

Para la diena y sus mercederos (1) que es a saber un negro su propio esclavo llamado Valentin, el que se vende por libre de otra yenta censo, empiezo ni hipotesa tacita ni expresa y por tal lo asegura en cantidad de trescientos y cinquenta pesos de plata que confierto haber recibido los ciento ochenta al contado, y el resto de ciento setenta a satisfacerlos dentro del término y plazo de cinco meses en la especie de oro en polvo y por no ser de presente el.

El gobierno certifica que el documento es auténtico y no tiene alteraciones. Y vendido que los trescientos cinquenta pesos de plata es el justo precio y verdad valor de dicho esclavo que no vale más, y caso de que mas valga y su demasia en poca o mucho suma le hace gracia y donación a la compradora x los suyos buen.

La prueba que ha hecho el testigo, Don Bonifacio, es satisfactoria. Salazar a nombre de Agustin Daza, cuatro, por el derecho de alcañoval al 4, 4 deducidos de doscientos y quarenta en que vendro Marcelino Valencia un negrito llamado Manuel Quibdo 3 de marzo de 1808. Jose Maria Valencia.

Ley del ordenamiento real fechado en Cortes de Alcalá de Henares y los cuatro años en ella declarados para repetir.
"""
    summary = summarize_text(prompt, text)
    print(f"\nLLM output:\n{summary}")