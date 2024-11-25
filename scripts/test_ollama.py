from langchain_ollama.chat_models import ChatOllama
from langchain.schema import HumanMessage
import os

def main():
    llm_model = "llama3.1:8b"  # Replace with your model name
    llm = ChatOllama(model=llm_model, format="json", num_ctx=8000, temperature=0)
    
    text = (
        """Trinat Superior del Distrito Sta
ministro dicial del Cauca
Buga, Julio catorce del mil ochocientos
Lugares: Libro de la muerte de los novatos y los Vientos. Por ordenación del defensor de Luciano.
JUAN Y FRANCISCO RIVAS, RECO DEL DELITO DE MAL TRATAMIENTO DE OTRA EN BASTILLA JUAN, HAN VENIDO A ESTOS SUPERIORIDAD LA SENTENCIA DEL SUEC DE CIR.
El juicio, en que se condenó a la ejecución en la cárcel de Novita, y las anexos correspondientes.
compasado los hechos con la disposición legal invocadas en la sentencia, hallate que ita se encuentra alegada al mérito del cruceo, salvo la inco- rección que proviene del 2: arista de los considera-
No cabe la menor duda acerca de qui, si no se deduce provocaciones hecha por la ofendida
a mis agrecidos, no hay razones siligencia para in
fligirles solo la tercera parte de la pena, al tenor del
artículo 661 del Código Penal
Maire del Rosario y Francisco Esteban
Julio Rivas y Maria del Carmen Mo reno, declaran que el viéte de Julio de mil ochio- cientos noventa y uno, previaron la riña o peli-
que, en su casa, todo entre las dos encuadadas y
Brabía Díaz, en la "madre mía" donde había
un plato de maíz perteneciente al segundo de los
telégrafos nombrados. Lucianer Juán Córdoba, tue
Bienvenido a mi casa, donde te espero con paciencia y amabilidad. Te gustará la comida que preparo con amor y cuidado. casa, donde te sentirás cómodo y seguro. Te gustará mi amor y mis cuidados.
las ropas, preciósit ninguna lnea e, para trabajas en la oficina.
"""
    )
    
    prompt = f"Please summarize the following text in Spanish: {text}"
    
    print(f"Input text:\n{prompt}")
    
    print("Invoking LLM...")
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        if response:
            print(f"\nLLM output:\n{response.content}")
        else:
            print("No valid response from LLM.")
    except Exception as e:
        print(f"Error invoking LLM: {e}")

if __name__ == "__main__":
    main()