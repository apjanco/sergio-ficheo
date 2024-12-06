from langchain_ollama.chat_models import ChatOllama
from langchain.schema import HumanMessage
import os

def main():
    llm_model = "llama3.1:8b"  # Replace with your model name
    llm = ChatOllama(model=llm_model, format="json", num_ctx=8000, temperature=0)
    
    text = (
        """Pasó en el comandador y sus sucesores, 
que en señal de posesión y para título de ella, otorga a su favor esta escritura por la
cual ha de servirse haberla adquirido sin que necesite de otro cota de perpendicula de 
que lo poseya. Se obliga a la ejecución y pago de agresión de que lo lleva, y se obliga
a la excusión saneamiento de esta venta a su costo y mención hasta dejar al comprador 
en quietud y pacífica posesión. No pudiendo la canearle de volver a los descuentos, 
cincuenta pesos de plata si ya los hubiera recibido, y le pagará los costos y gastos de
su ineficiencia cuya prueba debe depurar en su simple juramento relevante de otra 
aunque por derecho se requiera.

Siendo presente dicho ciudadano Juan de Mené a quien asimismo se conoció en el año de 
1949. Ciudadano Juan de Nena, quien asimismo doy conocido en tercado de esta escritura,
dijo que la acepta y confesada deber al ciudadano Joquín Freire de Andrade los 
descuentos con cuenta pesos de plata valor de la escala Ramisidón que le acaba de 
comprar. Se obliga a satisfacerse los derechos del mío de seis meses contados desde hoy
en la especie de oro en polvo limpio y soplando a diez y seis reales el castellano con 
las costas y gastos de su cobranza.

Ambos vendedores y comprador, por lo que a cada uno fuese obligado a la exacta 
observancia y cumplimiento de todo lo referido con sus personas y bienes habidos y por 
haber con el poderío de Justicias Santo. Siendo necesaria la renuncia de leyes en 
derecho, más con la general en forma. En su testimonio, la general en forma.

Firmó el vendedor y por decir el comprador no saber lo hizo a su riesgo uno de los 
testigos que le fueron los ciudadanos Manuel Flórez, Luis Alcaro y Nicolás Roxas. 
Joaquín Freire de Andrade, Aruego de Juan de Mené y como testigo, Arz. mi Vicente 
Olavarría. Ante mi Vicente Olaguecha.

En la ciudad de Quito, capital de la provincia, a ocho de Enero de mil ochocientos diez
y seis [1816-01-08]. Ante mi el escribano y testigos que se nombraron, pareció la 
ciudadana Maria Manuela Scarpeta, vecina de ella y consorte, apoderada del sudadero 
Carlos Ferrer con consentimiento exento. El día trece de Agosto del año próximo pasado 
de mil ochocientos quince [1815-08-13] de que doy fe, y usando de las facultades que 
en.
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