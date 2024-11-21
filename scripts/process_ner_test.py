import spacy

def process_ner_test(text: str, spacy_model: str = "es_core_news_lg"):
    """
    Perform named entity recognition (NER) on the provided text using the specified spaCy model.
    """
    # Load or download spaCy model
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        from spacy.cli.download import download
        download(spacy_model)
        nlp = spacy.load(spacy_model)

    # Process the text with spaCy
    doc = nlp(text)
    # Extract named entities
    for ent in doc.ents:
        if ent.label_ == "PER":
            print(ent.text)  # ent.start_char, ent.end_char, ent.label_
        elif ent.label_ == "LOC":
            # print(ent.text)  # ent.start_char, ent.end_char, ent.label_
            pass
        elif ent.label_ == "ORG":
            # print(ent.text)  # ent.start_char, ent.end_char, ent.label_
            pass
        else:
            # print("NOT PERSON", ent.text, ent.start_char, ent.end_char, ent.label_)
            pass

if __name__ == "__main__":
    sample_text = """
    En el pueblo de San Francisco de Quibdo, capital de la provincia del Cauca, a ocho días del mes de marzo de mil ochocientos ochenta y ocho, ante mi don Carlos de Sigüenza, gobernador político y militar de estas provincias, compareció don José María Pérez, vecino de esta ciudad, quien declaró lo siguiente...
    """
    process_ner_test(sample_text)