import json
import spacy
from spacy.training import Example
import random

with open('movies.json', 'r') as file:
    data = json.load(file)
    TRAIN_DATA = []
    for dialog in data:
        for turn in dialog["turns"]:
            anno = []
            for frame in turn["frames"]:
                for slot in frame["slots"]:

                    anno.append((slot["start"], slot["exclusive_end"],slot["slot"]))
            TRAIN_DATA.append((turn["utterance"], {"entities": anno}))

    nlp = spacy.load('en_core_web_md')
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    optimizer = nlp.create_optimizer()  
    with nlp.disable_pipes(*other_pipes):
        for i in range(10):
            random.shuffle(TRAIN_DATA)
            print("it")
            for text, entities in TRAIN_DATA:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, entities)
                nlp.update([example], sgd=optimizer)
    text = "I found 1 showtime for this movie at 3 pm in Cinemark USA."

    doc = nlp(text)
    print(doc)
    for ent in doc.ents:
        print(ent.text, ent.label_)

    TRAIN_DATA = []
    for dialog in data:
        for turn in dialog["turns"]:
            frame = turn["frames"][0]
            if "state" in frame:
                TRAIN_DATA.append((turn["utterance"], frame["state"]["active_intent"]))
    texts = [data[0] for data in TRAIN_DATA]
    labels = [data[1] for data in TRAIN_DATA]


    textcat = nlp.add_pipe("textcat", last=True)
    for label in set(labels):
        textcat.add_label(label)
    train_data = list(zip(texts, [{'cats': {label: (label == y) for label in textcat.labels}} for y in labels]))


    train_examples = [Example.from_dict(nlp.make_doc(data[0]),data[1]) for
    data in train_data]
    textcat.initialize(lambda: train_examples, nlp=nlp)

    for epoch in range(10):
        random.shuffle(train_data)
        losses = {}
        for text, annotations in train_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], sgd=optimizer, losses=losses)

    def predict_intent(text):
        doc = nlp(text)
        predicted_labels = doc.cats
        intent = max(predicted_labels, key=predicted_labels.get)
        return intent

    text = "Які фільми сьогодні грають в кінотеатрі?"
    intent = predict_intent(text)
    print(intent)