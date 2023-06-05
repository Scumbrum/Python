import spacy
from spacy.matcher import Matcher
import re
from nltk.stem.snowball import EnglishStemmer

def getTextFromFile (filename):
    with open(filename, 'r', encoding='UTF-8') as fp:
        return fp.read()
    
def prepareText(text):
    words = text.split(' ')
    stemmer = EnglishStemmer()

    result = []
    for word in words:
        regexDigit = re.compile(r'.*\d+.*')
        special = re.compile(r'.*[\-&*#$%~<>].*')
        if (not regexDigit.match(word)
            and not special.match(word)
        ):
            result.append(stemmer.stem(word))

    return ' '.join(result)

def task1():
    nlp = spacy.blank("en")
    matcher = Matcher(nlp.vocab)

    # Set up the custom character tokenizer
    tokenizer = CharacterTokenizer(nlp.vocab)
    nlp.tokenizer = tokenizer

    # Define the pattern to match
    pattern = [{'TEXT': {'REGEX': '[^aeiouAEIOU124567890]'}}]
    matcher.add("pattern", [pattern])

    text = getTextFromFile('text3.txt')

    doc = nlp(text)
    matches = matcher(doc)

    filtered_text = []

    for _, start, end in matches:
        for idx in range(start, end):
            filtered_text.append(doc[idx].text)
            

    print(''.join(filtered_text))

def task2():
    text = getTextFromFile('lab7-4.txt')
    text = prepareText(text)
    nlp = spacy.load("uk_core_news_sm")
    doc = nlp(text)
    verbs = []
    for token in doc:
        
        if token.pos_ == "VERB":
            verbs.append(token.lemma_)
    

    print("Дієслова у тексті:")
    for verb in verbs:
        print(verb)

def task3():
    text = getTextFromFile('lab7-4.txt')
    text = prepareText(text)
    nlp = spacy.load("uk_core_news_sm")
    doc = nlp(text)
    sentences = list(doc.sents)
    if len(sentences) >= 3:
        third_sentence = sentences[2]
        lemmas = [token.lemma_ for token in third_sentence if not re.compile(r'.*[\–\-&*#$%~<>\"\.].*').match(token.lemma_)]        
    for lem in lemmas:
        print(lem)

def task4():
    text = getTextFromFile('lab7-4.txt')
    text = prepareText(text)
    nlp = spacy.load("uk_core_news_sm")
    doc = nlp(text)
    locations = []
    for ent in doc.ents:
        if ent.label_ == "GPE":
            locations.append(ent.text)     
              
    for location in locations:
        print(location)

class CharacterTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = [char for char in text]
        return spacy.tokens.Doc(self.vocab, words=words)

def main():
    # Load a language model and create a Matcher
    print("Task1:")
    task1()
    print("Task2:")
    task2()
    print("Task3:")
    task3()
    print("Task4:")
    task3()
   

if __name__ == '__main__':
    main()
