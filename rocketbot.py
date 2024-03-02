#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import wikipedia
import requests
import aiml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver

def check_kb_integrity(kb):
    for expr in kb:
        # Construct the negation of the current expression
        neg_expr = Expression.fromstring('not (' + str(expr) + ')')
        
        # Create a temporary KB without the current expression
        temp_kb = [e for e in kb if e != expr]

        print(f"Checking integrity of KB with expression: {expr}")
        
        # Check if the negation of the expression can be proved using the temporary KB
        if ResolutionProver().prove(neg_expr, temp_kb, verbose=True):
            raise ValueError(f"Contradiction found in KB with expression: {expr}")
    print("KB integrity check passed. No contradictions found.")

def get_spaceX_rockets():
    url = "https://api.spacexdata.com/v4/rockets"
    response = requests.get(url)
    if response.status_code == 200:
        rockets = response.json()
        print("---------------------------------------------------")
        for rocket in rockets:
            print(f"Rocket Name: {rocket['name']}")
            print(f"Description: {rocket['description']}")
            print(f"Height: {rocket['height']['meters']} meters")
            print(f"Diameter: {rocket['diameter']['meters']} meters")
            print(f"Mass: {rocket['mass']['kg']} kg")
            print("---------------------------------------------------")
    else:
        print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")

def get_spaceX_launches(num):
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    if response.status_code == 200:
        launches = response.json()
        print("---------------------------------------------------")
        for launch in launches[:int(num)]:
            print(f"Name: {launch['name']}")
            print(f"Success: {launch['success']}")
            print(f"Details: {launch['details']}")
            print(f"Date: {launch['date_utc']}")
            print("---------------------------------------------------")
    else:
        print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")

def get_similar_question(user_input, threshold=0.5):
    query_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(query_vector, vectorizer.transform(questions))
    max_similarity = similarity.max()
    if max_similarity < threshold: return 0
    index = similarity.argmax()
    return answers[index]

def main():
    print("Welcome to this chat bot. Please feel free to ask questions from me!")
    while True:
        try:
            userInput = input("> ")
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye!")
            break
        responseAgent = 'aiml'
        if responseAgent == 'aiml':
            answer = kern.respond(userInput)
            if answer == "": 
                answer = get_similar_question(userInput)
                if answer == 0: answer = "Sorry, I do not know that. Be more specific!"
                print(answer)
            elif answer[0] == '#':
                params = answer[1:].split('$')
                if params[0] == 'SpaceXAPI':
                    if params[1] == 'rockets':
                        get_spaceX_rockets()
                    elif params[1] == 'launches':
                        num = input("How many launches do you want to see?\n> ")
                        get_spaceX_launches(num)
                else:
                    cmd = int(params[0])
                    if cmd == 0:
                        print(params[1])
                        break
                    elif cmd == 1:
                        try:
                            wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=False)
                            print(wSummary)
                        except:
                            print("Sorry, I do not know that. Be more specific!")
                    elif cmd == 31:  # if input pattern is "I know that * is *"
                        object, subject = params[1].split(' is ')
                        expr = read_expr(subject + '(' + object + ')')
                        
                        # Check for contradiction
                        neg_expr = read_expr('-' + subject + '(' + object + ')')
                        if ResolutionProver().prove(neg_expr, kb, verbose=False):
                            print("Error: This contradicts what I already know.")
                        else:
                            kb.append(expr)
                            print('OK, I will remember that', object, 'is', subject)
                    elif cmd == 32:  # if the input pattern is "check that * is *"
                        object, subject = params[1].split(' is ')
                        expr = read_expr(subject + '(' + object + ')')
                        neg_expr = read_expr('-' + subject + '(' + object + ')')
                        
                        if ResolutionProver().prove(expr, kb, verbose=False):
                            print('Correct.')
                        elif ResolutionProver().prove(neg_expr, kb, verbose=False):
                            print('Incorrect.')
                        else:
                            print("Sorry, I don't know.")
                    elif cmd == 99:
                        print("I did not get that, please try again.")
            else:
                print(answer)

if __name__ == "__main__":
    kern = aiml.Kernel()
    kern.setTextEncoding(None)
    kern.bootstrap(learnFiles="rockets-aiml.xml")
    kern.verbose(False)

    # AIML agent
    df = pd.read_csv('rockets-task-a.csv')
    questions = df['question'].tolist()
    answers = df['answer'].tolist()

    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(questions)

    # Knowledge Base
    read_expr = Expression.fromstring
    kb = []
    data = pd.read_csv('rockets-kb.csv', delimiter=';', header=None)
    [kb.append(read_expr(row)) for row in data[0]]
    check_kb_integrity(kb)

    main()