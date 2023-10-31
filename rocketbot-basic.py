#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import wikipedia
import requests
import aiml
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
                            wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=False)
                            print(wSummary)
                        except:
                            print("Sorry, I do not know that. Be more specific!")
                    elif cmd == 99:
                        print("I did not get that, please try again.")

if __name__ == "__main__":
    kern = aiml.Kernel()
    kern.setTextEncoding(None)
    kern.bootstrap(learnFiles="rockets-basic.xml")
    kern.verbose(False)

    df = pd.read_csv('rockets.csv')
    questions = df['question'].tolist()
    answers = df['answer'].tolist()

    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(questions)

    main()