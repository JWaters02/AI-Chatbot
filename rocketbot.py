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
from tensorflow import keras
import numpy as np
import tkinter as tk
from tkinter import filedialog

def get_image_class():
    print("Please upload the image you want to classify.")
    uploaded_image_path = filedialog.askopenfilename(
        title='Select an Image',
        filetypes=[('Image Files', '*.png;*.jpg;*.jpeg')]
    )
    image = keras.preprocessing.image.load_img(uploaded_image_path, target_size=(224, 224))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    
    class_indices = {0: 'drone', 1: 'fighter jet', 2: 'helicopter', 3: 'missile', 4: 'passenger plane', 5: 'rocket'}
    return class_indices[predicted_class[0]].lower()

def predict_image_class():
    image_class = get_image_class()
    print(f"This image is most likely a {image_class}.")

def is_image_type(image_type):
    image_class = get_image_class()
    
    if image_class == image_type.lower():
        print(f"Yes, this image is most likely a {image_type}.")
    else:
        print(f"No, this image is most likely a {image_class}.")

def check_kb_integrity(kb):
    for expr in kb:
        # Construct the negation of the current expression
        neg_expr = Expression.fromstring('not (' + str(expr) + ')')
        
        # Create a temporary KB without the current expression
        temp_kb = [e for e in kb if e != expr]

        # print(f"Checking integrity of KB with expression: {expr}")
        
        # Check if the negation of the expression can be proved using the temporary KB
        if ResolutionProver().prove(neg_expr, temp_kb, verbose=False):
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
                elif params[0] == 'CNN':
                    if params[1] == 'predict':
                        predict_image_class()
                    elif params[1] == 'isType':
                        image_type = params[2]
                        is_image_type(image_type)
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
                        # remove words like "a", "an", "the" from the input
                        text = ' '.join([word for word in params[1].lower().split() if word not in ('a', 'an', 'the')])
                        object, subject = text.split(' is ')
                        
                        expr = read_expr(subject + '(' + object + ')')
                        
                        # Check for contradiction
                        neg_expr = read_expr('-' + subject + '(' + object + ')')
                        if ResolutionProver().prove(neg_expr, kb, verbose=False):
                            print("Error: This contradicts what I already know.")
                        else:
                            print('OK, I will remember that', object, 'is', subject)
                            remember = input("Would you like for me to permanently remember this fact? (y/n)\n> ")
                            if remember.lower() == 'y': kb.append(expr)
                    elif cmd == 32:  # if the input pattern is "check that * is *"
                        # remove words like "a", "an", "the" from the input
                        text = ' '.join([word for word in params[1].lower().split() if word not in ('a', 'an', 'the')])
                        object, subject = text.split(' is ')

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
    # AIML agent
    kern = aiml.Kernel()
    kern.setTextEncoding(None)
    kern.bootstrap(learnFiles="rockets-aiml.xml")
    kern.verbose(False)

    # Predefined questions and answers
    df = pd.read_csv('rockets-task-a.csv')
    questions = df['question'].tolist()
    answers = df['answer'].tolist()

    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(questions)

    # Knowledge base
    read_expr = Expression.fromstring
    kb = []
    data = pd.read_csv('rockets-kb.csv', delimiter=';', header=None)
    [kb.append(read_expr(row)) for row in data[0]]
    check_kb_integrity(kb)

    # CNN agent
    model = keras.models.load_model('rocket-detection.h5')

    # File dialog
    root = tk.Tk()
    root.withdraw()

    main()

    # Save the KB
    with open('rockets-kb.csv', 'w') as f:
        for expr in kb:
            f.write(str(expr) + '\n')