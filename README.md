Very simple medical assistant chatbot that uses natural language processing and machine learning to diagnose diseases and recommend medications based on user input. It is designed with falcon 180b chat moddel.
It takes symptoms from user input (e.g. fever, headache), and then analises that input using a seperated dataset written in json format.
Then It shows the possible disease and recommend a medicine that it gets from another dataset.
Dataset.
The dataset used for this project consists of two JSON files:
diseases.json: contains a list of diseases with their symptoms and medications
medicines.json: contains a dictionary of medications with their dosages

Disclaimer:
This project is for demonstration purposes only and should not be used for actual medical diagnosis or treatment, you should always consult a doctor for accurate diagnosis and treatment.