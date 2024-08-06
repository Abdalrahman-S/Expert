import json
import os
from ai71 import AI71
from dotenv import load_dotenv

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def configure():
    load_dotenv()

# Load NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Loading diseases and medications data from JSON datasets
try:
    with open('diseases.json') as f:
        diseases = json.load(f)

    with open('medicines.json') as f:
        medicines = json.load(f)
except FileNotFoundError:
    print("Error: Disease or medicine data file not found.")
    exit(1)

def preprocess_symptoms(symptoms: str) -> list:

    # Tokenize the input symptoms
    tokens = word_tokenize(symptoms)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def get_disease_from_symptoms(symptoms: str, diseases: list) -> str:
    
    try:
        # Preprocess user input
        tokens = preprocess_symptoms(symptoms)

        # Create a dictionary to store disease-symptom relationships
        disease_symptoms = {}
        for disease in diseases:
            # Preprocess symptoms from JSON as well
            processed_symptoms = preprocess_symptoms(' '.join(disease['symptoms']))
            disease_symptoms[disease['name']] = processed_symptoms

        # Find the disease with the most matching symptoms
        max_matches = 0
        best_match = None
        for disease, symptoms_list in disease_symptoms.items():
            matches = sum(1 for symptom in tokens if symptom in symptoms_list)
            if matches > max_matches:
                max_matches = matches
                best_match = disease

        # Return the best match, or None if no match found
        return best_match

    except Exception as e:
        print(f"Error: {e}")
        return None

def get_medicine_for_disease(disease: str) -> dict:
    medication_data = medicines.get(disease, {})
    return medication_data

def main():
    configure()
    # Initialize AI71 client
    try:
        client = AI71(os.getenv("AI71_API_KEY"))
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


    messages = [{"role": "system", "content": "Hi! I'm a medical assistant, my name is Expert. How do you feel?"}]

    print("Hi! I'm a medical assistant, my name is Expert. How do you feel?")
    print("Please enter a description of your symptoms separated by commas (e.g., fever, headache).")

    while True:
        print("Hi there, how are you? Please describe your symptoms:")
        user_symptoms = input("Patient: ")
        
        # Validatint user input
        if not user_symptoms.strip():
            print("Please enter a description of your symptoms.")
            continue

        messages.append({"role": "user", "content": user_symptoms})
        print("Expert:", sep="", end="", flush=True)
        
        # Diagnosis
        diagnosed_disease = get_disease_from_symptoms(user_symptoms, diseases)
        if diagnosed_disease:
            print(f"Based on your symptoms, it could be {diagnosed_disease}.")

            # Medication Recommendation
            recommended_medications = get_medicine_for_disease(diagnosed_disease)
            if recommended_medications:
                print("Possible medications include:")
                medication = recommended_medications.get("medication")
                dosage = recommended_medications.get("dosage")
                print(f"- {medication} ({dosage})")
            else:
                print("No specific medication recommendations available. Please consult a doctor.")
        else:
            print("I couldn't determine a potential diagnosis based on your symptoms. Please consult a doctor.")

        # Getting response from AI71 API
        response = client.chat.completions.create(
            messages=messages,
            model="tiiuae/falcon-180B-chat",
            stream=True,
        )

        # Print response from AI71 API
        delta_content = ""
        for chunk in response:
            delta_content += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, sep="", end="", flush=True)

        # Append assistant's response to messages list
        messages.append({"role": "assistant", "content": delta_content})
        print("\n")

if __name__ == "__main__":
    main()
