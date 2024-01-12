import streamlit as st
from transformers import pipeline

# Charger le modèle de classification d'émotions
emotion_classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")

# Titre de l'application
st.title("Classification d'émotions avec Hugging Face")

# Zone de texte pour entrer le texte
text_input = st.text_area("Entrez le texte que vous souhaitez classifier:", "I love using this product! It's amazing.")

# Bouton pour déclencher la classification
if st.button("Classifier l'émotion"):
    # Classifier l'émotion dans le texte
    result = emotion_classifier(text_input)

    # Afficher le résultat
    st.write("Résultat de la classification d'émotion :")
    st.write(result)
