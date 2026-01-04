#!/bin/bash

echo "🔴 Démarrage du serveur Ollama..."
ollama serve &

echo "⏳ Attente du démarrage d'Ollama..."
sleep 10  # On laisse le temps au serveur de se lancer

echo "⬇️ Téléchargement du modèle (Version légère pour le Cloud gratuit)..."
# On utilise qwen2.5:0.5b ou tinyllama car ils sont rapides sur CPU
ollama pull qwen2.5:7b

echo "🟢 Démarrage de l'application Streamlit..."
streamlit run app.py --server.port=7860 --server.address=0.0.0.0