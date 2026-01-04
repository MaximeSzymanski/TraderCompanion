# ---- Base image ----
FROM python:3.12-slim

# 1. EN TANT QUE ROOT : Installations système
# On installe curl ET Ollama tout de suite, car on a les droits admin ici.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://ollama.com/install.sh | sh

# 2. Création de l'utilisateur sécurisé (pour Hugging Face)
RUN useradd -m -u 1000 user

# 3. ON CHANGE D'UTILISATEUR MAINTENANT
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 4. Installation des dépendances Python (en tant que user)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copie du reste des fichiers
COPY --chown=user . .

# 6. Permissions sur le script de démarrage
RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]
