FROM python:3.11-slim

WORKDIR /app

# Installer les dépendances nécessaires pour h5py
RUN apt-get update && apt-get install -y \
    libhdf5-serial-dev \
    gcc \
    gfortran \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    pkg-config

# Copier les fichiers de dépendances dans le conteneur
COPY requirements.txt .

# Installer les dépendances
RUN pip install --upgrade pip && \
    pip install --no-cache-dir h5py && \
    pip install --no-cache-dir -r requirements.txt

# Copier le reste du code dans le conteneur
COPY . .

# Exécuter l'application
CMD ["python", "demo_app.py"]