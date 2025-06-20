# Verwende ein schlankes Python 3.9 Image als Basis
FROM python:3.11-slim-buster

# Lege das Arbeitsverzeichnis im Container fest
WORKDIR /app

# Kopiere zuerst die requirements.txt, um das Caching von Docker-Layern zu optimieren
COPY requirements.txt .

# Installiere die Python-Abhängigkeiten
# --no-cache-dir reduziert den Image-Speicherplatz
# -r installiert Pakete aus der requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den gesamten Rest des Anwendungsquellcodes in das Arbeitsverzeichnis
# Dazu gehören app.py, die 'models/' und 'templates/' Ordner
COPY . .

# Erstelle die Verzeichnisse für Modelle und Uploads, falls sie nicht vorhanden sind
# Dies stellt sicher, dass die Anwendung die erforderlichen Pfade findet
RUN mkdir -p models && mkdir -p uploads

# Exponiere Port 5000, da Flask standardmäßig auf diesem Port läuft
EXPOSE 5000

# Definiere den Befehl, der ausgeführt wird, wenn der Container startet
# Hier starten wir die Flask-Anwendung
CMD ["python", "app.py"]