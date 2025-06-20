import base64
import os

import numpy as np
import tensorflow as tf
# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torchvision.models as models  # Für VGG16
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# --- Flask App Konfiguration ---
app = Flask(__name__)
app.secret_key = 'ase_gruppe1'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- Globale Variablen für die KI-Modelle ---
model_A_info = {'model': None, 'framework': 'unknown', 'message': 'Modell A nicht geladen.'}
model_B_info = {'model': None, 'framework': 'unknown', 'message': 'Modell B nicht geladen.'}

# Pfade zu den KI-Modellen --> .h5 (Keras) oder .pth (PyTorch) möflich
MODEL_PATH_A = 'models/pneumonia_prediction.h5'
MODEL_PATH_B = 'models/vgg16.pth'


# --- Hilfsfunktion zur Definition der Keras Modellarchitektur ---
def create_pneumonia_model(input_shape=(180, 180, 3)):
    """
    Definiert die Architektur eines Convolutional Neural Network (CNN) Modells für Keras.
    Diese Architektur MUSS der Architektur deines trainierten Keras-Modells entsprechen.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid für binäre Klassifikation
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Hilfsfunktion zum sicheren Laden von TensorFlow/Keras Modellen ---
def load_keras_model_safe(model_path, model_name="KI-Modell", input_shape=(180, 180, 3)):
    """
    Versucht, ein Keras-Modell zu laden.
    Wenn das direkte Laden fehlschlägt (z.B. weil nur Gewichte gespeichert sind),
    wird versucht, eine vordefinierte Architektur zu erstellen und dann die Gewichte zu laden.
    """
    try:
        model = load_model('models/pneumonia_prediction.h5')
        print(f"{model_name} (Keras, vollständig) erfolgreich geladen von {model_path}")
        return model
    except Exception as e_full_load:
        print(f"Fehler beim Laden von {model_name} als vollständiges Keras-Modell aus {model_path}: {e_full_load}")
        print(f"Versuche, nur die Gewichte in eine vordefinierte Keras-Architektur zu laden...")
        try:
            model = create_pneumonia_model(input_shape=input_shape)
            model.load_weights(model_path)
            print(f"{model_name} (Keras, Architektur erstellt, Gewichte geladen) erfolgreich von {model_path}")
            return model
        except Exception as e_weights_only:
            print(f"Auch das Laden nur der Gewichte für {model_name} (Keras) ist fehlgeschlagen: {e_weights_only}")
            print(f"Stelle sicher, dass die Datei existiert und im richtigen Keras-Format ist (.h5).")
            return None


# --- Hilfsfunktion zum Laden des PyTorch Modells ---
def load_pytorch_model_safe(model_path, model_name="PyTorch-Modell"):
    """
    Versucht, ein PyTorch-Modell zu laden.
    Annahme: Es handelt sich um ein VGG16-Modell. Der Classifier wird für 2 Ausgabeklassen angepasst.
    Wenn dein Modell eine andere Architektur oder eine andere Anzahl von Ausgabeklassen hat (z.B. 1 für Sigmoid),
    musst du die Definition der `model`-Variable und des `classifier`-Layers entsprechend anpassen.
    """
    try:
        # Lade ein VGG16-Modell von torchvision. Setze pretrained=False, da wir unsere eigenen Gewichte laden.
        model = models.vgg16(pretrained=False)

        # Passe den Classifier des VGG16-Modells an dein spezifisches Problem an.
        # Annahme: Dein VGG16-Modell hat 2 Ausgabeklassen (z.B. Normal / Pneumonie).
        # Die VGG16-Classifier haben standardmäßig 4096 Input-Features für den letzten Layer.
        # Wenn dein PyTorch-Modell EINE Ausgabe (z.B. für Sigmoid) hat, ändere dies zu `nn.Linear(4096, 1)`.
        model.classifier[6] = nn.Linear(4096, 2) # Beispiel für 2 Ausgabeklassen

        # Lade die gespeicherten Gewichte in das Modell
        # map_location='cpu' ist wichtig, wenn das Modell auf einer GPU trainiert wurde,
        # aber auf einer CPU geladen werden soll.
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Setze das Modell in den Evaluierungsmodus (wichtig für Dropout/BatchNorm)

        print(f"{model_name} (PyTorch) erfolgreich geladen von {model_path}")
        return model
    except Exception as e:
        print(f"DEBUGGING ERROR (PyTorch Modellladen für {model_name}): {e}")
        print(f"Fehler beim Laden von {model_name} aus {model_path}: {e}")
        print(f"Stelle sicher, dass die Datei existiert, ein gültiges PyTorch-Modell ist und die Architektur (insbesondere der Classifier-Layer) korrekt definiert wurde.")
        return None

# --- Hilfsfunktion zum Laden beliebiger Modelle basierend auf Dateiendung ---
def load_any_model_safe(model_path, model_name, input_shape=(180, 180, 3)):
    """
    Versucht, ein Modell basierend auf seiner Dateiendung (.h5 für Keras, .pth für PyTorch) zu laden.
    Gibt ein Dictionary mit dem geladenen Modell, dem Framework-Typ und einer Statusnachricht zurück.
    """
    if not os.path.exists(model_path):
        print(f"Fehler: Modellpfad '{model_path}' nicht gefunden für {model_name}.")
        return {'model': None, 'framework': 'unknown', 'message': f"{model_name} nicht gefunden: '{model_path}'"}

    file_extension = os.path.splitext(model_path)[1].lower()

    if file_extension == '.h5':
        model_obj = load_keras_model_safe(model_path, model_name, input_shape)
        if model_obj:
            return {'model': model_obj, 'framework': 'keras', 'message': f"{model_name} (Keras) erfolgreich geladen."}
        else:
            return {'model': None, 'framework': 'keras', 'message': f"Fehler beim Laden von {model_name} (Keras)."}
    elif file_extension == '.pth':
        model_obj = load_pytorch_model_safe(model_path, model_name)
        if model_obj:
            return {'model': model_obj, 'framework': 'pytorch', 'message': f"{model_name} (PyTorch) erfolgreich geladen."}
        else:
            return {'model': None, 'framework': 'pytorch', 'message': f"Fehler beim Laden von {model_name} (PyTorch)."}
    else:
        print(f"Fehler: Unbekannte Dateiendung '{file_extension}' für Modell {model_name} unter Pfad {model_path}.")
        return {'model': None, 'framework': 'unknown', 'message': f"Unbekannte Dateiendung für {model_name}: '{file_extension}'"}


# Modelle laden beim Serverstart
with app.app_context():
    # Lade Modell A (kann Keras oder PyTorch sein, abhängig von MODEL_PATH_A)
    model_A_info = load_any_model_safe(MODEL_PATH_A, "KI-Modell A")
    # Lade Modell B (kann Keras oder PyTorch sein, abhängig von MODEL_PATH_B)
    model_B_info = load_any_model_safe(MODEL_PATH_B, "KI-Modell B")


# --- Hilfsfunktion für erlaubte Dateierweiterungen ---
def allowed_file(filename):
    """
    Überprüft, ob die Dateierweiterung in der Liste der erlaubten Erweiterungen ist.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# --- Funktion zur Vorverarbeitung eines Bildes ---
def preprocess_image(image_path, target_size=(180, 180), framework="keras"):
    """
    Lädt und verarbeitet ein Bild für die KI-Modelle.
    Passt sich an Keras- oder PyTorch-Anforderungen an.
    """
    img = Image.open(image_path)
    if img is None:
         raise ValueError(f"Bild konnte nicht geladen werden oder ist ungültig: {image_path}")

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize(target_size, Image.Resampling.LANCZOS)

    if framework == "keras":
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen
        return img_array
    elif framework == "pytorch":
        # PyTorch erwartet FloatTensor, Channel-First (C, H, W) und normalisierte Werte.
        # Standard ImageNet Normalisierungswerte sind hier als Beispiel verwendet.
        # Diese Werte sollten zu den Trainingswerten deines PyTorch-Modells passen!
        transform = transforms.Compose([
            transforms.ToTensor(),  # Konvertiert PIL Image zu PyTorch Tensor (H, W, C) -> (C, H, W) und skaliert auf 0-1
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Batch-Dimension hinzufügen
        return img_tensor
    else:
        raise ValueError("Ungültiges Framework für Bildvorverarbeitung angegeben. Wähle 'keras' oder 'pytorch'.")


# --- Funktion zur Durchführung der Vorhersage ---
def make_prediction(image_data_preprocessed, model, model_name, framework="keras"):
    """
    Führt eine Vorhersage mit dem gegebenen KI-Modell aus und interpretiert das Ergebnis.
    Passt sich an Keras- oder PyTorch-Inferenz an.
    """
    if model is None:
        return {"status": "nicht verfügbar", "message": f"{model_name} konnte nicht geladen werden."}, 0.0

    try:
        prediction_raw = 0.0
        if framework == "keras":
            prediction_raw = model.predict(image_data_preprocessed)[0][0]
        elif framework == "pytorch":
            with torch.no_grad():  # Deaktiviert Gradientenberechnung für schnellere Inferenz
                output = model(image_data_preprocessed)

                # Annahme der Ausgabestruktur des PyTorch-Modells:
                # Prüfe die Anzahl der Ausgabemerkmale des letzten Klassifizierer-Layers.
                # Dies ist eine Annahme basierend auf dem VGG16-Classifier-Setup oben.
                # Wenn dein PyTorch-Modell anders aufgebaut ist, musst du dies anpassen.
                if hasattr(model, 'classifier') and len(model.classifier) > 0 and \
                   hasattr(model.classifier[-1], 'out_features') and model.classifier[-1].out_features == 1:
                    # Wenn der letzte Layer 1 Ausgabe hat (typisch für binäre Klassifikation mit Sigmoid)
                    prediction_raw = torch.sigmoid(output).item()
                elif hasattr(model, 'classifier') and len(model.classifier) > 0 and \
                     hasattr(model.classifier[-1], 'out_features') and model.classifier[-1].out_features == 2:
                    # Wenn der letzte Layer 2 Ausgaben hat (typisch für Softmax, z.B. [Normal, Pneumonie])
                    probabilities = torch.softmax(output, dim=1)
                    # Annahme: Pneumonie ist die zweite Klasse (Index 1)
                    prediction_raw = probabilities[0][1].item()
                else:
                    # Fallback, wenn die Ausgabestruktur nicht eindeutig ist, versuche Sigmoid.
                    # Dies sollte nur als Debugging-Hilfe dienen. Besser ist es, die Modellarchitektur genau zu kennen.
                    print(f"WARNUNG: Unklare PyTorch-Modellausgabe für {model_name}. Versuche Sigmoid.")
                    # Hier könntest du auch einen Fehler werfen oder eine andere Standardannahme treffen.
                    # Für den Moment nehmen wir Sigmoid an, da es am häufigsten ist.
                    prediction_raw = torch.sigmoid(output).item()

        else:
            raise ValueError("Ungültiges Framework für Vorhersage angegeben.")

        result_status = "positiv" if prediction_raw > 0.5 else "negativ"
        return {"status": result_status, "probability": f"{prediction_raw * 100:.2f}%"}, prediction_raw
    except Exception as e:
        # Hier wird die detaillierte Fehlermeldung von der Vorhersage ausgegeben
        print(f"DEBUGGING ERROR (Vorhersage mit {model_name}): {e}")
        return {"status": "Fehler", "message": f"Fehler bei der Vorhersage von {model_name}: {e}"}, 0.0


# --- Flask Routen ---
@app.route('/')
def index():
    """
    Rendert die Hauptseite mit dem Upload-Formular.
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Verarbeitet den Bild-Upload, speichert das Bild temporär,
    führt die KI-Analyse durch und zeigt die Ergebnisse an.
    """
    if 'file' not in request.files:
        flash('Keine Datei ausgewählt.')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('Keine Datei ausgewählt.')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            with open(filepath, 'rb') as f:
                image_content = f.read()
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            mime_type = filename.rsplit('.', 1)[1].lower()
            if mime_type == 'jpg':
                mime_type = 'jpeg'
            image_base64_uri = f"data:image/{mime_type};base64,{image_base64}"

            # Initialisiere die Ergebnisse als "nicht verfügbar"
            result_A = {"status": "nicht verfügbar", "message": model_A_info['message']}
            prob_A = 0.0
            result_B = {"status": "nicht verfügbar", "message": model_B_info['message']}
            prob_B = 0.0

            # Vorverarbeitung und Vorhersage für Modell A, falls geladen
            if model_A_info['model'] is not None:
                processed_image_A = preprocess_image(filepath, framework=model_A_info['framework'])
                result_A, prob_A = make_prediction(
                    processed_image_A, model_A_info['model'], "KI-Modell A", framework=model_A_info['framework']
                )

            # Vorverarbeitung und Vorhersage für Modell B, falls geladen
            if model_B_info['model'] is not None:
                processed_image_B = preprocess_image(filepath, framework=model_B_info['framework'])
                result_B, prob_B = make_prediction(
                    processed_image_B, model_B_info['model'], "KI-Modell B", framework=model_B_info['framework']
                )

            os.remove(filepath)

            return render_template('result.html',
                                   image_base64_uri=image_base64_uri,
                                   result_A=result_A,
                                   result_B=result_B,
                                   prob_A=prob_A,
                                   prob_B=prob_B)

        except ValueError as e:
            flash(f"Fehler bei der Bildverarbeitung: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))
        except Exception as e:
            flash(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))
    else:
        flash('Unerlaubter Dateityp. Bitte lade eine Bilddatei (png, jpg, jpeg, gif, bmp) hoch.')
        return redirect(request.url)


# --- Start der Anwendung ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
