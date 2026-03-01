import os
import uuid
import numpy as np
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
from googletrans import Translator, LANGUAGES
from gtts import gTTS

app = Flask(__name__)

translator = Translator()
languages = LANGUAGES

# ================= MODEL =================
model = load_model("disease_model.h5")
class_names = sorted(os.listdir("dataset"))

API_KEY = "3b2fb0329afdd2008105a295a6e81745"

# ================= IMAGE =================
def preprocess(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((128,128))
    arr = np.array(img)/255.0
    return np.expand_dims(arr,0)

# ================= LOCATION =================
def get_location(lat,lon):

    try:
        url=f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lon}&localityLanguage=en"
        data=requests.get(url).json()

        city=data.get("city") or data.get("locality")
        state=data.get("principalSubdivision")
        country=data.get("countryName")

        return f"{city}, {state}, {country}"
    except:
        return "Unknown Location"

# ================= WEATHER =================
def get_weather(lat,lon):

    try:
        url=f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        d=requests.get(url).json()

        temp=d["main"]["temp"]
        humidity=d["main"]["humidity"]
        condition=d["weather"][0]["description"]

        return temp,humidity,condition
    except:
        return "--","--","Unavailable"

# ================= FARM SOLUTIONS =================
def advice(disease):

    db={
"Bacterial_spot":
("Remove infected leaves",
"Copper oxychloride spray",
"Avoid excess watering"),

"Early_blight":
("Spray fungicide weekly",
"Mancozeb 2g/L",
"Maintain spacing")
    }

    return db.get(
        disease,
        ("Remove affected leaves",
         "Neem oil spray",
         "Maintain irrigation")
    )

# ================= ROUTE =================
@app.route("/",methods=["GET","POST"])
def home():

    if request.method=="POST":

        image=request.files["image"]
        lang=request.form["language"]

        lat=request.form.get("lat")
        lon=request.form.get("lon")

        lat=float(lat) if lat else 0
        lon=float(lon) if lon else 0

        filename=str(uuid.uuid4())+".jpg"
        path=os.path.join("static",filename)
        image.save(path)

        arr=preprocess(path)
        pred=model.predict(arr)

        confidence=float(np.max(pred))*100

        label=class_names[np.argmax(pred)]
        parts=label.split("___")

        plant=parts[0]
        disease=parts[-1]

        treat,med,prev=advice(disease)

        location=get_location(lat,lon)
        temp,humidity,condition=get_weather(lat,lon)

        report=f"""
Location: {location}

Plant: {plant}
Disease: {disease}
Accuracy: {round(confidence,2)} %

Temperature: {temp}°C
Humidity: {humidity}%
Weather: {condition}

Treatment:
{treat}

Medicine:
{med}

Prevention:
{prev}
"""

        translated=translator.translate(report,dest=lang).text

        audio=str(uuid.uuid4())+".mp3"
        gTTS(text=translated,lang=lang).save("static/"+audio)

        return render_template(
            "index.html",
            result=True,
            report=translated,
            image_path=path,
            audio_file=audio,
            languages=languages
        )

    return render_template(
        "index.html",
        result=False,
        languages=languages
    )

if __name__=="__main__":
    app.run(debug=True)