import cv2
import os
from flask import Flask, request, render_template, jsonify
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import base64
import io
from PIL import Image
import pytz
import requests
from datetime import datetime, timezone

# Defining Flask App
app = Flask(__name__)

def get_india_datetime():
    india_tz = pytz.timezone('Asia/Kolkata')
    india_time = datetime.now(india_tz)
    return {
        'date': india_time.strftime("%d-%B-%Y"),
        'time': india_time.strftime("%I:%M:%S %p"),
        'day': india_time.strftime("%A")
    }

def get_india_weather():
    try:
        # Using OpenWeatherMap API for New Delhi (you can change the city)
        # You should replace 'YOUR_API_KEY' with an actual API key from openweathermap.org
        api_key = 'YOUR_API_KEY'  # Get this from openweathermap.org
        city = 'New Delhi'
        url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': f"{data['main']['temp']}°C",
                'humidity': f"{data['main']['humidity']}%",
                'description': data['weather'][0]['description']
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    return None

# Saving Date today in 2 different formats using Indian timezone
india_datetime = get_india_datetime()
datetoday = datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%m_%d_%y")
datetoday2 = india_datetime['date']

# Initialize face detector
# Get OpenCV's installation path and construct the cascade classifier path
opencv_path = os.path.dirname(cv2.__file__)
cascade_path = os.path.join(opencv_path, 'data', 'haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier(cascade_path)

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    if isinstance(img, str):  # If image is base64 string
        try:
            # Remove the "data:image/jpeg;base64," part if present
            if ',' in img:
                img = img.split(',')[1]
            img_data = base64.b64decode(img)
            img = Image.open(io.BytesIO(img_data))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error decoding image: {e}")
            return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    
    # Get current time in Indian timezone
    india_tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(india_tz).strftime("%I:%M:%S %p")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{current_time}')

################## ROUTING FUNCTIONS #########################

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    india_data = get_india_datetime()
    weather_data = get_india_weather()
    
    return render_template('home.html',
                         names=names,
                         rolls=rolls,
                         times=times,
                         l=l,
                         totalreg=totalreg(),
                         datetoday2=datetoday2,
                         india_time=india_data['time'],
                         india_day=india_data['day'],
                         weather=weather_data)

@app.route('/start', methods=['POST'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return jsonify({
            'success': False,
            'message': 'There is no trained model in the static folder. Please add a new face to continue.'
        })

    try:
        # Get the image data from the request
        request_data = request.json
        if not request_data or 'image' not in request_data:
            return jsonify({'success': False, 'message': 'No image data received'})
            
        image_data = request_data['image']
        if not image_data:
            return jsonify({'success': False, 'message': 'Empty image data received'})

        # Process the image
        faces = extract_faces(image_data)
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'})

        # Get the first face
        (x, y, w, h) = faces[0]
        
        # Convert base64 to image
        img_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Extract and process face
        face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
        identified_person = identify_face(face.reshape(1, -1))[0]
        
        # Add attendance
        add_attendance(identified_person)
        
        # Get updated attendance
        names, rolls, times, l = extract_attendance()
        
        return jsonify({
            'success': True,
            'person': identified_person,
            'attendance': {
                'names': names.tolist(),
                'rolls': rolls.tolist(),
                'times': times.tolist(),
                'length': l
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/add', methods=['POST'])
def add():
    try:
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        userimagefolder = f'static/faces/{newusername}_{newuserid}'
        
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)

        # Get the list of images
        images_data = request.form.getlist('images[]')
        if not images_data:
            return jsonify({'success': False, 'message': 'No images received'})

        saved_count = 0
        for idx, image_data in enumerate(images_data):
            if not image_data:
                continue

            # Process and save the image
            try:
                # Remove the "data:image/jpeg;base64," part if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                img_data = base64.b64decode(image_data)
                img = Image.open(io.BytesIO(img_data))
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                faces = extract_faces(frame)
                if len(faces) == 0:
                    continue

                # Save the face image
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                img_name = f"{newusername}_{saved_count}.jpg"
                cv2.imwrite(os.path.join(userimagefolder, img_name), face_img)
                saved_count += 1

            except Exception as e:
                print(f"Error processing image {idx}: {str(e)}")
                continue

        if saved_count == 0:
            return jsonify({'success': False, 'message': 'No valid face images were captured'})

        # Train the model
        train_model()

        # Get updated attendance
        names, rolls, times, l = extract_attendance()
        
        return jsonify({
            'success': True,
            'message': f'User added successfully with {saved_count} images',
            'attendance': {
                'names': names.tolist(),
                'rolls': rolls.tolist(),
                'times': times.tolist(),
                'length': l
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/get_time')
def get_time():
    india_data = get_india_datetime()
    return jsonify({
        'time': india_data['time'],
        'date': india_data['date'],
        'day': india_data['day']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)