# Face Recognition Based Attendance System

A smart attendance system that uses facial recognition to mark attendance automatically. Built with Python and modern web technologies, it provides real-time attendance tracking with Indian Standard Time (IST).

## 🚀 Technologies Used

### Backend Technologies
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
- **OpenCV** - For face detection and image processing
- **Flask** - Web framework to serve the application
- **scikit-learn** - For face recognition (KNN classifier)
- **pandas** - For handling attendance data in CSV format

### Frontend Technologies
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white)
- **JavaScript** - For real-time camera handling and UI updates
- **Material Icons** - For beautiful UI elements

## 💻 How It Works

1. **Face Detection**
   - Uses OpenCV's Haar Cascade Classifier
   - Detects faces in real-time from webcam feed
   - Works with both laptop and mobile cameras

2. **Face Recognition**
   - Uses K-Nearest Neighbors (KNN) algorithm
   - Trained on multiple face images per person
   - Fast and accurate recognition

3. **Attendance System**
   - Automatically marks attendance with IST timestamp
   - Stores data in CSV format
   - Creates automatic backups
   - Prevents duplicate entries

## 🛠️ Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone <your-repo-link>
   cd face-recognition-attendance
   ```

2. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Access the Application**
   - On same computer: `http://localhost:5000`
   - From other devices: `http://your-computer-ip:5000`

## 📱 How to Use

### Adding a New User
1. Click on "Add New User" button
2. Enter the person's name and ID
3. The system will capture 50 images of their face
4. Wait for training to complete

### Taking Attendance
1. Click "Start Attendance" button
2. Look at the camera
3. The system will:
   - Detect your face
   - Recognize your identity
   - Mark attendance with time
   - Show confirmation message

### Viewing Attendance
- Attendance is displayed in real-time
- Stored in `Attendance` folder as CSV files
- Format: Name, ID, Time (IST)

## 📊 Features

- ✅ Real-time face detection
- ✅ Automatic attendance marking
- ✅ Indian Standard Time (IST)
- ✅ Mobile responsive design
- ✅ CSV data storage
- ✅ Automatic backups
- ✅ User-friendly interface
- ✅ Multi-platform support

## 📸 Screenshots

### Home Page
![Home Page](static/screenshots/home.png)

### Adding New User
![Add User](static/screenshots/add_user.png)

### Attendance View
![Attendance](static/screenshots/attendance.png)

## ⚠️ Important Notes

1. **Camera Access**
   - Allow camera permissions when prompted
   - Good lighting improves recognition
   - Keep face centered in frame

2. **System Requirements**
   - Python 3.8 or higher
   - Webcam/Camera
   - Modern web browser

3. **Best Practices**
   - Add multiple face angles for better recognition
   - Regular backups of attendance data
   - Update the model periodically

## 👨‍💻 Developer
Developed by: Manjeet Singh

## 🤝 Support
For any issues or questions:
- Create an issue on GitHub
- Email: your-email@example.com

---
Made with ❤️ in India 🇮🇳 