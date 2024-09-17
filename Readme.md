<p>
  <div align="center">
  <h1>
<br >
  Interview Proctoring<br /> <br />
    <a href="https://github.com/psf/black">
      <img
        src="https://img.shields.io/badge/code%20style-black-000000.svg"
        alt="The Uncompromising Code Formatter"
      />
    </a>
      <a>
      <img
        src="https://img.shields.io/badge/python-3.9%20%7C%203.10-blue"
        alt="Python Versions"
      />
    </a>
    <a href="https://opensource.org/licenses/MIT">
      <img
        src="https://img.shields.io/badge/GUI-PyQt6-green"
        alt="PyQt6"
      />
    </a>
    <a href="https://opensource.org/licenses/MIT">
      <img
        src="https://img.shields.io/badge/CV-OpenCV-red"
        alt="CV-OpenCV-red"
      />
    </a>
    <a href="https://opensource.org/licenses/MIT">
      <img
        src="https://img.shields.io/badge/DL-TensorFlow-orange"
        alt="TensorFlow"
      />
    </a>
     <a href="https://opensource.org/licenses/MIT">
      <img
        src="https://img.shields.io/badge/License-MIT-blue.svg"
        alt="License: MIT"
      />
    </a>
  </h1>
  </div>
   <h3>InterviewProctorApp is a tool designed to monitor and analyze participant behavior during remote interviews. It uses computer vision and Deep learning techniques to detect multiple faces, eye movements, and head positions, ensuring a fair and secure interview process.
</h3>
  <h4>Welcome to the setup guide for the Interview Proctoring Module. Follow these steps to get your environment ready and run the application.</h4>
</p>

---
## Prerequisites
- Ensure you have Git installed on your system.
- Python 3 should be installed on your system.

## Step 1: Clone the Repository
First, clone the repository to your local machine using Git. Open your terminal and run:

```
git clone https://deep_07@bitbucket.org/codistedev/interview-proctoring.git
```

## Step 2: Create a Virtual Environment
Creating a virtual environment is crucial to manage dependencies.

### For Mac & Linux:
Run the following commands:

```
python3 -m venv env/interview_proctoring
source env/interview_proctoring/bin/activate
```

### For Windows:
Run these commands in your Command Prompt or PowerShell:

```
python -m venv env\interview_proctoring
.\env\interview_proctoring\Scripts\activate
```

With your virtual environment active, install the required Python packages:


#### For Windows:
```
pip install -r requirements.txt
```

#### For Mac & Linux:
```
pip3 install -r requirements.txt
```

## Step 3: Run the Application
Finally, start the application with the following command:

#### For Windows:
```
python main.py
```

#### For Mac & Linux:
```
python3 main.py
```

### Demo

You can view a demonstration of the InterviewProctorApp in action : 

https://github.com/user-attachments/assets/9e4b6a11-6232-40ca-a183-14cca81fe512

## File Structure

- `main.py`: The main application script
- `utils.py`: Utility functions for detection and image processing
- `model/`: Directory containing the TensorFlow Lite model
- `temp/`: Directory where screenshots are saved (created at runtime)

## Acknowledgments

- This project uses the MoveNet model for pose estimation
- Face detection is powered by the InsightFace library

## Conclusion
Your setup is now complete! If you encounter any issues, submit an issue on the GitHub repository.
---