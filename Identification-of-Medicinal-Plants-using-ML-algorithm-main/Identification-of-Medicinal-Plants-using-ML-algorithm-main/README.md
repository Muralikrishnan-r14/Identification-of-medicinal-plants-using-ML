Medicinal Plant Detection Using Machine Learning

📖 Overview
This project uses machine learning techniques to identify medicinal plants. It processes images of plants and classifies them based on their features. The goal is to provide a tool that can help in identifying plants for medicinal purposes using image recognition.

⚙️ Technologies Used
Python
TensorFlow / Keras
OpenCV
Machine Learning Algorithms
Convolutional Neural Networks (CNN)

📂 Folder Structure
/project-folder
  ├── /data_set
  │     ├── /Single_predection
  │     └── Test 
  │     └── Train
  ├── code.py
  └── README.md
  
📝 How to Run the Project

1. Install Dependencies
To get started, make sure to install the required libraries. Run the following command to install them using pip:

pip install tensorflow keras opencv-python

2. Prepare Your Dataset
Place your plant images inside the /data_set/Single_predection folder.
Make sure the images are named properly (e.g., plant_name.jpg).

4. Run the Code
Run the main script (main.py) to start the plant image classification.
python main.py

4. Check Results
After running the code, the result (image classification) will be displayed.

🔧 Troubleshooting
File Path Issues:
If the code cannot find the image, make sure the file path is correct. You can check if the file exists using os.path.exists().

Permissions:
If you encounter permission errors, try running the script as an administrator or move the project outside OneDrive if using Windows.



