1. Install ultralytics using 'pip install ultralytics'
2. Install cv2 using 'pip install opencv-python'
3-1. use 'python detect.py --source your_image_name.jpg' for image
3-2. use 'python detect.py --source your_video_name.mp4' for video
3-3. use 'python detect.py --source camera' for live detection
4. In runs/detect/predict, marked images or videos and position txt files will be generated.
5. The detect function will return the following content:
    #    {
    #        "id": 1,  #Bauklötze ID
    #        "type": "cylinder",  # Bauklötze type, e.g., cylinder, cube, triangle
    #        "color": "red",  # Bauklötze color
    #        "position": {  # Coordinates of the Bauklötze in the field of view
    #            "x": 0.5,
    #            "y": 0.4,
    #            "z": 0.0  # If it's 2D, z can be ignored
    #        },
    #        "confidence": 0.98  # Recognition confidence
    #        "timestamp": 00:08.77  # Timestamp information
    #    },
    #    {
    #        "id": 2,
    #        "type": "cube",
    #        "color": "blue",
    #        "position": {"x": 0.3, "y": 0.6, "z": 0.0},
    #        "confidence": 0.95
    #        "timestamp": 00:13.17
    #    },
 