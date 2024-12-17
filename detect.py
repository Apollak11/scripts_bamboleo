import argparse
from ultralytics import YOLO
import cv2
import math
import os
import numpy as np
from datetime import timedelta
import json


def save_json_output(data, source_path, is_video=False):
    import os
    import json
    
    # Create json directory if it doesn't exist
    os.makedirs('json', exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    
    if is_video:
        # Group first frame of each block by ID
        first_frames_by_id = {}
        for block in data:
            block_id = block['id']
            if block_id not in first_frames_by_id:
                first_frames_by_id[block_id] = block  # Save the first occurrence (first frame)

        # Prepare the final data containing only the first frames
        output_data = list(first_frames_by_id.values())
        
        # Save a single JSON file for the video
        output_path = os.path.join('json', f'{base_name}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
    else:
        # Save single JSON file for image
        output_path = os.path.join('json', f'{base_name}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    _, ext = os.path.splitext(filename)
    return ext.lower() in video_extensions

def get_color_label(frame, x, y):
    # Get RGB value at the position
    b, g, r = frame[int(y), int(x)]
    if (10 <= r <= 40) and (10 <= g <= 40) and (10 <= b <= 40):
        return "red"
    elif (0 <= r <= 5) and (0 <= g <= 5) and (0 <= b <= 5):
        return "black"
    return "unknown"

def process_image(frame, block_model, plate_model, previous_blocks=None, frame_count=None, fps=None):
    block_results = block_model.predict(
        source=frame,
        show=False,
        save=True,
        conf=0.5
    )
    
    plate_results = plate_model.predict(
        source=frame,
        show=False,
        save=True,
        conf=0.5
    )
    
    current_blocks = []
    current_plates = []
    
    # Keep track of block IDs
    current_block_types = set()
    if previous_blocks:
        last_id = max([block.get('id', 0) for block in previous_blocks])
    else:
        last_id = 0
    
    for result in block_results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Get color at block position
            color = get_color_label(frame, center_x, center_y)
            
            # Calculate block dimensions
            width_pixels = x2 - x1
            height_pixels = y2 - y1
            
            block_type = block_model.names[int(class_id)]
            current_block_types.add(block_type)
            
            block = {
                "type": block_type,
                "color": color,
                "position": {
                    "x": center_x,
                    "y": center_y,
                },
                "dimensions_pixels": {
                    "width": width_pixels,
                    "height": height_pixels
                },
                "confidence": round(confidence, 2),
            }
            
            # Add ID and timestamp for video processing
            if frame_count is not None and fps is not None:
                # Assign ID based on type changes
                if previous_blocks:
                    matching_blocks = [b for b in previous_blocks if b['type'] == block_type]
                    if matching_blocks:
                        block['id'] = matching_blocks[0]['id']
                    else:
                        last_id += 1
                        block['id'] = last_id
                else:
                    block['id'] = 1
                
                # Add timestamp
                seconds = frame_count / fps
                timestamp = str(timedelta(seconds=seconds))
                block['timestamp'] = timestamp
            
            current_blocks.append(block)

    for result in plate_results:
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection
            plate = {
                "type": plate_model.names[int(class_id)],
                "position": {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                },
                "length": y2 - y1,
                "width": x2 - x1,
                "confidence": round(confidence, 2),
            }
            current_plates.append(plate)
            
    return current_blocks, current_plates

def calculate_distance(block, plate):
    # Previous calculation code remains the same
    circle_center_x = plate['position']['x']
    circle_center_y = plate['position']['y']
    circle_image_diameter = max(plate['length'], plate['width'])
    circle_real_diameter = 36  # cm

    object_center_x = block['position']['x']
    object_center_y = block['position']['y']

    image_distance = math.sqrt(
            (circle_center_x - object_center_x) ** 2 +
            (circle_center_y - object_center_y) ** 2
        )
        
    scale_ratio = circle_real_diameter / circle_image_diameter
    
    # Calculate block dimensions in cm
    block_width_cm = block['dimensions_pixels']['width'] * scale_ratio
    block_height_cm = block['dimensions_pixels']['height'] * scale_ratio
    
    real_distance = image_distance * scale_ratio
    
    x_axis_image_distance = object_center_x - circle_center_x
    y_axis_image_distance = circle_center_y - object_center_y

    x_axis_real_distance = x_axis_image_distance * scale_ratio
    y_axis_real_distance = y_axis_image_distance * scale_ratio

    distance_information = {
        'circle_center_x': circle_center_x,
        'circle_center_y': circle_center_y,
        'circle_image_diameter': circle_image_diameter,
        'object_center_x': object_center_x,
        'object_center_y': object_center_y,
        'image_distance': image_distance,
        'scale_ratio': scale_ratio,
        'real_distance': real_distance,
        'x_axis_image_distance': x_axis_image_distance,
        'y_axis_image_distance': y_axis_image_distance,
        'x_axis_real_distance': x_axis_real_distance,
        'y_axis_real_distance': y_axis_real_distance,
        'block_width_cm': block_width_cm,
        'block_height_cm': block_height_cm
    }

    return distance_information

def draw_detections(frame, blocks, plates):
    if not plates:
        return frame
    
    for block in blocks:
        distance_information = calculate_distance(block, plates[0])

        circle_center_x = distance_information['circle_center_x']
        circle_center_y = distance_information['circle_center_y']
        object_center_x = distance_information['object_center_x']
        object_center_y = distance_information['object_center_y']
        real_distance = distance_information['real_distance']
        x_axis_real_distance = distance_information['x_axis_real_distance']
        y_axis_real_distance = distance_information['y_axis_real_distance']
        block_width_cm = distance_information['block_width_cm']
        block_height_cm = distance_information['block_height_cm']
        
        # Draw circles and connecting line
        cv2.circle(frame, (int(circle_center_x), int(circle_center_y)), 
                   5, (0, 0, 255), -1)
        cv2.circle(frame, (int(object_center_x), int(object_center_y)), 
                   5, (0, 0, 255), -1)
        cv2.line(frame, (int(circle_center_x), int(circle_center_y)),
                 (int(object_center_x), int(object_center_y)), (0, 0, 255), 2)

        # Draw arrows
        x_arrow_end = (int(object_center_x), int(circle_center_y))
        cv2.arrowedLine(frame, 
                        (int(circle_center_x), int(circle_center_y)), 
                        x_arrow_end, (0, 255, 0), 2, tipLength=0.2)
        
        y_arrow_end = (int(circle_center_x), int(object_center_y))
        cv2.arrowedLine(frame, 
                        (int(circle_center_x), int(circle_center_y)), 
                        y_arrow_end, (255, 255, 0), 2, tipLength=0.2)
        
        # Set text position
        text_start_x = frame.shape[1] - 300
        text_start_y = frame.shape[0] // 2 - 50
        
         # 確定方向，顯示正負號
        dx_direction = "+" if distance_information['x_axis_real_distance'] >= 0 else "-"
        dy_direction = "+" if distance_information['y_axis_real_distance'] >= 0 else "-"

        # Display text
        cv2.putText(frame, f"Type: {block['type']}", 
                    (text_start_x, text_start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Color: {block['color']}", 
                    (text_start_x, text_start_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"BoundarySize: {block_width_cm:.1f}x{block_height_cm:.1f} cm", 
                    (text_start_x, text_start_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Distance: {real_distance:.2f} cm", 
                    (text_start_x, text_start_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"dx: {x_axis_real_distance:.2f} cm", 
                    (text_start_x, text_start_y + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"dy: {y_axis_real_distance:.2f} cm", 
                    (text_start_x, text_start_y + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add ID and timestamp for video
        if 'id' in block:
            cv2.putText(frame, f"ID: {block['id']}", 
                        (text_start_x, text_start_y + 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if 'timestamp' in block:
            cv2.putText(frame, f"Time: {block['timestamp']}", 
                        (text_start_x, text_start_y + 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def detect(opt):
    block_model = YOLO("block.pt")
    plate_model = YOLO("plate.pt")

    previous_blocks = None
    all_blocks = []  # Store all blocks for JSON output

    if opt.source == "camera":
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_blocks, current_plates = process_image(frame, block_model, plate_model)
            frame = draw_detections(frame, current_blocks, current_plates)
            
            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        
    elif is_video_file(opt.source):
        cap = cv2.VideoCapture(opt.source)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = 'labeled_' + opt.source
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}")
            
            current_blocks, current_plates = process_image(
                frame, 
                block_model, 
                plate_model,
                previous_blocks,
                frame_count,
                fps
            )
            
            # Add distance information to blocks
            if current_plates:
                for block in current_blocks:
                    distance_info = calculate_distance(block, current_plates[0])
                    block['distance_information'] = distance_info
                    all_blocks.append(block)
            
            frame = draw_detections(frame, current_blocks, current_plates)
            
            cv2.imshow("Detections", frame)
            out.write(frame)
            previous_blocks = current_blocks
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        save_json_output(all_blocks, opt.source, is_video=True)
        print(f"Video processing is complete and has been saved to {output_path}")
        print(f"JSON files have been saved in the json directory")
        
    else:
        frame = cv2.imread(opt.source)
        if frame is None:
            print(f"Can't read image: {opt.source}")
            return
            
        current_blocks, current_plates = process_image(frame, block_model, plate_model)
        
        # Add distance information to blocks
        if current_plates:
            for block in current_blocks:
                distance_info = calculate_distance(block, current_plates[0])
                block['distance_information'] = distance_info
        
        frame = draw_detections(frame, current_blocks, current_plates)
        
        cv2.imshow("Detections", frame)
        cv2.waitKey(0)
        
        output_path = 'labeled_' + opt.source
        cv2.imwrite(output_path, frame)
        save_json_output(current_blocks, opt.source, is_video=False)
        print(f"Image processing is complete and has been saved to {output_path}")
        print(f"JSON file has been saved in the json directory")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bamboleo Detection Script")
    parser.add_argument(
        "--source", 
        type=str, 
        required=True, 
        help="Path to the input image, video file, or 'camera' for live detection"
    )

    opt = parser.parse_args()
    detect(opt)