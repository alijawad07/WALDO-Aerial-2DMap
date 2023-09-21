import cv2
import numpy as np
import torch 
import onnxruntime as rt
import os
import random
import argparse
import uuid
import re

CONFIDENCE_THRESHOLD = 0.60
OUTPUT_VIDEO_WIDTH = 1920
OUTPUT_VIDEO_HEIGHT = 1080



def get_resolution_from_model_path(model_path):
    # Check for rectangular pattern first
    rect_match = re.search(r"_rect_(\d+)_(\d+)_", model_path)
    if rect_match:
        return int(rect_match.group(1)), int(rect_match.group(2))
    
    # Check for the square pattern next
    square_match = re.search(r"(\d+)px", model_path)
    if square_match:
        res = int(square_match.group(1))
        return res, res  # Return height and width

    return None, None  # If neither match, return None for both dimensions

def generate_uuid():
    return str(uuid.uuid4())

def scale_based_on_bbox(bbox):
    # Compute the diagonal length of the bounding box
    diag_length = np.sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)

    # Linearly scale the text size and thickness based on the diagonal length
    text_size = max(0.4, diag_length / 300)

    return text_size

def resize_and_pad(frame, expected_width, expected_height):
    if expected_width == expected_height:
        height, width, _ = frame.shape
        new_dim = min(height, width)
        start_x = (width - new_dim) // 2
        start_y = (height - new_dim) // 2
        frame = frame[start_y:start_y+new_dim, start_x:start_x+new_dim]
        
    ratio = min(expected_width / frame.shape[1], expected_height / frame.shape[0])
    new_width = int(frame.shape[1] * ratio)
    new_height = int(frame.shape[0] * ratio)
    frame = cv2.resize(frame, (new_width, new_height))
    padded_frame = np.ones((expected_height, expected_width, 3), dtype=np.uint8)
    y_offset = (expected_height - new_height) // 2
    x_offset = (expected_width - new_width) // 2
    padded_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame

    return padded_frame

def replace_black_with_color(image, color):
    
    # Resize the image to have a maximum dimension of 50 pixels
    max_dimension = 50
    aspect_ratio = image.shape[1] / image.shape[0]  # width/height
    if image.shape[1] > image.shape[0]:  # if width > height
        new_width = max_dimension
        new_height = int(max_dimension / aspect_ratio)
    else:
        new_height = max_dimension
        new_width = int(max_dimension * aspect_ratio)
    image = cv2.resize(image, (new_width, new_height))
    
    # Replace black with the desired color
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.all(image[i, j, :3] == [0, 0, 0]):
                image[i, j, :3] = color
    return image

def draw_grid(img, line_color=(0, 0, 0), thickness=1, type_=cv2.LINE_AA, pxstep=50):
    """
    Draw a grid on the image
    """
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep

    return img

def create_enhanced_2d_map(frame, outputs, names, colors):
    """Create an enhanced 2D map using PNG icons."""
    map_frame = 255 * np.ones_like(frame)
    map_frame = draw_grid(map_frame)
    # Load the PNG icons and replace black with respective colors
    icons = {
        'car': replace_black_with_color(cv2.imread('icons/car.png', cv2.IMREAD_UNCHANGED), colors['car']),
        'van': replace_black_with_color(cv2.imread('icons/van.png', cv2.IMREAD_UNCHANGED), colors['van']),
        'truck': replace_black_with_color(cv2.imread('icons/truck.png', cv2.IMREAD_UNCHANGED), colors['truck']),
        'building': replace_black_with_color(cv2.imread('icons/building.png', cv2.IMREAD_UNCHANGED), colors['building']),
        'human': replace_black_with_color(cv2.imread('icons/human.png', cv2.IMREAD_UNCHANGED), colors['human']),
        'u_pole': replace_black_with_color(cv2.imread('icons/u_pole.png', cv2.IMREAD_UNCHANGED), colors['u_pole']),
        'boat': replace_black_with_color(cv2.imread('icons/boat.png', cv2.IMREAD_UNCHANGED), colors['boat']),
        'bike': replace_black_with_color(cv2.imread('icons/bike.png', cv2.IMREAD_UNCHANGED), colors['bike']),
        'smoke': replace_black_with_color(cv2.imread('icons/smoke.png', cv2.IMREAD_UNCHANGED), colors['smoke']),
        'bus': replace_black_with_color(cv2.imread('icons/bus.png', cv2.IMREAD_UNCHANGED), colors['bus']),
        'container': replace_black_with_color(cv2.imread('icons/container.png', cv2.IMREAD_UNCHANGED), colors['container'])
        # ... add other icons similarly
    }

    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        box_center = (int((x0 + x1) / 2), int((y0 + y1) / 2))
        cls_id = int(cls_id)
        icon = icons[names[cls_id]]

        icon_height, icon_width, _ = icon.shape

        # Calculate the top-left corner of the position where the icon will be placed
        top_left_x = max(0, box_center[0] - icon_width // 2)
        top_left_y = max(0, box_center[1] - icon_height // 2)

        # Calculate the bottom-right corner of the position where the icon will be placed
        bottom_right_x = min(map_frame.shape[1], top_left_x + icon_width)
        bottom_right_y = min(map_frame.shape[0], top_left_y + icon_height)

        # Adjust the top-left corner if the icon goes out of bounds
        if bottom_right_x - top_left_x < icon_width:
            top_left_x = bottom_right_x - icon_width
        if bottom_right_y - top_left_y < icon_height:
            top_left_y = bottom_right_y - icon_height

        for i in range(icon_height):
            for j in range(icon_width):
                if icon[i, j, 3] > 0:  # Check the alpha value to ensure it's not transparent
                    map_frame[top_left_y + i, top_left_x + j] = icon[i, j, :3]

    return map_frame


def process_frame(frame, sess, max_outputs):
    colors = {
    'car': (255, 0, 0),  # Blue
    'van': (255, 0, 0),  # Cyan
    'truck': (0, 255, 0),  # Green
    'building': (0, 42, 92),  # Brown
    'human': (203, 192, 255),  # Pink
    'gastank': (0, 255, 255),  # Yellow
    'digger': (0, 0, 255),  # Red
    'container': (255, 255, 255),  # White
    'bus': (128, 0, 128),  # Purple
    'u_pole': (255, 0, 255),  # Magenta
    'boat': (0, 0, 139),  # Dark red
    'bike': (144, 238, 144),  # Light green
    'smoke': (0, 230, 128),  # Grey
    'solarpanels': (0, 0, 0),  # Black
    }
    names = ['car', 'van', 'truck', 'building', 'human', 'gastank', 'digger', 'container', 'bus', 'u_pole', 'boat', 'bike', 'smoke',
             'solarpanels']

    resolution = 960

    # Initialize a dictionary to store the count of each category (this is not really used in this vid engine)
    category_count = {name: 0 for name in names}

    image = frame.copy()
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255

    inp = {input_name: im}
    outputs = sess.run(None, inp)[0]
    thickness = 1
    category_counts = {}

    # Create an empty list to store detections for the map
    detections_for_map = []

    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        if score < CONFIDENCE_THRESHOLD:
            continue
        box = np.array([x0, y0, x1, y1])
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 1)
        name = names[cls_id]
        if name == 'van':  # Map van detections to car
            name = 'car'
        color = colors[name]
        name += ' ' + str(score)
        
        text_size= scale_based_on_bbox(box)

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness)
        cv2.putText(frame, name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, text_size, color, thickness=thickness)
        if max_outputs is not None:
            cv2.putText(frame, f"ONNX network max Outputs: {max_outputs}", (frame.shape[1] - 250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


        # Get the name of the class without the score
        class_name = name.split()[0]

        # Increment the count for this class in the dictionary
        if class_name in category_counts:
            category_counts[class_name] += 1
        else:
            category_counts[class_name] = 1

        # Append detections to the list
        detections_for_map.append((batch_id, x0, y0, x1, y1, cls_id, score))
   


    # Write the category counts on the frame
    y_position = 20  # Initial y position
    for category, count in category_counts.items():
        cv2.putText(frame, f"{category}: {count}", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_position += 20  # Increment the y position for the next text

     # Create enhanced 2D map
    map_frame = create_enhanced_2d_map(frame, detections_for_map, names, colors)
    
    # Concatenate processed frame and 2D map
    combined_frame = np.hstack((frame, map_frame))
    combined_frame = cv2.resize(combined_frame, (OUTPUT_VIDEO_WIDTH, OUTPUT_VIDEO_HEIGHT))


    # Return the combined frame with detection boxes and 2D map
    return combined_frame

    # # Return the frame with detection boxes
    # return frame

def save_image(output_frame, model, frame_count, UUID):
    # Create the output directory if it does not exist
    if not os.path.exists(f"./output_images/{model[:-5]}"):
        os.makedirs(f"./output_images/{model[:-5]}")

    cv2.imwrite(f"./output_images/{model[:-5]}/{UUID}_frame_{frame_count}.jpg", output_frame)



# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--frame_limit', type=int, default=None,
                    help='frame count limit (default: unlimited)')
parser.add_argument('--frame_skip', type=int, default=1,
                    help='frame skip count')
args = parser.parse_args()

for model in os.listdir("./"):
    if model.endswith(".onnx"):
        expected_width, expected_height = get_resolution_from_model_path(model)  # Get expected resolution

        max_outputs_match = re.search(r"topk.\d+", model)
        max_outputs = None
        if max_outputs_match:
            max_outputs = int(re.search(r"\d+", max_outputs_match.group(0)).group(0))



        # Initialize video writer for the output video with the expected resolution
        out = cv2.VideoWriter(f"./output_vids/{model[:-5]}.mp4", 
                       cv2.VideoWriter_fourcc(*'mp4v'), 
                       24, 
                       (OUTPUT_VIDEO_WIDTH, OUTPUT_VIDEO_HEIGHT))


        print(f"Processing model: {model}")
        cuda = torch.cuda.is_available()


        # Initialize ONNX runtime session with CUDA execution
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        

        sess = rt.InferenceSession("./" + model, providers=providers)

        input_name = sess.get_inputs()[0].name

        # Iterate over video files
        for video_file in sorted(os.listdir("./input_vids")): # sort the list to maintain the same order
            if video_file.endswith(".mp4"):
                print(f"Processing video: {video_file}")

                # Initialize video capture and writer
                cap = cv2.VideoCapture("./input_vids/" + video_file)
                width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
                fps = cap.get(cv2.CAP_PROP_FPS)

                frame_count = 0
                
                uuid_of_vid_base = generate_uuid()

                # Process frames
                while cap.isOpened() and (args.frame_limit is None or frame_count < args.frame_limit):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % args.frame_skip == 0:
                        print("processing frame number : " + str(frame_count))
                        
                        ret, frame = cap.read()
                    if ret:
                        # Prepare the frame
                        frame = resize_and_pad(frame, expected_width, expected_height)
                        
                        # Process frame
                        output_frame = process_frame(frame, sess, max_outputs)
                        # Show the output frame in real-time
                        cv2.imshow("Processed Video and 2D Map", output_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
                        out.write(output_frame)
                    else:
                        break
                    frame_count += 1

                # Release capture and writer
            # Release capture and writer
            cap.release()

        out.release()

    print(f"Finished processing model: {model}")
