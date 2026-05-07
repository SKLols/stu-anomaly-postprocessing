##=========== below is coversion for extrinsic file from.txt to json
# import yaml
# import json
# import os

# def convert_extrinsics_to_json(txt_file_path, json_file_path):
#     """
#     Convert extrinsics.txt (YAML format) to JSON
#     """
#     try:
#         # Read the YAML content from txt file
#         with open(txt_file_path, 'r') as f:
#             content = f.read()
        
#         # Split the YAML documents (separated by '---')
#         yaml_documents = content.split('---')
        
#         # Parse each YAML document
#         transforms = []
#         for doc in yaml_documents:
#             if doc.strip():  # Skip empty documents
#                 data = yaml.safe_load(doc)
#                 if data and 'transforms' in data:
#                     transforms.extend(data['transforms'])
        
#         # Create the JSON structure
#         output_data = {
#             "transforms": transforms
#         }
        
#         # Write to JSON file
#         with open(json_file_path, 'w') as f:
#             json.dump(output_data, f, indent=2)
        
#         print(f"✅ Successfully converted {txt_file_path} to {json_file_path}")
#         print(f"📊 Found {len(transforms)} transform entries")
        
#     except Exception as e:
#         print(f"❌ Error converting file: {e}")

# # Your specific file paths
# input_file = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/camera_config/extrinsics.txt"
# output_file = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/camera_config/extrinsics.json"

# # Convert the file
# convert_extrinsics_to_json(input_file, output_file)
#=========== below is coversion for intrinsic file from.txt to json

import yaml
import json
import re

def convert_intrinsics_to_json(txt_file_path, json_file_path):
    """
    Convert camera intrinsics TXT file to JSON
    """
    try:
        # Read the file content
        with open(txt_file_path, 'r') as f:
            content = f.read()
        
        # Split by camera sections (empty lines between cameras)
        # Use a regex to split by camera names followed by colon
        camera_blocks = re.split(r'\n(?=\w+_camera_\d+:)', content.strip())
        
        cameras_data = {}
        
        for block in camera_blocks:
            if not block.strip():
                continue
                
            # Extract camera name from first line
            first_line = block.strip().split('\n')[0]
            camera_name = first_line.replace(':', '').strip()
            
            # Parse the YAML content for this camera
            camera_yaml = block.strip()
            
            # Fix the YAML format - remove the camera name line and parse the rest
            yaml_content = '\n'.join(camera_yaml.split('\n')[1:])
            
            try:
                camera_data = yaml.safe_load(yaml_content)
                cameras_data[camera_name] = camera_data
            except yaml.YAMLError as e:
                print(f"Warning: Could not parse {camera_name}: {e}")
                continue
        
        # Write to JSON file
        with open(json_file_path, 'w') as f:
            json.dump(cameras_data, f, indent=2)
        
        print(f"✅ Successfully converted {txt_file_path} to {json_file_path}")
        print(f"📷 Found {len(cameras_data)} camera configurations:")
        for cam_name in cameras_data.keys():
            print(f"   - {cam_name}")
        
    except Exception as e:
        print(f"❌ Error converting file: {e}")

# Your file paths
input_file = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/camera_config/intrinsics.txt"  # Assuming this is the file
output_file = "/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/camera_config/intrinsics.json"

# Convert the file
convert_intrinsics_to_json(input_file, output_file)