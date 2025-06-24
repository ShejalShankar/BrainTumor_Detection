#import os 
#import torch
#from PIL import Image
#import numpy as np
#from RealESRGAN import RealESRGAN
#
#
#def main() -> int:
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    model = RealESRGAN(device, scale=8)
#    model.load_weights('/Users/shaivalvora/Downloads/MRI-Image-Super-Resolution-Using-GANs-main/Real-ESRGAN/weights/RealESRGAN_x8.pth', download=True)
#    image = Image.open("/Users/shaivalvora/Downloads/MRI-Image-Super-Resolution-Using-GANs-main/Real-ESRGAN/Enhancement/Test/volume_2_slice_73_jpg.rf.902bada75838f45043e447b5c7c8716f.jpg").convert('RGB')
#    sr_image = model.predict(image)
#    sr_image.save('/Users/shaivalvora/Downloads/MRI-Image-Super-Resolution-Using-GANs-main/Real-ESRGAN/results/volume_2_slice_73_jpg.rf.902bada75838f45043e447b5c7c8716f.jpg')
#
#if __name__ == '__main__':
#    main()

import os
import torch
from PIL import Image
from RealESRGAN import RealESRGAN

def main() -> int:
    # Define the paths
    input_folder = r"data\BrainTumorDetectionYolov9\BrainTumorDetectionYolov9\test\images"
    output_folder = r"MRI-Image-Super-Resolution-Using-GANs-main\Real-ESRGAN\results\output_test"
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Set up the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=8)
    model.load_weights(r'MRI-Image-Super-Resolution-Using-GANs-main\Real-ESRGAN\weights\RealESRGAN_x8.pth', download=True)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # Open the image
                image = Image.open(input_path).convert('RGB')
                
                # Process the image
                sr_image = model.predict(image)
                
                # Save the super-resolved image
                sr_image.save(output_path)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

if __name__ == '__main__':
    main()
