import os
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
from torchvision import transforms

class PreprocessPipeline:
    def __init__(self):
        self.load_height = 1024
        self.load_width = 768
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Initialize OpenPose (simplified - you may need to adjust based on your OpenPose installation)
        self.pose_model = self._init_pose_model()
        
    def _init_pose_model(self):
        """Initialize OpenPose model for pose detection"""
        # Note: This is a placeholder. You'll need to implement actual OpenPose initialization
        # based on your specific setup and the pose model files in checkpoints/pose/
        try:
            # Construct absolute paths relative to the project root
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prototxt_path = os.path.join(project_root, "checkpoints", "pose", "pose_deploy_linevec.prototxt")
            # Using the available model file (440000 instead of 584000)
            model_path = os.path.join(project_root, "checkpoints", "pose", "pose_iter_584000.caffemodel")
            
            if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
                print(f"Warning: Pose model files not found at {prototxt_path} or {model_path}")
                return None
            
            net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            print("Successfully loaded OpenPose model")
            return net
        except Exception as e:
            print(f"Warning: Could not load pose model. Error: {e}")
            print("Using dummy pose detection.")
            return None
    
    def process_images(self, person_path, cloth_path, session_id):
        """Process person and cloth images for VITON-HD pipeline"""
        
        # Load and resize images
        person_img = Image.open(person_path).convert('RGB')
        cloth_img = Image.open(cloth_path).convert('RGB')
        
        person_img = person_img.resize((self.load_width, self.load_height), Image.LANCZOS)
        cloth_img = cloth_img.resize((self.load_width, self.load_height), Image.LANCZOS)
        
        # Save resized images
        person_resized_path = f"static/uploads/{session_id}_person_resized.jpg"
        cloth_resized_path = f"static/uploads/{session_id}_cloth_resized.jpg"
        person_img.save(person_resized_path)
        cloth_img.save(cloth_resized_path)
        
        # Generate pose estimation
        pose_data, pose_img = self._get_pose(person_img, session_id)
        
        # Generate human parsing (simplified - you may need a proper human parsing model)
        parse_img = self._get_human_parsing(person_img, session_id)
        
        # Generate cloth mask (simplified)
        cloth_mask = self._get_cloth_mask(cloth_img, session_id)
        
        # Generate agnostic representations
        img_agnostic = self._get_img_agnostic(person_img, parse_img, pose_data, session_id)
        parse_agnostic = self._get_parse_agnostic(parse_img, pose_data, session_id)
        
        return {
            'session_id': session_id,
            'person_img': person_resized_path,
            'cloth_img': cloth_resized_path,
            'pose_img': f"static/uploads/{session_id}_pose.png",
            'parse_img': f"static/uploads/{session_id}_parse.png",
            'cloth_mask': f"static/uploads/{session_id}_cloth_mask.png",
            'img_agnostic': f"static/uploads/{session_id}_agnostic.jpg",
            'parse_agnostic': f"static/uploads/{session_id}_parse_agnostic.png",
            'pose_data': pose_data
        }
    
    def _get_pose(self, img, session_id):
        """Extract pose keypoints from person image"""
        img_array = np.array(img)
        
        if self.pose_model is not None:
            # Use actual OpenPose model
            blob = cv2.dnn.blobFromImage(img_array, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
            self.pose_model.setInput(blob)
            out = self.pose_model.forward()
            
            # Extract keypoints (simplified)
            pose_data = self._extract_keypoints(out, img_array.shape)
        else:
            # Dummy pose data for testing
            pose_data = self._get_dummy_pose_data()
        
        # Create pose visualization
        pose_img = self._draw_pose(img, pose_data)
        pose_path = f"static/uploads/{session_id}_pose.png"
        pose_img.save(pose_path)
        
        return pose_data, pose_img
    
    def _extract_keypoints(self, output, img_shape):
        """Extract keypoints from OpenPose output"""
        # This is a simplified implementation
        # You may need to adjust based on your specific OpenPose setup
        H, W = img_shape[:2]
        points = []
        
        # OpenPose has 18 keypoints for COCO model
        for i in range(18):
            # Find maximum probability location for each keypoint
            heatmap = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatmap)
            
            x = int((point[0] * W) / output.shape[3])
            y = int((point[1] * H) / output.shape[2])
            
            if conf > 0.1:  # confidence threshold
                points.append([x, y])
            else:
                points.append([0, 0])  # invisible point
        
        return np.array(points)
    
    def _get_dummy_pose_data(self):
        """Generate dummy pose data for testing"""
        # 18 keypoints in COCO format
        pose_data = np.array([
            [384, 150],  # nose
            [364, 170],  # left_eye
            [404, 170],  # right_eye
            [344, 180],  # left_ear
            [424, 180],  # right_ear
            [304, 250],  # left_shoulder
            [464, 250],  # right_shoulder
            [284, 350],  # left_elbow
            [484, 350],  # right_elbow
            [264, 450],  # left_wrist
            [504, 450],  # right_wrist
            [324, 500],  # left_hip
            [444, 500],  # right_hip
            [314, 650],  # left_knee
            [454, 650],  # right_knee
            [304, 800],  # left_ankle
            [464, 800],  # right_ankle
            [384, 200],  # neck (custom)
        ])
        return pose_data
    
    def _draw_pose(self, img, pose_data):
        """Draw pose keypoints on image"""
        pose_img = img.copy()
        draw = ImageDraw.Draw(pose_img)
        
        # COCO pose connections
        connections = [
            [0, 1], [0, 2], [1, 3], [2, 4],  # head
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # arms
            [5, 11], [6, 12], [11, 12],  # torso
            [11, 13], [13, 15], [12, 14], [14, 16]  # legs
        ]
        
        # Draw connections
        for connection in connections:
            start_point = pose_data[connection[0]]
            end_point = pose_data[connection[1]]
            
            if (start_point[0] > 0 and start_point[1] > 0 and 
                end_point[0] > 0 and end_point[1] > 0):
                draw.line([tuple(start_point), tuple(end_point)], fill='red', width=3)
        
        # Draw keypoints
        for point in pose_data:
            if point[0] > 0 and point[1] > 0:
                draw.ellipse([point[0]-3, point[1]-3, point[0]+3, point[1]+3], 
                           fill='blue', outline='blue')
        
        return pose_img
    
    def _get_human_parsing(self, img, session_id):
        """Generate human parsing map with improved segmentation"""
        try:
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            
            # Initialize parsing map with background (0)
            parse_map = np.zeros((h, w), dtype=np.uint8)
            
            # Convert to grayscale for thresholding
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Use GrabCut for better foreground/background segmentation
            mask = np.zeros(gray.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Define a rectangle around the main subject (adjust as needed)
            rect = (50, 50, w-100, h-100)
            cv2.grabCut(img_array, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create a mask where sure or likely foreground
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Find contours
            contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (main subject)
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(parse_map, [largest_contour], -1, 5, -1)  # Fill with upper clothes (5)
                
                # Refine the parsing (very simplified)
                head_region = (slice(0, h//3), slice(w//4, 3*w//4))
                parse_map[head_region] = 4  # Face/neck
                
                # Add arms (simplified)
                parse_map[h//3:2*h//3, :w//4] = 14  # Left arm
                parse_map[h//3:2*h//3, 3*w//4:] = 15  # Right arm
                
                # Add lower body
                parse_map[2*h//3:, :] = 9  # Pants/skirt
            
            parse_img = Image.fromarray(parse_map, mode='L')
            parse_path = f"static/uploads/{session_id}_parse.png"
            parse_img.save(parse_path)
            
            return parse_img
            
        except Exception as e:
            print(f"Error in human parsing: {e}")
            # Fallback to simple segmentation
            h, w = np.array(img).shape[:2]
            parse_map = np.zeros((h, w), dtype=np.uint8)
            parse_map[h//3:2*h//3, w//4:3*w//4] = 5  # Upper clothes
            parse_map[2*h//3:, w//4:3*w//4] = 9  # Lower body
            return Image.fromarray(parse_map, mode='L')
    
    def _get_cloth_mask(self, cloth_img, session_id):
        """Generate cloth mask with improved background removal"""
        try:
            cloth_array = np.array(cloth_img)
            
            # Convert to grayscale and apply adaptive thresholding
            gray = cv2.cvtColor(cloth_array, cv2.COLOR_RGB2GRAY)
            
            # Use Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if needed (assuming white background)
            if np.mean(gray) > 127:
                mask = 255 - mask
            
            # Morphological operations to clean up the mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours and keep only the largest one
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(mask)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            # Smooth edges
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
            
            mask_img = Image.fromarray(mask, mode='L')
            mask_path = f"static/uploads/{session_id}_cloth_mask.png"
            mask_img.save(mask_path, 'PNG')
            
            return mask_img
            
        except Exception as e:
            print(f"Error in cloth mask generation: {e}")
            # Fallback to simple thresholding
            gray = cv2.cvtColor(np.array(cloth_img), cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            return Image.fromarray(mask, mode='L')
    
    def _get_img_agnostic(self, img, parse_img, pose_data, session_id):
        """Generate person image with clothes removed"""
        parse_array = np.array(parse_img)
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)
        
        # Mask upper body clothes (parsing label 5)
        mask = (parse_array == 5).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask, mode='L')
        
        # Fill the masked area with gray
        agnostic.paste((128, 128, 128), None, mask_img)
        
        # Also mask arms using pose data
        r = 20
        for i in [3, 4, 6, 7]:  # arm keypoints
            if i < len(pose_data) and pose_data[i][0] > 0 and pose_data[i][1] > 0:
                pointx, pointy = pose_data[i]
                agnostic_draw.ellipse(
                    (pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 
                    'gray', 'gray'
                )
        
        agnostic_path = f"static/uploads/{session_id}_agnostic.jpg"
        agnostic.save(agnostic_path)
        
        return agnostic
    
    def _get_parse_agnostic(self, parse_img, pose_data, session_id):
        """Generate parsing map with clothes removed"""
        parse_array = np.array(parse_img)
        agnostic_parse = parse_array.copy()
        
        # Remove upper clothes (set to background)
        agnostic_parse[parse_array == 5] = 0
        
        # Remove arms
        agnostic_parse[parse_array == 14] = 0  # left arm
        agnostic_parse[parse_array == 15] = 0  # right arm
        
        agnostic_parse_img = Image.fromarray(agnostic_parse, mode='L')
        agnostic_parse_path = f"static/uploads/{session_id}_parse_agnostic.png"
        agnostic_parse_img.save(agnostic_parse_path)
        
        return agnostic_parse_img