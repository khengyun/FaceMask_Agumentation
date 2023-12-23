import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from FaceDetector import MTCNNFaceDetector  # Import class FaceDetector from module FaceDetector

SHOW_LOG = False


class Cropper:
    def __init__(self, landmarks_to_crop):
        self.landmarks_to_crop = landmarks_to_crop

    def crop_by_landmarks(self, image, landmarks):
        min_x, max_x, min_y, max_y = float('inf'), 0, float('inf'), 0

        for landmark_index in self.landmarks_to_crop:
            try:
                x, y = landmarks.part(landmark_index).x, landmarks.part(landmark_index).y
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
            except Exception as e:
                if SHOW_LOG:
                    print(f"Error in cropping: {e}")

        min_x, max_x, min_y, max_y = int(min_x), int(max_x), int(min_y), int(max_y)
        cropped_image = image[min_y:max_y, min_x:max_x]
        return cropped_image

class FaceCropper:
    def __init__(self, out_size=100):
        self.detector = MTCNNFaceDetectorWithCropper(out_size)

    def preprocess_image(self, image):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            return image
        except Exception as e:
            if SHOW_LOG:
                print(f"Error in preprocessing image: {e}")
            return None

    def crop_faces_and_concat(self, image, mask):
        try:
            # resize image with opencv . resize to 244x244
            image = cv2.resize(image, (244, 244))

            if mask:
                landmarks_to_crop = [19, 24, 1, 15]
                cropped_faces = self.detector.crop_faces_by_landmarks(image, landmarks_to_crop, return_shape=True)
                if cropped_faces is not None:
                    cropped_faces, landmarks, _ = cropped_faces
                    # Resize and concatenate the cropped faces
                    faces=self.resize_and_preprocess(cropped_faces[0])
                    concatenated_faces = torch.cat([faces,faces], dim=0)
                    return concatenated_faces
                else: 
                    faces=self.resize_and_preprocess(Image.fromarray(image))
                    return torch.cat([faces,faces], dim=0)
                
            else:
                landmarks_to_crop = list(range(68))  # Use all 68 landmark points when mask is False

                cropped_faces = self.detector.crop_faces_by_landmarks(image, landmarks_to_crop, return_shape=True)
                resized_image = self.resize_and_preprocess(Image.fromarray(image))

                if cropped_faces is not None:
                    _, landmarks, _ = cropped_faces
                    for cropped_face in cropped_faces[0]:
                        resized_face = self.resize_and_preprocess(cropped_face)
                        resized_image = torch.cat([resized_image, resized_face], dim=0)
                        
                else: 
                    faces=self.resize_and_preprocess(Image.fromarray(image))
                    return torch.cat([faces,faces], dim=0)              

                return resized_image
        except Exception as e:
            faces=self.resize_and_preprocess(Image.fromarray(image))
            if SHOW_LOG:
                print(f"Error in cropping faces and concatenating: {e}")
            return torch.cat([faces,faces], dim=0)
        # finally:
        #     faces=self.resize_and_preprocess(Image.fromarray(image))
        #     return torch.cat([faces,faces], dim=0)

    def resize_and_preprocess(self, image):
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            return transform(image)
        except Exception as e:
            # print(image.shape)
            if SHOW_LOG:
                print(f"Error in resizing and preprocessing: {e}")
            return None

class MTCNNFaceDetectorWithCropper(MTCNNFaceDetector):
    def __init__(self, out_size=160):
        super().__init__(out_size)
        self.cropper = Cropper([])  # Cropper will be initialized with landmarks dynamically

    def crop_faces_by_landmarks(self, image, landmarks_to_crop, get_largest=True, return_shape=False):
        try:
            self.cropper.landmarks_to_crop = landmarks_to_crop
            faces, _, pts = self.detector.detect(Image.fromarray(image), landmarks=True)

            if faces is not None and len(faces) > 0:
                if get_largest:
                    faces = [faces[0]]

                landmarks = self.align_faces(image, faces)[1]
                # print("Number of faces:", len(faces))

                cropped_faces = [self.cropper.crop_by_landmarks(image, landmark) for landmark in landmarks]

                if return_shape:
                    return cropped_faces, landmarks, pts
                return cropped_faces
            else:
                if SHOW_LOG:
                    print("No faces detected.")
                return None
        except Exception as e:
            if SHOW_LOG:
                print(f"Error in cropping faces by landmarks: {e}")
            return None

    def show_cropped_faces(self, image):
        try:
            cropped_faces = self.crop_faces_by_landmarks(image, return_shape=True)
            if cropped_faces is not None:
                cropped_faces, landmarks, pts = cropped_faces
                res = image.copy()
                # print("Number of faces:", len(cropped_faces))

                for i, (cropped_face, landmarks_i) in enumerate(zip(cropped_faces, landmarks)):
                    rec = landmarks_i.rect
                    res = cv2.rectangle(res, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 0, 255), 2)

                    cv2.imshow(f"Cropped Face {i + 1}", cropped_face)
                cv2.imshow("MTCNN Cropped Faces", res)
        except Exception as e:
            if SHOW_LOG:
                print(f"Error in showing cropped faces: {e}")

if __name__ == '__main__':
    try:
        face_cropper = FaceCropper()

        cam = cv2.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            # frame = cv2.flip(frame, 1)

            # Set the mask value based on some condition (e.g., user input)
            mask = True  # Change this based on your condition

            # Process the frame using the FaceCropper class
            result = face_cropper.crop_faces_and_concat(frame, mask)
            print(result.shape)
            reduced_matrix = result[:, :, :3]

            if result is not None:
                result_numpy = result.cpu().detach().numpy()

                reduced_matrix = result_numpy[0:3, :, :]

                cv2.imshow('Processed Frame', reduced_matrix.transpose(1, 2, 0))

            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break

        cam.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error in main execution: {e}")
