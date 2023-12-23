import cv2
import dlib
import numpy as np
from PIL import Image
from Models.mtcnn import MTCNN, extract_face

class BaseFaceDetector:
    def __init__(self, out_size=160):
        self.out_size = out_size
        self.shaper = dlib.shape_predictor('./Models/models/shape_predictor_68_face_landmarks.dat')

    def align_faces(self, image, faces):
        aligned_faces = []
        landmarks = []

        for face in faces:
            # Extract coordinates from the 1D array
            left, top, right, bottom = map(int, face)
            
            # Convert the coordinates to a dlib rectangle
            face_rect = dlib.rectangle(left, top, right, bottom)
            
            shaped = self.shaper(image, face_rect)
            landmarks.append(shaped)
            aligned = dlib.get_face_chip(image, shaped, size=self.out_size)
            aligned_faces.append(aligned)

        return np.array(aligned_faces), landmarks

class HOGFaceDetector(BaseFaceDetector):
    def __init__(self, out_size=160):
        super().__init__(out_size)
        self.detector = dlib.get_frontal_face_detector()

    def get_faces(self, image, scale=1, get_largest=True, return_shape=False):
        faces = self.detector(image, scale)
        if len(faces) > 0:
            if get_largest:
                idx = np.argmax([rec.width() * rec.height() for rec in faces])
                faces = [faces[idx]]

            aligned_faces, landmarks = self.align_faces(image, faces)

            if return_shape:
                return aligned_faces, landmarks
            return aligned_faces
        else:
            print("No faces detected.")
            return None

    def show_detected(self, image):
        faces = self.get_faces(image, return_shape=True)
        if faces is not None:
            faces, landmarks, pts = faces
            res = image.copy()
            print("Number of faces:", len(faces))

            for i, (face, pt) in enumerate(zip(faces, pts)):
                rec = landmarks[i].rect
                res = cv2.rectangle(res, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 0, 255), 2)
                for pt in landmarks[i].parts():
                    res = cv2.circle(res, (pt.x, pt.y), 2, (0, 0, 255), -1)

                # Remove the loop for pt, as it represents a single point, not an iterable
                res = cv2.circle(res, (pt.x, pt.y), 3, (255, 0, 0), 2)

            cv2.imshow("MTCNN Detected Faces", res)
            cv2.waitKey(1)  # Adjust the waitKey value if needed



class MTCNNFaceDetector(BaseFaceDetector):
    def __init__(self, out_size=160):
        super().__init__(out_size)
        self.detector = MTCNN(image_size=out_size, keep_all=True)

    def get_faces(self, image, get_largest=True, return_shape=False):
        faces, _, pts = self.detector.detect(Image.fromarray(image), landmarks=True)
        if faces is not None and len(faces) > 0:
            if get_largest:
                faces = [faces[0]]

            aligned_faces, landmarks = self.align_faces(image, faces)
            print("Number of faces:", len(faces))


            if return_shape:
                return aligned_faces, landmarks, pts
            return aligned_faces
        else:
            print("No faces detected.")
            return None

    def show_detected(self, image):
        faces = self.get_faces(image, return_shape=True)
        if faces is not None:
            faces, landmarks, pts = faces
            res = image.copy()
            print("Number of faces:", len(faces))

            for i, (face, landmarks_i) in enumerate(zip(faces, landmarks)):
                rec = landmarks_i.rect
                res = cv2.rectangle(res, (rec.left(), rec.top()), (rec.right(), rec.bottom()), (0, 0, 255), 2)
                
                # Iterate over the facial landmarks
                n=0
                for pt in landmarks_i.parts():
                    res = cv2.circle(res, (pt.x, pt.y), 2, (0, 0, 255), -1)
                    cv2.putText(res, str(n), (pt.x, pt.y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),
                                1, cv2.LINE_AA)
                    image = res.copy()
                    n+=1

        cv2.imshow("MTCNN Detected Faces", image)

        
 

if __name__ == '__main__':
    # hogdt = HOGFaceDetector()
    mtcnn = MTCNNFaceDetector()

    # Open a connection to the camera (in this case, the default camera (0))
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the HOG detected faces
        # hogdt.show_detected(frame)

        # Display the MTCNN detected faces
        mtcnn.show_detected(frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
