import cv2
import pickle
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from FaceDetector import MTCNNFaceDetector

MASK_PTS_FILE = 'mask_pts.pkl'
TRI_MASK_IDX = [[0, 1, 3], [3, 1, 4], [3, 4, 6], [6, 4, 7],
                [4, 7, 8], [4, 5, 8], [1, 5, 4], [1, 2, 5]]
DEFAULT_TRI_FACE_IDX = [[1, 28, 3], [3, 28, 30], [3, 30, 5], [5, 30, 8],
                        [30, 8, 11], [30, 13, 11], [28, 13, 30], [28, 15, 13]]
DEFAULT_MASK_PTS = np.array([(30, 12), (125, 5), (220, 12), (20, 80), (125, 80),
                             (230, 80), (65, 140), (125, 160), (185, 140)])


def get_tri_mask_points(pts_mask, tri_mask_idx):
    tri_mask_pts = np.zeros((len(tri_mask_idx), 6), dtype=np.float32)
    for i in range(len(tri_mask_idx)):
        tri_mask_pts[i] = pts_mask[tri_mask_idx[i]].ravel()
    return tri_mask_pts


def closest_point(pt, pts):
    dist = np.sum((pts - pt) ** 2, axis=1)
    return np.argmin(dist), np.min(dist)



class FaceMasker:
    def __init__(self, mask_pts_file=MASK_PTS_FILE):
        self.masks_pts_file = MASK_PTS_FILE

        self.num_pts = 9
        self.tri_mask_idx = TRI_MASK_IDX
        self.tri_face_idx = DEFAULT_TRI_FACE_IDX
        self.masks = None
        self.load_mask()

    def load_mask(self):
        try:
            with open(self.masks_pts_file, 'rb') as file:
                masks = pickle.load(file)

            self.masks = []
            for m in masks:
                try:
                    if os.path.exists(m['file']):
                        print(f"Loading mask file '{m['file']}'...")
                        img = cv2.imread(m['file'], cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                            self.masks.append({'img': img,
                                            'pts': m['pts'],
                                            'tri': get_tri_mask_points(m['pts'], self.tri_mask_idx)})
                        else:
                            raise FileNotFoundError(f"File exists but cannot be read: {m['file']}")
                    else:
                        raise FileNotFoundError(f"File not found: {m['file']}")
                except Exception as e:
                    print(f"Error loading mask file '{m['file']}': {e}")

        except Exception as e:
            print(f"Error loading mask file: {e}")

        finally:
            print(f"Loaded {len(self.masks)} mask(s).")

    def get_tri_face_points(self, shape_pts):
        tri_face = np.zeros((len(self.tri_face_idx), 6), dtype=np.float32)
        for i in range(len(self.tri_face_idx)):
            for j in range(3):
                pt = shape_pts[self.tri_face_idx[i][j]]
                tri_face[i, [j+j, j+j+1]] = pt.x, pt.y
        return tri_face

    def wear_mask_to_face(self, image, face_shape, mask_idx=None):
        if mask_idx is None:
            mask_idx = np.random.randint(len(self.masks))

        image_mask = self.masks[mask_idx]['img']
        tri_mask_pts = self.masks[mask_idx]['tri']
        tri_face = self.get_tri_face_points(face_shape)

        image_face = Image.fromarray(image)
        masked_images = []  # Danh sách để lưu trữ các ảnh tam giác đã xử lý

        for idx, (pts1, pts2) in enumerate(zip(tri_mask_pts, tri_face)):
            pts1 = pts1.copy().reshape(3, 2)
            pts2 = pts2.copy().reshape(3, 2)

            rect1 = cv2.boundingRect(pts1)
            pts1[:, 0] = pts1[:, 0] - rect1[0]
            pts1[:, 1] = pts1[:, 1] - rect1[1]

            croped_tri_mask = image_mask[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]]

            rect2 = cv2.boundingRect(pts2)
            pts2[:, 0] = pts2[:, 0] - rect2[0]
            pts2[:, 1] = pts2[:, 1] - rect2[1]

            mask_croped = np.zeros((rect2[3], rect2[2]), np.uint8)
            cv2.fillConvexPoly(mask_croped, pts2.astype(np.int32), 255)

            M = cv2.getAffineTransform(pts1, pts2)
            warped = cv2.warpAffine(croped_tri_mask, M, (rect2[2], rect2[3]))
            warped = cv2.bitwise_and(warped, warped, mask=mask_croped)

            warped = Image.fromarray(warped)
            image_face_copy = image_face.copy()
            image_face_copy.paste(warped, (rect2[0], rect2[1]), warped)
            masked_images.append(np.array(image_face_copy))

            # # Hiển thị từng ảnh tam giác
            # plt.figure()
            # plt.imshow(masked_images[-1])
            # plt.title(f'Triangle {idx + 1}')
            # plt.show()

        return masked_images

class AugmentMasking:
    def __init__(self, mask_chance=0.5, post_augment=None, mask_pts_file=MASK_PTS_FILE):
        self.detector = MTCNNFaceDetector()
        self.chance = mask_chance
        self.masker = FaceMasker(mask_pts_file=mask_pts_file)
        self.post_augment = post_augment

    def __call__(self, image, mask=True):
        is_mask = False
        if np.random.rand() < self.chance and mask:
            face, shapes, _ = self.detector.get_faces(image, return_shape=True)
            if face is not None:
                image = self.masker.wear_mask_to_face(image, shapes[0].parts())
                is_mask = True
        if self.post_augment is not None:
            image = self.post_augment(image=image)
        return image, is_mask


if __name__ == '__main__':

    masker = FaceMasker()
    detector = MTCNNFaceDetector()  # Use MTCNNFaceDetector instead of the previous FaceDetector

    cam = cv2.VideoCapture(0)  # Adjust the camera index if needed
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        faces = detector.get_faces(frame, return_shape=True)
        detector.show_detected(frame)
        if faces is not None:
            faces, shaped, _ = faces
            masked_images = masker.wear_mask_to_face(frame, shaped[0].parts())

            # Hiển thị ảnh gốc
            cv2.imshow('Original Image', frame)

            for idx, masked_image in enumerate(masked_images):
                # Hiển thị từng ảnh tam giác đã xử lý sử dụng OpenCV
                cv2.imshow(f'Masked Image {idx + 1} {masked_image.shape}', masked_image)


        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cam.release()
    cv2.destroyAllWindows()



