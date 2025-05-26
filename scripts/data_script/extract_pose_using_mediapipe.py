import cv2
import mediapipe as mp
import os
import json


class PoseDetector():
    def __init__(self, staticImageMode=False, modelComplexity=1, smoothLandmarks=True, enableSegmentation=False,
                 smoothSegmentation=True, minDetectionConfidence=0.5, minTrackingConfidence=0.5):
        self.staticImageMode = staticImageMode
        self.modelComplexity = modelComplexity
        self.smoothLandmarks = smoothLandmarks
        self.enableSegmentation = enableSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.staticImageMode, self.modelComplexity, self.smoothLandmarks,
                                     self.enableSegmentation, self.smoothSegmentation, self.minDetectionConfidence,
                                     self.minTrackingConfidence)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img, self.results

    def findPosition(self, img, draw=False):
        lmList = []
        lmListShow = []
        if self.results.pose_world_landmarks:
            for id, lm in enumerate(self.results.pose_world_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = lm.x, lm.y
                lmList.append([id,round(float(cx),3), round(float(cy),3), round(float(lm.z),3)])
                lmListShow.append({"id":id,"x":cx * w, "y":cy * h, "z": lm.z})
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
    
import numpy as np

def create_mask(w,h):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.rectangle(mask, (0, 0), (w//2, h), 255, -1)
    return mask

def skeletal_extraction(video_path, output_path,do_mask=False,mask_func = None):
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frames = []
    while True:
        success, img = cap.read()
        if not success:
            break
        if do_mask and mask_func is not None and callable(mask_func):
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            mask = mask_func(width, height)
            mask_3d = cv2.merge([mask, mask, mask])
            img = cv2.bitwise_and(img, mask_3d)

        img, results = detector.findPose(img)
        lmList = detector.findPosition(img)

        frame_data = {'landmarks': lmList}
        frames.append(frame_data)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    with open(output_path, 'w') as f:
        json.dump(frames, f)
    return frames

def parse_pose_from_video(root_folder,mask_func = None,do_mask = False):
    for foldername, subfolders, filenames in os.walk(root_folder):
        if os.path.basename(foldername) == 'remote':
            for filename in filenames:
                if filename.endswith('.mp4') or filename.endswith('.mkv'):
                    output_folder = os.path.join(*foldername.split("/")[0:-2], 'pose')
                    os.makedirs(output_folder, exist_ok=True)
                    pose = skeletal_extraction(os.path.join(foldername, filename), os.path.join(output_folder,'pose.json'),do_mask=do_mask,mask_func = mask_func)


if __name__ == '__main__':
    root_folder = r"../sampledata"
    do_mask = True
    mask_func = create_mask
    parse_pose_from_video(root_folder,mask_func=mask_func,do_mask=do_mask)