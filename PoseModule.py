import cv2
import mediapipe as mp
import time
import math
import numpy as np

class PoseDetector:
    def __init__(self, static_image_mode=False, 
                 upper_body_only=False, 
                 smooth_landmarks=True, 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        """
        Initializes the pose detection model using MediaPipe.

        Parameters:
        - static_image_mode (bool): If True, treats each frame as a static image (no tracking).
        - upper_body_only (bool): Deprecated, no longer used in current versions of MediaPipe.
        - smooth_landmarks (bool): Whether to smooth landmark values for better tracking stability.
        - min_detection_confidence (float): Minimum confidence required to detect a pose.
        - min_tracking_confidence (float): Minimum confidence to keep tracking a detected pose.
        """
        self.mpDraw = mp.solutions.drawing_utils  # Drawing utilities (for landmark lines and circles)
        self.mpPose = mp.solutions.pose  # Pose module from MediaPipe

        # Create a Pose detector instance
        self.pose_detector = self.mpPose.Pose(static_image_mode=static_image_mode,
                                              model_complexity=1,
                                              smooth_landmarks=smooth_landmarks,
                                              min_detection_confidence=min_detection_confidence,
                                              min_tracking_confidence=min_tracking_confidence)

    def findPose(self, img, draw=True):
        """
        Detects the body pose on the input image.

        Parameters:
        - img (ndarray): Input image (BGR).
        - draw (bool): If True, draws the landmarks and connections on the image.

        Returns:
        - The image with pose drawings if draw=True, otherwise the original image.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose_detector.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img

    def getPoseLandmarks(self, img):
        """
        Extracts the landmark positions from the detected pose.

        Parameters:
        - img (ndarray): Input image (used for dimension scaling).

        Returns:
        - A list of landmarks in the format [id, x, y].
        """
        self.lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
        return self.lmlist

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Calculates the angle formed by three landmarks (joints).

        Parameters:
        - img (ndarray): Image for drawing (optional).
        - p1, p2, p3 (int): Landmark indices (joint points).
        - draw (bool): If True, draws angle visualizations on the image.

        Returns:
        - Angle in degrees between the three points (p1-p2-p3).
        """
        if self.lmlist:
            x1, y1 = self.lmlist[p1][1:]
            x2, y2 = self.lmlist[p2][1:]
            x3, y3 = self.lmlist[p3][1:]

            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                 math.atan2(y1 - y2, x1 - x2))
            if angle < 0:
                angle += 360

            if draw:
                # Draw lines
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.line(img, (x2, y2), (x3, y3), (255, 255, 0), 3)

                # Draw points
                for x, y in [(x1, y1), (x2, y2), (x3, y3)]:
                    cv2.circle(img, (x, y), 10, (0, 0, 255), cv2.FILLED)
                    cv2.circle(img, (x, y), 15, (0, 0, 255), 2)

                # Draw angle text
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            return angle
        else:
            return 0

    def findPercentage(self, img, minAngle, maxAngle, border_1_from, border_2_end,
                       p1, p2, p3, drawAngle=False, drawPercentage=True):
        """
        Calculates a percentage value (0-100%) based on the joint angle.

        Useful for showing progress in an exercise (e.g., squats, curls).

        Parameters:
        - img (ndarray): Image for drawing (optional).
        - minAngle (float): Angle representing 100% (fully extended).
        - maxAngle (float): Angle representing 0% (fully bent).
        - border_1_from (int): Starting percentage value.
        - border_2_end (int): Ending percentage value.
        - p1, p2, p3 (int): Landmark indices.
        - drawAngle (bool): Whether to draw the angle.
        - drawPercentage (bool): Whether to draw the percentage near the joint.

        Returns:
        - Tuple: (percentage, actual_angle)
        """
        angle = self.findAngle(img, p1, p2, p3, drawAngle)
        percentage = np.interp(angle, (minAngle, maxAngle), (border_1_from, border_2_end))

        if (drawAngle or drawPercentage) and self.lmlist:
            x2, y2 = self.lmlist[p2][1:]
            cv2.putText(img, f"{int(percentage)}%", (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return (percentage, angle)

# --- Test block for debugging purposes ---
def main():
    cap = cv2.VideoCapture("Videos/football.mp4")
    detector = PoseDetector()
    pTime = 0 
    
    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lm_list = detector.getPoseLandmarks(img)

        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (25, 75),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
