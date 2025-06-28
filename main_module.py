import cv2
import mediapipe as mp
import time
import PoseModule as pm
import numpy as np

# Constants for window
FRAME_WIDTH = 900
FRAME_HEIGHT = 720
COUNTDOWN_DURATION = 3

def show_start_prompt(img):
    """Show blinking warning on the screen"""
    if int(time.time() * 2) % 2 == 0:
        cv2.putText(img, "Press SPACE to start", (250, 100),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)

def show_countdown(img, start_time):
    """Countdown"""
    seconds_elapsed = int(time.time() - start_time)
    countdown = COUNTDOWN_DURATION - seconds_elapsed
    if countdown > 0:
        cv2.putText(img, str(countdown), (420, 200),
                    cv2.FONT_HERSHEY_COMPLEX, 4, (0, 255, 255), 6)
        return False
    return True

def draw_reps_circle(img, count):
    """The circle with counter"""
    text = f"Reps: {int(count)}"
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.8
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    center_x = 30 + text_width // 2
    center_y = img.shape[0] - 80
    radius = int(1.2 * max(text_width, text_height) // 2)

    cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), -1)
    cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), 3)

    text_x = center_x - text_width // 2
    text_y = center_y + text_height // 2 - 4
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

def draw_progress_bar(img, percentage):
    """Progress which fills when the user pull up dembell"""
    bar_x = img.shape[1] - 60
    bar_top = 200
    bar_bottom = 600
    bar_fill = int(np.interp(percentage, [0, 100], [bar_bottom, bar_top]))

    cv2.rectangle(img, (bar_x, bar_top), (bar_x + 30, bar_bottom), (0, 0, 255), 3)
    cv2.rectangle(img, (bar_x, bar_fill), (bar_x + 30, bar_bottom), (0, 255, 0), -1)
    cv2.putText(img, f'{int(percentage)}%', (bar_x - 10, bar_top - 20),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

def draw_timer(img, start_time):
    """Timer for exercise"""
    elapsed = int(time.time() - start_time)
    mins, secs = divmod(elapsed, 60)
    timer_text = f"{mins:02}:{secs:02}"
    cv2.putText(img, f"Time: {timer_text}", (700, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

def calculate_reps(lm_list, percentage, direction, count):
    """Calculate reps"""
    try:
        x12, y12 = lm_list[12][1:]
        x14, y14 = lm_list[14][1:]
        x16, y16 = lm_list[16][1:]
    except IndexError:
        return direction, count

    if all(map(lambda v: v is not None and v != 0, [x12, y12, x14, y14, x16, y16])):
        if percentage >= 95 and direction == 1:
            direction = 0
            count += 0.5
        elif percentage <= 5 and direction == 0:
            direction = 1
            count += 0.5
    return direction, count

def main():
    cap = cv2.VideoCapture(0)
    detector = pm.PoseDetector()

    direction = 1
    count = 0.0
    started = False
    countdown_done = False
    countdown_start_time = 0
    exercise_start_time = 0
    pTime = 0

    while True:
        # Read an image
        success, img = cap.read()
        if not success:
            break
        
        # Resize camera
        img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Find users body
        img = detector.findPose(img)
        
        # Find landmarks
        lm_list = detector.getPoseLandmarks(img)
        
        # Delay for showing frames
        key = cv2.waitKey(1)

        # Logic for launching exercise and timer
        if not started:
            show_start_prompt(img)
            cv2.imshow("Image", img)
            if key == 32:
                started = True
                countdown_start_time = time.time()
            elif key == ord('q'):
                break
            continue

        if started and not countdown_done:
            if not show_countdown(img, countdown_start_time):
                cv2.imshow("Image", img)
                if key == ord('q'):
                    break
                continue
            else:
                countdown_done = True
                exercise_start_time = time.time()

        # Calculating percentage + reps which were done my user
        percentage, _ = detector.findPercentage(img, 50, 130, 100, 0, 12, 14, 16)
        if lm_list and len(lm_list) > 16:
            direction, count = calculate_reps(lm_list, percentage, direction, count)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (25, 75),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

        # Visualisation of counter circle, progress bar, timer
        draw_reps_circle(img, count)
        draw_progress_bar(img, percentage)
        draw_timer(img, exercise_start_time)

        cv2.imshow("Image", img)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
