import cv2
import mediapipe as mp
import numpy as np

# some constants
NUM_PARTICLES = 100
WIDTH, HEIGHT = 1280, 720

# cv config
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# particles on a unit sphere
def fibonacci_sphere(n):
    golden = np.pi * (3 - np.sqrt(5))
    points = []
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = golden * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points, dtype=np.float32)

base_positions = fibonacci_sphere(NUM_PARTICLES)
jitter = np.random.randn(NUM_PARTICLES, 3) * 0.01
velocities = np.zeros_like(jitter)

# perspective projection
def project(pts3d):
    z = pts3d[:, 2] + 4.5
    x = (pts3d[:, 0] / z) * 500 + WIDTH // 2
    y = (pts3d[:, 1] / z) * 500 + HEIGHT // 2
    return np.stack([x, y], axis=-1).astype(np.int32)

# initial velocity and radius values
radius = 1.5
energy = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # detect hand distances
    distances = {"Left": None, "Right": None}
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label
            lm = hand_landmarks.landmark
            x1, y1 = int(lm[4].x * WIDTH), int(lm[4].y * HEIGHT)  # thumb
            x2, y2 = int(lm[8].x * WIDTH), int(lm[8].y * HEIGHT)  # index 
            dist = np.hypot(x2 - x1, y2 - y1)
            dist = np.clip(dist, 20, 200)  # clamp distance for stability
            distances[label] = dist

            # visual feedback, indication circles and line between fingers
            cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} Hand", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Dist: {int(dist)} px", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # right hand controls radius
    if distances["Right"] is not None:
        radius = np.interp(distances["Right"], [20, 200], [0.6, 2.5])
    else:
        radius = 1.0  # default value

    # left hand controls velocity
    if distances["Left"] is not None:
        energy = np.interp(distances["Left"], [20, 200], [0.0, 1.0])
    else:
        energy = 0.0  # default value

    # display radius and energy on screen
    cv2.putText(frame, f"Radius: {radius:.2f}", (10, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Energy: {energy:.2f}", (10, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # update particle velocities
    velocities += (np.random.randn(*velocities.shape) * 0.01 - velocities * 0.05)
    jitter += velocities * energy
    jitter *= 0.9  # damping

    # normalize to stay on sphere surface
    noisy_positions = base_positions + jitter
    normed = noisy_positions / np.linalg.norm(noisy_positions, axis=1, keepdims=True)
    final_positions = normed * radius

    # project and draw over
    pts2d = project(final_positions)
    for pt in pts2d:
        cv2.circle(frame, tuple(pt), 2, (255, 255, 255), -1)

    # display interface
    cv2.imshow("Interactive Particle Sphere", frame)
    if cv2.waitKey(1) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
