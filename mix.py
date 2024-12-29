import pygame
import random
import sys
import cv2
import mediapipe as mp
import math
import time  

# Initialize pygame
pygame.init()

# Screen dimensions
screen_width = 1400
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
#pygame.mixer.music.load('music.mp3')

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)


clock = pygame.time.Clock()


block_size = 20


font = pygame.font.SysFont(None, 35)

# MediaPipe hands setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def message(msg, color, y_displace=0):
    text_surf = font.render(msg, True, color)
    text_rect = text_surf.get_rect(center=(screen_width / 2, screen_height / 2 + y_displace))
    screen.blit(text_surf, text_rect)

def display_score(score):
    text = font.render("Score: " + str(score), True, white)
    screen.blit(text, [0, 0])

def display_time(time_left):
    text = font.render("Time Left: " + str(time_left), True, white)
    screen.blit(text, [screen_width - 180, 0])

def vector_2d_angle(v1, v2):
    try:
        angle = math.degrees(math.acos((v1[0] * v2[0] + v1[1] * v2[1]) /
                                       (math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2))))
    except:
        angle = 180
    return angle

def hand_angle(hand):
    angle_list = []
    # Thumb angle
    angle_list.append(vector_2d_angle(
        (hand[0][0] - hand[2][0], hand[0][1] - hand[2][1]),
        (hand[3][0] - hand[4][0], hand[3][1] - hand[4][1])
    ))
    # Index finger angle
    angle_list.append(vector_2d_angle(
        (hand[0][0] - hand[6][0], hand[0][1] - hand[6][1]),
        (hand[7][0] - hand[8][0], hand[7][1] - hand[8][1])
    ))
    # Middle finger angle
    angle_list.append(vector_2d_angle(
        (hand[0][0] - hand[10][0], hand[0][1] - hand[10][1]),
        (hand[11][0] - hand[12][0], hand[11][1] - hand[12][1])
    ))
    # Ring finger angle
    angle_list.append(vector_2d_angle(
        (hand[0][0] - hand[14][0], hand[0][1] - hand[14][1]),
        (hand[15][0] - hand[16][0], hand[15][1] - hand[16][1])
    ))
    # Pinky finger angle
    angle_list.append(vector_2d_angle(
        (hand[0][0] - hand[18][0], hand[0][1] - hand[18][1]),
        (hand[19][0] - hand[20][0], hand[19][1] - hand[20][1])
    ))
    return angle_list

def hand_pos(finger_angle):
    f1 = finger_angle[0]   # thumb
    f2 = finger_angle[1]   # forefinger
    f3 = finger_angle[2]   # middlefinger
    f4 = finger_angle[3]   # ringfinger
    f5 = finger_angle[4]   # picky
    # < 50 stretch，>= 50 curl

    if f1 >= 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return "Exit"
    elif f1 >= 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return "One"
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return "Two"
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 >= 50:
        return "Three"
    elif f1 >= 50 and f2 >= 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return "OK"
    else:
        return "Unknown"

def hand_pos_rock(finger_angle):
    f1, f2, f3, f4, f5 = finger_angle

    if f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return "good"
    elif f1 >= 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return "rock"
    elif f1 < 50 and f2 < 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return "paper"
    elif f1 >= 50 and f2 < 50 and f3 < 50 and f4 >= 50 and f5 >= 50:
        return "scissors"
    elif f1 < 50 and f2 >= 50 and f3 >= 50 and f4 >= 50 and f5 < 50:
        return "six"
    elif f1 >= 50 and f2 < 50 and f3 >= 50 and f4 >= 50 and f5 >= 50:
        return "1"
    elif f1 >= 50 and f2 >= 50 and f3 < 50 and f4 < 50 and f5 < 50:
        return "ok"
    else:
        return ""
    
def recognize_direction(hand_landmarks, img):
    img_height, img_width, _ = img.shape
    index_tip = hand_landmarks.landmark[8]
    wrist = hand_landmarks.landmark[0]

    index_tip_x = int(index_tip.x * img_width)
    index_tip_y = int(index_tip.y * img_height)
    wrist_x = int(wrist.x * img_width)
    wrist_y = int(wrist.y * img_height)

    if abs(index_tip_y - wrist_y) > abs(index_tip_x - wrist_x):
        if index_tip_y < wrist_y:
            return "Up"
        else:
            return "Down"
    else:
        if index_tip_x < wrist_x:
            return "Left"
        else:
            return "Right"

def game_intro(cap, hands):
    #pygame.mixer.music.play(-1)
    name = ""
    difficulty = None
    while True:
        screen.fill(black)
        message("Gesture to Start Game", white, 0)
        message("1 (Snake), 2 (Rock Paper Scissors)", white, 100)
        pygame.display.update()

        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_points = [(landmark.x * img.shape[1], landmark.y * img.shape[0]) for landmark in hand_landmarks.landmark]

                if finger_points:
                    finger_angles = hand_angle(finger_points)
                    gesture = hand_pos(finger_angles)

                    if gesture == "Exit":
                        pygame.quit()
                        sys.exit()

                    elif gesture == "One":
                        return "snake"
                    elif gesture == "Two":
                        return "rock"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

def game_intro_snake(cap, hands):
    name = ""
    difficulty = None
    while True:
        screen.fill(black)
        
        if difficulty is None:
            message("Choose Difficulty: 1 (Easy), 2 (Medium), 3 (Hard)", white, 0)
        if difficulty is not None:
            message("OK to start the game", white, 100)
        pygame.display.update()

        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_points = [(landmark.x * img.shape[1], landmark.y * img.shape[0]) for landmark in hand_landmarks.landmark]

                if finger_points:
                    finger_angles = hand_angle(finger_points)
                    gesture = hand_pos(finger_angles)
                    direction = recognize_direction(hand_landmarks, img)

                    if gesture == "Exit":
                        pygame.quit()
                        sys.exit()
                    elif gesture == "OK" and difficulty is not None:
                        return name, difficulty

                    elif difficulty is None:
                        if gesture == "One":
                            difficulty = 1
                        elif gesture == "Two":
                            difficulty = 2
                        elif gesture == "Three":
                            difficulty = 3

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and name and difficulty is not None:
                    return name, difficulty
                elif event.key == pygame.K_BACKSPACE:
                    if difficulty is None:
                        name = name[:-1]
                    else:
                        difficulty = None
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:
                    if difficulty is None:
                        difficulty = int(pygame.key.name(event.key))
                elif event.unicode.isalpha():
                    name += event.unicode

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

def game_loop_snake(difficulty, cap, hands):
    game_exit = False
    game_over = False
    gesture = "None"


    # Snake initial position
    lead_x = screen_width / 2
    lead_y = screen_height / 2

    # Snake movement
    lead_x_change = block_size  # Initial direction to the right
    lead_y_change = 0
    current_direction = 'RIGHT'

    # Snake body
    snake_list = []
    snake_length = 3  # Initial snake length

    # Initial snake segments
    for i in range(snake_length):
        snake_list.append([lead_x - i * block_size, lead_y])

    # Food
    food_x = round(random.randrange(0, screen_width - block_size) / block_size) * block_size
    food_y = round(random.randrange(0, screen_height - block_size) / block_size) * block_size

    # Obstacles
    obstacles = []
    if difficulty > 1:
        for _ in range(5 * difficulty):  # More obstacles for higher difficulty
            obs_x = round(random.randrange(0, screen_width - block_size) / block_size) * block_size
            obs_y = round(random.randrange(0, screen_height - block_size) / block_size) * block_size
            obstacles.append((obs_x, obs_y))

    # Speed settings
    if difficulty == 1:
        speed = 3
    elif difficulty == 2:
        speed = 5
    else:
        speed = 8

    # Hard mode time limit
    start_ticks = pygame.time.get_ticks() if difficulty == 3 else None

    while not game_exit:
        while game_over:
            screen.fill(black)
            message("Game Over! OK to Play Again or Exit to Quit", red, -50)
            message("Your Score: " + str(snake_length - 3), white, 50)
            pygame.display.update()

            ret, img = cap.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    finger_points = [(landmark.x * img.shape[1], landmark.y * img.shape[0]) for landmark in hand_landmarks.landmark]

                    if finger_points:
                        finger_angles = hand_angle(finger_points)
                        gesture = hand_pos(finger_angles)

                        if gesture == "Exit":
                            pygame.quit()
                            sys.exit()
                        elif gesture == "OK":
                            return False  # Play again

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_c:
                        return False  # Play again

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_exit = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and current_direction != 'RIGHT':
                    lead_x_change = -block_size
                    lead_y_change = 0
                    current_direction = 'LEFT'
                elif event.key == pygame.K_RIGHT and current_direction != 'LEFT':
                    lead_x_change = block_size
                    lead_y_change = 0
                    current_direction = 'RIGHT'
                elif event.key == pygame.K_UP and current_direction != 'DOWN':
                    lead_x_change = 0
                    lead_y_change = -block_size
                    current_direction = 'UP'
                elif event.key == pygame.K_DOWN and current_direction != 'UP':
                    lead_x_change = 0
                    lead_y_change = block_size
                    current_direction = 'DOWN'

        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                direction = recognize_direction(hand_landmarks, img)

                if direction == "Left" and current_direction != 'RIGHT':
                    lead_x_change = -block_size
                    lead_y_change = 0
                    current_direction = 'LEFT'
                elif direction == "Right" and current_direction != 'LEFT':
                    lead_x_change = block_size
                    lead_y_change = 0
                    current_direction = 'RIGHT'
                elif direction == "Up" and current_direction != 'DOWN':
                    lead_x_change = 0
                    lead_y_change = -block_size
                    current_direction = 'UP'
                elif direction == "Down" and current_direction != 'UP':
                    lead_x_change = 0
                    lead_y_change = block_size
                    current_direction = 'DOWN'

        # Boundary conditions
        if lead_x >= screen_width or lead_x < 0 or lead_y >= screen_height or lead_y < 0:
            game_over = True

        lead_x += lead_x_change
        lead_y += lead_y_change

        screen.fill(black)

        # Draw food
        pygame.draw.rect(screen, green, [food_x, food_y, block_size, block_size])

        # Draw obstacles
        if difficulty > 1:
            for obs in obstacles:
                pygame.draw.rect(screen, red, [obs[0], obs[1], block_size, block_size])

        # Snake movement
        snake_head = []
        snake_head.append(lead_x)
        snake_head.append(lead_y)
        snake_list.append(snake_head)

        if len(snake_list) > snake_length:
            del snake_list[0]

        for segment in snake_list[:-1]:
            if segment == snake_head:
                game_over = True

        # Collision with obstacles
        if difficulty > 1:
            for obs in obstacles:
                if lead_x == obs[0] and lead_y == obs[1]:
                    game_over = True

        for segment in snake_list:
            pygame.draw.rect(screen, blue, [segment[0], segment[1], block_size, block_size])

        display_score(snake_length - 3)

        # Hard mode time limit
        if difficulty == 3:
            text = font.render("Target 3 points in 30 sec", True, white)
            screen.blit(text, [0, 50])
            seconds = (pygame.time.get_ticks() - start_ticks) / 1000  # calculate seconds
            time_left = 30 - int(seconds)
            display_time(time_left)
            if seconds > 30:  
                game_over = True
            elif snake_length - 3 >= 3:
                break  # Win condition

        pygame.display.update()

        # Food collision
        if lead_x == food_x and lead_y == food_y:
            food_x = round(random.randrange(0, screen_width - block_size) / block_size) * block_size
            food_y = round(random.randrange(0, screen_height - block_size) / block_size) * block_size
            snake_length += 1

        clock.tick(speed)

    if difficulty == 3 and snake_length - 3 >= 3:
        screen.fill(black)
        message("You Win! OK to Play Again or Exit to Quit", green, -50)
        message("Your Score: " + str(snake_length - 3), white, 50)
        pygame.display.update()
        while not game_exit:
            ret, img = cap.read()
            if not ret:
                continue

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    finger_points = [(landmark.x * img.shape[1], landmark.y * img.shape[0]) for landmark in hand_landmarks.landmark]

                    if finger_points:
                        finger_angles = hand_angle(finger_points)
                        gesture = hand_pos(finger_angles)

                        if gesture == "Exit":
                            pygame.quit()
                            sys.exit()
                        elif gesture == "OK":
                            return False  # Play again

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_c:
                        return False  # Play again
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    return True

def game_loop_rock(cap, hands):
    def determine_winner(left, right):
        rules = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
        if left == right:
            return "Draw"
        elif rules.get(left) == right:
            return "Left Hand Wins"
        else:
            return "Right Hand Wins"

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    lineType = cv2.LINE_AA

    if not cap.isOpened():
        print("Cannot open camera")
        return

    countdown = False
    start_time = 0
    result_time = 0
    left_hand_gesture, right_hand_gesture = "", ""
    result = ""

    while True:
        ret, img = cap.read()
        img = cv2.resize(img, (screen_width, screen_height))
        if not ret:
            print("Cannot receive frame")
            break
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img3 = cv2.flip(img2, 1) 
        results = hands.process(img3)

        hands_detected = {}
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_type in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                finger_points = []
                for i in hand_landmarks.landmark:
                    x = int(i.x * screen_width)
                    y = int(i.y * screen_height)
                    finger_points.append((x, y))
                if finger_points:
                    finger_angle = hand_angle(finger_points)
                    gesture = hand_pos_rock(finger_angle)
                    hands_detected[hand_type.classification[0].label] = gesture

        if "Left" in hands_detected:
            left_hand_gesture = hands_detected["Left"]
        if "Right" in hands_detected:
            right_hand_gesture = hands_detected["Right"]
        if "six" in [left_hand_gesture, right_hand_gesture]:
            print("Exit Pose detected. Exiting...")
            cv2.destroyWindow("view")
            break

        if not countdown and (left_hand_gesture or right_hand_gesture):
            countdown = True
            start_time = time.time()

        if countdown:
            elapsed_time = time.time() - start_time
            if elapsed_time <= 3:
                cv2.putText(
                    img, f"Countdown: {3 - int(elapsed_time)}", (30, 160), fontFace, 4, (0, 255, 0), 4, lineType
                )
            elif not result:
                result = determine_winner(left_hand_gesture, right_hand_gesture)
                result_time = time.time()

        if result:
            cv2.putText(img, result, (30, 160), fontFace, 4, (255, 255, 0), 4, lineType)
            if time.time() - result_time > 5:
                result = ""
                countdown = False
                left_hand_gesture, right_hand_gesture = "", ""

        cv2.imshow("view", img)
        if cv2.waitKey(5) == ord("q"):
            break


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while True:
            game = game_intro(cap, hands)
            if game == "snake":
                pygame.display.update()
                for i in range(3, 0, -1):
                    screen.fill(black)
                    message(f"Starting Snake in {i} seconds...", white, 150)
                    pygame.display.update()
                    time.sleep(1)

                name, difficulty = game_intro_snake(cap, hands)
                
                should_exit = game_loop_snake(difficulty, cap, hands)
                if should_exit:
                    continue

            game = game_intro(cap, hands)
            
            if game == "rock":
                game_loop_rock(cap, hands)
                continue
            
            game = game_intro(cap, hands)
            
            if game == "Exit":
                break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

