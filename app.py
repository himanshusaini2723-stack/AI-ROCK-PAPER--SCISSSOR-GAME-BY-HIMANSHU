from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
import random

app = Flask(__name__)

# ---------------- MEDIAPIPE SETUP ----------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- GAME VARIABLES ----------------

choices = ["Rock", "Paper", "Scissors"]

player_score = 0
computer_score = 0

# ---------------- WINNER LOGIC ----------------

def decide_winner(player, computer):

    global player_score
    global computer_score

    if player == computer:

        return "Draw"

    elif (
        (player == "Rock" and computer == "Scissors")
        or
        (player == "Paper" and computer == "Rock")
        or
        (player == "Scissors" and computer == "Paper")
    ):

        player_score += 1

        return "You Win!"

    else:

        computer_score += 1

        return "Computer Wins!"

# ---------------- HOME PAGE ----------------

@app.route('/')
def landing():

    return render_template("home.html")

# ---------------- GAME PAGE ----------------

@app.route('/game')
def game():

    return render_template("index.html")

# ---------------- RESET ----------------

@app.route('/reset', methods=['POST'])
def reset():

    global player_score
    global computer_score

    player_score = 0
    computer_score = 0

    return jsonify({
        "success": True
    })

# ---------------- PREDICT ----------------

@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.get_json()['image']

        encoded_data = data.split(',')[1]

        image_data = base64.b64decode(encoded_data)

        np_arr = np.frombuffer(
            image_data,
            np.uint8
        )

        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        detected_choice = "None"

        # ---------------- HAND DETECTION ----------------

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                landmarks = hand_landmarks.landmark

                fingers = []

                # THUMB

                if abs(
                    landmarks[4].x -
                    landmarks[3].x
                ) > 0.04:

                    fingers.append(1)

                else:

                    fingers.append(0)

                # OTHER FINGERS

                tip_ids = [8, 12, 16, 20]

                for tip in tip_ids:

                    if landmarks[tip].y < landmarks[tip - 2].y:

                        fingers.append(1)

                    else:

                        fingers.append(0)

                # ---------------- GESTURE LOGIC ----------------

                # ROCK

                if fingers == [0,0,0,0,0]:

                    detected_choice = "Rock"

                # SCISSORS

                elif (
                    fingers == [0,1,1,0,0]
                    or
                    fingers == [1,1,1,0,0]
                ):

                    detected_choice = "Scissors"

                # PAPER

                elif (
                    fingers == [1,1,1,1,1]
                    or
                    fingers == [0,1,1,1,1]
                ):

                    detected_choice = "Paper"

        # ---------------- GAME LOGIC ----------------

        computer_choice = "Waiting..."
        result = "Show Hand"

        if detected_choice != "None":

            computer_choice = random.choice(choices)

            result = decide_winner(
                detected_choice,
                computer_choice
            )

        return jsonify({

            "prediction": detected_choice,

            "computer": computer_choice,

            "result": result,

            "player_score": player_score,

            "computer_score": computer_score

        })

    except Exception as e:

        print(e)

        return jsonify({

            "prediction": "Error",

            "computer": "Error",

            "result": "Error",

            "player_score": player_score,

            "computer_score": computer_score

        })

# ---------------- RUN ----------------

if __name__ == '__main__':

    app.run(debug=True)