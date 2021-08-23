import mediapipe as mp
import cv2

class HandDetector():
    def __init__(self,  no_of_hand=2, detection_confidence=0.5, track_confidence=0.5):
        """[summary]

        Args:
            no_of_hand (int, optional): Number of hand for detection. Defaults to 2.
            detection_confidence (float, optional): detection confidence. Defaults to 0.5.
            track_confidence (float, optional): tracking confidence. Defaults to 0.5.
        """
        self.maxHands = no_of_hand
        self.detection_conf = detection_confidence
        self.track_conf = track_confidence

        # Detection = True, Maximum Hand =2, Detection Threshold = 0.5
        # Tracking Confidence = 0.5
        self.hand_model = mp.solutions.hands.Hands(True,2,0.5,0.5)

    def get_hands(self, image):
        """[summary]

        Args:
            image ([type]): Image object

        Returns:
            [type]: It will return landmark position
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.result = self.hand_model.process(rgb_image)
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(image, handLms,
                                               mp.solutions.hands.HAND_CONNECTIONS)
        return image

    def hand_position(self, img, hand_no=0, draw=False):
        """[summary]

        Args:
            img ([type]): Image 
            hand_no (int, optional): Hand Number . Defaults to 0.
            draw (bool, optional): Drawling circle on landmark keypoints. Defaults to False.

        Returns:
            [type]: It will return handmark position i.e. landmark 
        """
        landmark = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myHand.landmark):
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                landmark.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return landmark

