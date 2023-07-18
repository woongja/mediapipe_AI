import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['1', '2', '3','4','5']
seq_length = 30 #윈도우의 사이즈
secs_for_action = 10 #데이터 녹화 시간


# MediaPipe hands model 미디어 파이프 initialize 시키는 것 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True) #데이터셋 저장할 폴더 생성

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1) #웹캠의 영상을 읽고 플립 한번 시켜주기 -> 거울처럼 변환

        #어떠한 액션을 표현하는지 3초 동안 기다리기
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()
        #30초동안 반복을 하는데 하나씩 프레임을 읽어서 미디어파이프에 넣어준다.
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            #손의 각도를 뽑아내는 코드
            if result.multi_hand_landmarks is not None:
                
                for res in result.multi_hand_landmarks:
                    #x,y,z, visibility가 있기 때문에 21,4로 설정
                    joint = np.zeros((21, 4))
                    #각 조인트마다 랜드마크를 저장한다.
                    for j, lm in enumerate(res.landmark):
                        #visibility는 손의 노드 랜드마크가 보이는지 안보이는지 파악
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints 손 관절 사이의 코드 구현
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    #백터를 계산해서 관절 사이를 계산한다. 
                    v = v2 - v1 # [20, 3]
                    # Normalize v 각 백터의 길이를 구한다. 각 백터의 길이로 길이를 나눠준다. -> 크기 일짜리 기본 백터가 나온다.
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    # dot product를 arcos을 사용해 각도를 구해준다.
                    # einsum은 내적, 외적, 행렬곱 등을 해준다.
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                    #앵글이 라디안 값으로 나오기 때문에 디그리 값으로 바꿔준다.
                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    #마지막으로 라벨을 넣어주는데 idx를 통해 0인지 1인지 2인지 넣어준다.
                    
                    angle_label = np.append(angle_label, idx)
                    
                    #x와 y와 y와 비저빌리티가 들어있는 행렬을 concatenate시키면 100개의 행렬로 펼쳐준다.
                    d = np.concatenate([joint.flatten(), angle_label])

                    #data라는 변수에 전부 추가해준다.
                    data.append(d)
                    #손의 랜드마크를 그리는 코드
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data) #데이터를 30초동안 모았으면 numpy배열 형태로 변환시켜준다.
        print(action, data.shape)
        #npy형태로 저장
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        # 시퀸스 데이터는 순서가 존재하는 데이터이다.
        # 입력 x1이 들어가면 출력 y1이 나오는 데이터
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        #seq라 붙은 이름으로 저장해줄것이다.
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break
