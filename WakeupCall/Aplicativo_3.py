import cv2
import mediapipe as  mp
import numpy as np
import time
import pygame

pygame.mixer.init()

p_olho_d = [160,144,158,153,33,133]
p_olho_e = [385,380,387,373,362,263]
p_olhos = p_olho_e + p_olho_d
sound_state = False  
def play_alarm():
    pygame.mixer.music.load("WakeupCall/audio/alarm1.mp3", "alarme")
    pygame.mixer.music.play()
def stop_alarm():
    pygame.mixer.music.stop()

def calc_EAR(face,p_olho_d,p_olho_e):
    try:
        print("tried")
        face = np.array([[coord.x,coord.y] for coord in face])
        face_e =face[p_olho_e, :]
        face_d = face[p_olho_d, :]
        print("found")
        ear_e = (np.linalg.norm(face_e[0] - face_e[1]) + np.linalg.norm(face_e[2] - face_e[3])) / (2*np.linalg.norm(face_e[4] - face_e[5]))
        ear_d = (np.linalg.norm(face_d[0] - face_d[1]) + np.linalg.norm(face_d[2] - face_d[3])) / (2*np.linalg.norm(face_d[4] - face_d[5]))
        print("calculated")
    except:
        print("error")
        ear_e = 0.0
        ear_d = 0.0

    media_ear = (ear_e + ear_d)/2
    return media_ear

p_boca = [82, 87, 13, 14, 312, 317, 78, 308]
def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_boca = face[p_boca, :]

        mar = (np.linalg.norm(face_boca[0] - face_boca[1]) + np.linalg.norm(face_boca[2] - face_boca[3]) + np.linalg.norm(face_boca[4] - face_boca[5])) / (2 * (np.linalg.norm(face_boca[6] - face_boca[7])))
    except:
        mar = 0.0
    return mar
limiar = 0.29
m_limiar = 0.18
sleeping = 0


cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    
    while cap.isOpened():
        
        success, frame = cap.read()
        
        if not success:
            print("Ignorando o Frame vaziu da camera")
            continue
        comprimento, largura, _ = frame.shape
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        saida_facemesh = facemesh.process(frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        
        try:
            for face_landmarks in saida_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, 
                                          face_landmarks, 
                                          mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255,102,102),thickness=1, circle_radius=1),
                                          connection_drawing_spec = mp_drawing.DrawingSpec(color=(102,204,0),thickness=1, circle_radius=1)
                                          )
                
                face = face_landmarks.landmark
                for id_coord, coord_xyz in enumerate(face):
                    if id_coord in p_olhos:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x,coord_xyz.y,largura,comprimento)
                        cv2.circle(frame,coord_cv,2,(255,0,0),-1),
                    if id_coord in p_boca:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x,coord_xyz.y,largura,comprimento)
                        cv2.circle(frame,coord_cv,2,(255,150,0),-1),   
                
                ear = calc_EAR(face,p_olho_d,p_olho_e)
                mar = calculo_mar(face,p_boca)

                cv2.rectangle(frame, (0,1),(290,140),(58,58,55),-1)
                
                cv2.putText(frame, 
                            f"EAR: {round(ear, 2)}", 
                            (1, 24),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9, (255, 255, 255),
                            2
                            )
                cv2.putText(frame, 
                            f"MAR: {round(mar, 2)}", 
                            (1, 50),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9, (255, 255, 255),
                            2
                            )
                
                if ear < limiar and mar < m_limiar:
                    t_inicial = time.time() if dormindo == 0 else t_inicial
                    dormindo = 1
                    
                else:
                    dormindo = 0

                t_final = time.time()
          
                tempo = (t_final-t_inicial) if dormindo == 1 else 0.0
                cv2.putText(frame, f"Tempo: {round(tempo, 3)}", (1, 80),
                                        cv2.FONT_HERSHEY_DUPLEX,
                                        0.9, (255, 255, 255), 2)
                if tempo>=1.5:

                    cv2.rectangle(frame, (30, 400), (610, 452), (109, 233, 219), -1)
                    cv2.putText(frame, f"Muito tempo com olhos fechados!", (80, 435),
                                    cv2.FONT_HERSHEY_DUPLEX,
                                    0.85, (58,58,55), 1)
                    if not sound_state:  # Only play the alarm if it's not already playing
                            play_alarm()  # Start playing the alarm
                            sound_state = True  # Set sound_state to True to indicate alarm is playing

                else:
                    if sound_state:
                        stop_alarm()
                        sound_state = False
                
                if not sound_state:
                    stop_alarm()

                    
        except:
            print("Deu erro")
        
        finally:
            
            print("encerrando o processo")

        cv2.imshow("Camera",frame)

        if cv2.waitKey(10) & 0xFF == ord("c"):
            break

cap.release()
cv2.destroyAllWindows()