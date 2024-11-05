import cv2
import mediapipe as  mp

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    
    while cap.isOpened():
        
        success, frame = cap.read()
        
        if not success:
            print("Ignorando o Frame vaziu da camera")
            continue
        
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
                
                face = face_landmarks
                for id_coord, coord_xyz in enumerate(face.landmark):
                    print(id_coord,coord_xyz)
                
        except:
            print("Deu erro")
        
        finally:
            print("encerrando o processo")
        
        cv2.imshow("Camera",frame)

        if cv2.waitKey(10) & 0xFF == ord("c"):
            break

cap.release()
cv2.destroyAllWindows()