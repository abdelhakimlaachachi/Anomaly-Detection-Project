import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog


DATA_PATH = os.path.join('dataset_pose')
ACTIONS = np.array(['normal', 'anomalie'])
SEQUENCE_LENGTH = 30 

TARGET_WIDTH = 640
TARGET_HEIGHT = 480

for action in ACTIONS:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

print(" Chargement du modèle YOLOv8-Pose...")
model = YOLO('yolov8n-pose.pt')

def extract_keypoints(results):
    """Extrait les coordonnées normalisées (X, Y) des 17 points du squelette."""
    if results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
        keypoints = results[0].keypoints.xyn[0].cpu().numpy() 
        return keypoints.flatten()
    else:
        return np.zeros(34)


def collect_from_webcam():
    cap = cv2.VideoCapture(0)
    print("\n Caméra prête. Appuyez sur 'n' pour Normal, 'a' pour Anomalie, 'q' pour quitter.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        cv2.putText(annotated_frame, "Appuyez sur 'n' (Normal) ou 'a' (Anomalie)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Collecte Webcam', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('n'), ord('a')]:
            action_name = 'normal' if key == ord('n') else 'anomalie'
            files = os.listdir(os.path.join(DATA_PATH, action_name))
            file_id = len(files)
            
            print(f" Enregistrement de '{action_name}' (Sequence {file_id})... Bougez !")
            sequence_data = []
            
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                
                frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                
                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()
                
                cv2.putText(annotated_frame, f"Enregistrement {action_name} : {frame_num+1}/{SEQUENCE_LENGTH}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Collecte Webcam', annotated_frame)
                cv2.waitKey(1)
                
                keypoints = extract_keypoints(results)
                sequence_data.append(keypoints)
                
            save_path = os.path.join(DATA_PATH, action_name, f"{file_id}.npy")
            np.save(save_path, np.array(sequence_data))
            print(f" Séquence sauvegardée : {save_path}")

    cap.release()
    cv2.destroyAllWindows()

def collect_from_video():
    root = tk.Tk()
    root.withdraw()

    video_paths = filedialog.askopenfilenames(
        title="Sélectionner une ou plusieurs vidéos",
        filetypes=[("Video files", "*.mp4 *.avi *.mov")]
    )

    if not video_paths:
        print(" Aucune vidéo sélectionnée.")
        return

    action_name = input("Ces vidéos contiennent quelle action ? (normal / anomalie) : ").strip().lower()

    if action_name not in ['normal', 'anomalie']:
        print(" Action invalide.")
        return

    print(f"\n {len(video_paths)} vidéo(s) sélectionnée(s).")

    for video_path in video_paths:
        print(f"\n Traitement : {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        sequence_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            cv2.putText(
                annotated_frame,
                f"Traitement: {action_name} | Appuyez sur 'q' pour passer",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            window_name = f"Traitement - {os.path.basename(video_path)}"
            cv2.imshow(window_name, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            keypoints = extract_keypoints(results)
            sequence_data.append(keypoints)

            if len(sequence_data) == SEQUENCE_LENGTH:
                files = os.listdir(os.path.join(DATA_PATH, action_name))
                file_id = len(files)

                save_path = os.path.join(DATA_PATH, action_name, f"{file_id}.npy")
                np.save(save_path, np.array(sequence_data))
                print(f" Séquence sauvegardée : {save_path}")

                sequence_data = []

        cap.release()
        cv2.destroyAllWindows()  

    print("\nToutes les vidéos ont été traitées !")


if __name__ == "__main__":
    print("OUTILS DE COLLECTE DE DONNÉES")
    print("1. Utiliser la Webcam (En direct)")
    print("2. Extraire depuis une Vidéo (.mp4, .avi)")
    print("0. Quitter")
    
    choix = input("\n Choisissez une option (0, 1 ou 2) : ").strip()
    
    if choix == '1':
        collect_from_webcam()
    elif choix == '2':
        collect_from_video()
    else:
        print("Au revoir !")