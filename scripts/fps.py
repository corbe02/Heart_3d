import cv2

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Non riesco ad aprire il video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps  # durata in secondi
    
    cap.release()
    return fps, frame_count, duration

if __name__ == "__main__":
    video_path = "/home/corbe/heart_ws/src/heart_pkg/videos/HB_v1_src.mp4"
    fps, total_frames, duration = get_video_info(video_path)
    print(f"FPS del video: {fps}")
    print(f"Numero totale di frame: {total_frames}")
    print(f"Durata video: {duration:.2f} secondi")
