import cv2

class VideoReader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def read_frame(self):
        return self.cap.read()

    def video_parameters(self):
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, width, height, total_frames

    def release(self):
        self.cap.release()

class VideoWriter:
    def __init__(self, output_path, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
