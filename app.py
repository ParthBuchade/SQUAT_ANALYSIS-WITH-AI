import os
import gradio as gr
import cv2
from utils import get_mediapipe_pose
from process_frames import ProcessFrame
from thresholds import get_thresholds_beginner

sample_video = os.path.join(os.path.dirname(__file__), "sample-squats.mp4")

# Initialize pose model
POSE = get_mediapipe_pose()


def process_video(video_path):
    output_video_file = f"output_recorded.mp4"

    # Set thresholds (default to Beginner thresholds)
    thresholds = get_thresholds_beginner()

    upload_process_frame = ProcessFrame(thresholds=thresholds)

    # Read video and get frame properties
    vf = cv2.VideoCapture(video_path)
    fps = int(vf.get(cv2.CAP_PROP_FPS))
    width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

    count = 0
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break

        # Process the frame and convert BGR to RGB for the model
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_frame, _ = upload_process_frame.process(frame, POSE)

        # Write the output frame to the video
        video_output.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

        # Output a frame every 12 frames (just for demo)
        if not count % 12:
            yield out_frame, None

        count += 1

    vf.release()
    video_output.release()

    # Return the processed video file at the end
    yield None, output_video_file


# Input and output components
input_video = gr.Video(label="Input Video")
output_frames_up = gr.Image(label="Output Frames")
output_video_file_up = gr.Video(label="Output video")

# Custom Submit button
submit_button = gr.Button("Analyze")

# Create Gradio Interface
with gr.Blocks() as app:
    with gr.Row():
        input_video.render()

    with gr.Row():
        submit_button.render()

    with gr.Row():
        output_frames_up.render()
        output_video_file_up.render()

    # Bind the submit button to the process_video function
    submit_button.click(
        fn=process_video,
        inputs=[input_video],  # No choice selection now, only video input
        outputs=[output_frames_up, output_video_file_up]
    )

# Launch the app
app.queue().launch(share=True)
