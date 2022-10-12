import os
import subprocess
import tempfile
from typing import Tuple

import numpy as np
import torch
import librosa
import matplotlib
import matplotlib.animation as animation
from matplotlib.figure import Figure
import matplotlib.image as mpimg
import matplotlib.patches as patches
import soundfile as sf

from audio import extract_audio_features, write_video_wpts_wsound


class TalkingFaceTorchScript:
    def __init__(
        self,
        model_path: str,
        head_image: str,
        target_sample_rate: int = 44100,
        mouth_offset: Tuple[int, int] = None,
        mouth_stretch: int = 0,
        mouth_angle: float = 0.0,
        output_dir: str = 'generated-videos'
    ):
        self.model = torch.jit.load(model_path)
        self.target_sample_rate = target_sample_rate
        self.head_image = mpimg.imread(head_image)
        self.mouth_offset = mouth_offset if mouth_offset is not None else (0, 0)
        self.mouth_stretch = mouth_stretch
        self.mouth_angle = mouth_angle
        # TODO: get head image from UI instead
        print(f"Loaded Head Image {head_image}")

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    @torch.no_grad()
    def _predict(self, gradio_audio):
        source_sample_rate, source_audio_data = gradio_audio
        print(
            f"Model received audio: {source_audio_data}\nIt has shape {source_audio_data.shape} and type {source_audio_data.dtype}"
        )

        # Convert audio data to float (normalizes)
        audio_data = librosa.util.buf_to_float(
            source_audio_data, n_bytes=4, dtype=np.float32
        )

        print(
            f"After normalizing, it's {audio_data}\nIt has shape {audio_data.shape} and type {audio_data.dtype}"
        )
        input_audio = extract_audio_features(
            audio_data,
            source_sample_rate=source_sample_rate,
            target_sample_rate=self.target_sample_rate,
        )
        # TODO: convert input to tensor, return a more useful object
        print(f"Feeding model (possibly resampled audio) of shape {input_audio.shape}")
        predictions = self.model.predict(torch.tensor(input_audio))[0]
        predictions = predictions.reshape((-1, 68, 2)).numpy()

        return predictions

    def predict(self, gradio_audio):
        source_sample_rate, source_audio_data = gradio_audio
        predictions = self._predict(gradio_audio)

        print("Saving Video")
        # NOTE: Skipping Face Normalization (which is normally done during training) because I'm lazy. It seems to be OK without it
        # TODO: saving the video is a very slow operation. Check out `predict_animation`, which is closer to realtime
        write_video_wpts_wsound(
            predictions,
            source_audio_data,
            source_sample_rate,
            "./results",
            "PD_pts",
            [0, 1],
            [0, 1],
        )
        # TODO: save somewhere else
        return "./results/PD_pts_ws.mp4"

    def predict_stream(self, gradio_audio, state: Figure = None):
        source_sample_rate, source_audio_data = gradio_audio
        predictions = self._predict(gradio_audio)

        audio_output = (
            source_sample_rate,
            source_audio_data,
        )

        """
        from gr.Plot documentation:
            "As output: expects either a {matplotlib.figure.Figure}, a {plotly.graph_objects._figure.Figure}, or a {dict} corresponding to a bokeh plot (json_item format)"
        """

        """
        The plotting code may look scary and if it does, I recommend checking out [this writeup](http://web.archive.org/web/20100830233506/http://matplotlib.sourceforge.net/leftwich_tut.txt)
        and checking out the matplotlib Object Oriented API
        Summary: If a figure doesn't already exist, we create one and give it a new axes `ax`. Think of this axes as a layer that we can draw on. We then draw on that layer using `ax.scatter`.
        The next time this method gets called, we simply remove the old scatter plot data (via `ax.collections[0].remove()` and add our new data
        """
        if state is None:
            state = Figure(figsize=(3.00, 3.00))
            # The 111 specifies 1 row, 1 column on subplot #1
            ax = state.add_subplot(111)
            ax.set_title("Predicted Face Landmarks")
            ax.set_xlabel("time")
            ax.set_ylabel("volts")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.invert_yaxis()  # otherwise, the face will be upside-down
        else:
            ax = state.axes[0]
            # the next line is a reference to the scatterplot data further below
            ax.collections[0].remove()

        # These properties must be set everytime, OR the code needs refactored to store a reference to the scatter, which could then be `remove()`

        # TODO: Animate the result. For now, simply get the last frame of the batch
        x = predictions[-1, :, 0]
        y = predictions[-1, :, 1]

        ax.scatter(x, y, c="#1f77b4", marker=".")

        plot_output = state
        return audio_output, plot_output, state

    # This function is generally faster than `predict`. During local testing, it processed 15 seconds of audio in 12 seconds, whereas `predict` took 21 seconds
    def predict_animation(self, gradio_audio):
        source_sample_rate, source_audio_data = gradio_audio

        # Slightly abusing tempfile's ability to generate random file names. We need two video files since ffmpeg won't edit in-place
        # Also sharing that randomness for audio and video to make debugging easier in the future!
        source_video_id = next(tempfile._get_candidate_names())
        temp_audio_path = f"audio-{source_video_id}.wav"
        temp_video_path = f"animation-{source_video_id}.mp4"
        output_video_path = f"{self.output_dir}/output-{next(tempfile._get_candidate_names())}.mp4"

        # Scale predictions to head_image size
        # TODO: parameterize instead of hardcoding
        predictions = 600 * self._predict(gradio_audio)

        # Scale the mouth vertically and keep it mostly closed
        mouth_y_delta = predictions[:, 55:60, 1] - predictions[:, 48:49, 1]
        squared_delta_norm = mouth_y_delta**2.2 / np.max(mouth_y_delta)
        predictions[:, 55:60, 1] = np.maximum(
            predictions[:, 55:60, 1]
            - mouth_y_delta
            + squared_delta_norm
            - np.average(squared_delta_norm),
            predictions[:, 48:49, 1] + 5,  # +5 offsets some rotation weirdness
        )
        # Scale mouth width to match image (hacky)
        # TODO: replace with linspace or similar
        predictions[:, 48, 0] -= self.mouth_stretch
        predictions[:, 59, 0] -= 2 * self.mouth_stretch / 3
        predictions[:, 58, 0] -= self.mouth_stretch / 3
        predictions[:, 54, 0] += self.mouth_stretch
        predictions[:, 55, 0] += 2 * self.mouth_stretch / 3
        predictions[:, 56, 0] += self.mouth_stretch / 3
        # Add mouth offset
        predictions[:, :, 0] += self.mouth_offset[0]
        predictions[:, :, 1] += self.mouth_offset[1]

        figure = Figure(figsize=(6.00, 6.00))

        # The 111 specifies 1 row, 1 column on subplot #1
        ax = figure.add_subplot(111)
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 600)
        ax.set_axis_off()
        ax.invert_yaxis()  # otherwise, the face will be upside-down
        ax.imshow(self.head_image)

        mouth_polygon = patches.Polygon(
            [
                predictions[0, 54],
                *predictions[0, 55:60],
                predictions[0, 48],
            ],
            facecolor="k",
            edgecolor=(143 / 255.0, 67 / 255.0, 0.0),
            linewidth=3,
        )

        mouth_pivot_point = 48
        mouth_polygon.set_transform(
            ax.transData
            + matplotlib.transforms.Affine2D().rotate_deg_around(
                predictions[0, mouth_pivot_point, 0],
                predictions[0, mouth_pivot_point, 1],
                self.mouth_angle,
            )
        )
        ax.add_patch(mouth_polygon)

        def init():
            return (mouth_polygon,)

        # https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
        def animate(i):
            mouth_polygon.set_xy(
                [
                    *predictions[i, 54:60],
                    predictions[i, 48],
                ]
            )
            return (mouth_polygon,)

        anim = animation.FuncAnimation(
            figure,
            animate,
            init_func=init,
            frames=predictions.shape[0],
            interval=40,
            blit=True,
        )

        FFwriter = animation.FFMpegWriter(fps=25, extra_args=["-vcodec", "libx264"])
        anim.save(temp_video_path, writer=FFwriter)

        # add audio back in
        sf.write(temp_audio_path, source_audio_data, source_sample_rate, "PCM_24")
        cmd = f"ffmpeg -y -i {temp_video_path} -i {temp_audio_path} -map 0:v -map 1:a -c:v copy -shortest {output_video_path}"
        subprocess.call(cmd, shell=True)
        os.remove(temp_video_path)
        os.remove(temp_audio_path)
        return output_video_path
