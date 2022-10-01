import numpy as np
import torch
import librosa

from audio import extract_audio_features, write_video_wpts_wsound


class TalkingFaceTorchScript:
    def __init__(self, model_path: str, target_sample_rate: int = 44100):
        self.model = torch.jit.load(model_path)
        self.target_sample_rate = target_sample_rate

    @torch.no_grad()
    def predict(self, gradio_audio):
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

        print("Saving Video")
        # NOTE: Skipping Face Normalization (which is normally done during training) because I'm lazy. It seems to be OK without it
        # TODO: saving the video is a very slow operation
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
