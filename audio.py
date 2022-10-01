"""
These functions copied almost verbatim from the training repo
"""
import os
import subprocess

import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import soundfile as sf


Mouth = [
    [48, 49],
    [49, 50],
    [50, 51],
    [51, 52],
    [52, 53],
    [53, 54],
    [54, 55],
    [55, 56],
    [56, 57],
    [57, 58],
    [58, 59],
    [59, 48],
    [60, 61],
    [61, 62],
    [62, 63],
    [63, 64],
    [64, 65],
    [65, 66],
    [66, 67],
    [67, 60],
]

Nose = [
    [27, 28],
    [28, 29],
    [29, 30],
    [30, 31],
    [30, 35],
    [31, 32],
    [32, 33],
    [33, 34],
    [34, 35],
    [27, 31],
    [27, 35],
]

leftBrow = [[17, 18], [18, 19], [19, 20], [20, 21]]
rightBrow = [[22, 23], [23, 24], [24, 25], [25, 26]]

leftEye = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [36, 41]]
rightEye = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47]]

other = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [12, 13],
    [13, 14],
    [14, 15],
    [15, 16],
]

faceLmarkLookup = Mouth + Nose + leftBrow + rightBrow + leftEye + rightEye + other


def extract_audio_features(
    audio_data: np.ndarray,
    hsize: float = 0.04,
    wsize: float = 0.04,
    source_sample_rate: int = 44100,
    target_sample_rate: int = 44100,
):
    # Used for padding zeros to first and second temporal differences
    zeroVecD = np.zeros((1, 64), dtype="f16")
    zeroVecDD = np.zeros((2, 64), dtype="f16")

    # Load speech and extract features
    if source_sample_rate != target_sample_rate:
        sound = librosa.resample(
            audio_data, orig_sr=source_sample_rate, target_sr=target_sample_rate
        )
    melFrames = np.transpose(melSpectra(sound, target_sample_rate, wsize, hsize))
    melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)
    melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)

    features = np.concatenate((melDelta, melDDelta), axis=1)
    # features = addContext(features, ctxWin)  # TODO: revisit this!
    features = np.reshape(features, (1, features.shape[0], features.shape[1]))
    return features


def melSpectra(y, sr, wsize, hsize):
    cnst = 1 + (int(sr * wsize) / 2)  # 883.0
    y_stft_abs = (
        np.abs(
            librosa.stft(
                y,
                win_length=int(sr * wsize),  # 1764
                hop_length=int(sr * hsize),  # 1764
                n_fft=int(sr * wsize),
            )
        )
        / cnst
    )  # with division by cnst, np.max(y_stft_abs) == 0.1368

    melspec = np.log(
        1e-16
        + librosa.feature.melspectrogram(
            sr=sr,
            S=y_stft_abs
            ** 2,  # since all numbers are pretty small, this makes them even smaller
            n_mels=64,
        )
    )
    return melspec


def write_video_wpts_wsound(frames, sound, sample_rate, path, fname, xLim, yLim):
    os.makedirs(path, exist_ok=True)
    try:
        os.remove(os.path.join(path, fname + ".mp4"))
        os.remove(os.path.join(path, fname + ".wav"))
        os.remove(os.path.join(path, fname + "_ws.mp4"))
    except (IsADirectoryError, FileNotFoundError):
        # It's ok if the files didn't already exist
        pass

    if len(frames.shape) < 3:
        frames = np.reshape(frames, (frames.shape[0], frames.shape[1] / 2, 2))

    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=25, metadata=metadata)

    fig = plt.figure(figsize=(10, 10))
    (l,) = plt.plot([], [], "ko", ms=4)

    plt.xlim(xLim)
    plt.ylim(yLim)

    # librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, sample_rate)
    sf.write(os.path.join(path, fname + ".wav"), sound, sample_rate, "PCM_24")

    if frames.shape[1] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
        print(lookup)
    else:
        lookup = faceLmarkLookup

    lines = [plt.plot([], [], "k")[0] for _ in range(3 * len(lookup))]

    with writer.saving(fig, os.path.join(path, fname + ".mp4"), 150):
        plt.gca().invert_yaxis()
        for i in range(frames.shape[0]):
            l.set_data(frames[i, :, 0], frames[i, :, 1])
            cnt = 0
            for refpts in lookup:
                lines[cnt].set_data(
                    [frames[i, refpts[1], 0], frames[i, refpts[0], 0]],
                    [frames[i, refpts[1], 1], frames[i, refpts[0], 1]],
                )
                cnt += 1
            writer.grab_frame()

    cmd = (
        "ffmpeg -y -i "
        + os.path.join(path, fname)
        + ".mp4 -i "
        + os.path.join(path, fname)
        + ".wav -c:v copy -c:a aac -strict experimental "
        + os.path.join(path, fname)
        + "_ws.mp4"
    )
    subprocess.call(cmd, shell=True)
    print("Muxing Done")

    os.remove(os.path.join(path, fname + ".mp4"))
    os.remove(os.path.join(path, fname + ".wav"))
