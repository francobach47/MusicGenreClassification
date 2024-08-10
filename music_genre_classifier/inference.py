import data_managment.dataset as dataset
from model import CNNNetwork
import preprocessing
import numpy as np
import torch
import librosa
import argparse


def predict_song(
    audio_path: str, time_segments: int, model, class_mapping: dict
) -> dict:
    """Predicts the main genre of the input song.

    Parameters
    ----------
    audio_path : str
        Path of the song.
    time_segments : int > 0
        Time segment in which the audio will be divided.
    model : torch.nn.Module
        Model used to predict the genre.
    class_mapping : dict
        Class mapping. Genres and their indexes.
    plot : bool, optional
        Plots the probability distribution if it is true. The default is True.

    Returns
    -------
    prediction : str
        Main genre predicted of the input song.
    """
    spectrogram_tensors = inference_song_preprocessing(audio_path, time_segments)

    predictions = []
    for tensor in spectrogram_tensors:
        prediction = predict(model, tensor, class_mapping)
        predictions.append(prediction)
    prediction = np.mean(predictions, axis=0)
    print(prediction)
    predicted_index = np.argmax(prediction)
    genre_predicted = class_mapping[predicted_index]

    return genre_predicted


def predict(
    model: CNNNetwork, input_to_predict: torch.tensor, class_mapping: dict
) -> np.array:
    """Evaluate data input through the model and returns the probability distribution for all the genres.

    Parameters
    ----------
    model : torch.nn.Module
        Model used to predict the genre.
    input_to_predict : torch.tensor
        Data to be predicted and evaluated through the model.
    class_mapping : dict
        Class mapping. Genres and their indexes.

    Returns
    -------
    predictions : np.array
        Data input's prediction for all the genres.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(input_to_predict)
        predictions = np.array(predictions.tolist())

    return predictions


def inference_song_preprocessing(audio_path: str, time_segments: int) -> list:
    """Pre processes a song for the inference.

    Parameters
    ----------
    audio_path : str
        Path of the song.
    time_segments : int > 0
        Time segment in which the audio will be divided.

    Returns
    -------
    tensors : list
        List with the tensors relative to all the spectrograms.
    """
    signal, sr = librosa.load(audio_path)
    samples_per_segment = time_segments * sr
    n = int(len(signal) / samples_per_segment)  # Number of segments

    spectrograms = []
    for N in range(n):
        start_sample = samples_per_segment * N
        finish_sample = start_sample + samples_per_segment
        # Extract Mel spectrogram
        spectrograms.append(
            preprocessing.extract_mel_spectrogram(signal[start_sample:finish_sample])
        )
    spectrograms = preprocessing.normalization(spectrograms)  # Normalization

    tensors = []
    for spectrogram in spectrograms:
        tensor = torch.from_numpy(spectrogram).unsqueeze(0)  # Pytorch tensor (1, H, W)
        tensor = tensor.unsqueeze(0)  # Pytorch tensor (1, 1, H, W)
        tensors.append(tensor)

    return tensors


def main():
    parser = argparse.ArgumentParser(description="Predict the genre of a song.")
    parser.add_argument(
        "--model_path", type=str, help="Path to the model weights (.pth file)."
    )
    parser.add_argument("--audio_path", type=str, help="Path to the audio file")
    parser.add_argument(
        "--time_segments",
        type=int,
        default=3,
        help="Number of time segments to divide the audio into.",
    )
    args = parser.parse_args()

    # Load the model
    cnn = CNNNetwork()
    state_dict = torch.load(args.model_path)
    cnn.load_state_dict(state_dict)

    # Load the class mapping (make sure to adjust this if needed)
    class_mapping = dataset.genres_map

    # Make prediction
    genre_predicted = predict_song(
        args.audio_path, args.time_segments, cnn, class_mapping
    )
    print(f"The predicted genre is: {genre_predicted}")


if __name__ == "__main__":
    main()
