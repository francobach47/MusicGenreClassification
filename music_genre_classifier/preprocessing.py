import numpy as np
import librosa
import os
import argparse


def pre_process_dataset(
    dataset_path: str,
    n: int = 10,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 512,
):
    """Preprocesses dataset.

    Processes all the audios files and saves the normalized MEL spectrograms as .npz files in different folders per genre.

    Parameters
    ----------
    dataset_path : str
        Base dataset path.
    n : int > 0
        Number of segments in which the audio will be divided. 10 by default.
    sr : int > 0
        Sample rate. 22050 by default.
    n_fft : int > 0
        Lenght of the FFT window. 2048 by default.
    hop_length : int > 0
        Number of overlapping samples between successive frames. 512 by default.

    Returns
    -------
    None.
    """
    # Data check. Mono or Stereo / Sample Rate / Length / Possible errors
    nchannels, sample_rates, lengths = check_audio_features(dataset_path)

    if len(lengths) > 0:
        # By default: SR = 22050 and 30 s each audio
        samples_per_track = sr * 30  # 661500 samples per audio
    else:
        samples_per_track = lengths[0]  # Only 1 sample length in all the dataset
    samples_per_segment = int(samples_per_track / n)

    dataset_path = os.path.join(dataset_path, "genres_original")
    genres = os.listdir(dataset_path)
    spectrograms = []  # Get all Mel spectrograms to normalize
    metavector = []  # Get all the audio files name
    for genre in genres:
        print(f"Processing: {genre}")
        genre_path = os.path.join(dataset_path, genre)
        # Process each song
        list_songs = os.listdir(os.path.join(dataset_path, genre))
        for filename in list_songs:
            song_metavector, song_spectrograms = pre_process_song(
                genre_path, filename, samples_per_track, n, samples_per_segment
            )
            metavector.extend(song_metavector)
            spectrograms.extend(song_spectrograms)

    # Normalize and save the .npz files
    spectrograms = normalization(spectrograms)
    for i, genre in enumerate(genres):
        print(f"Saving .npz files: {genre}")

        # Generate folders to save the .npz files
        npz_path = os.path.join("dataset/genres_mel_npz", genre)
        if os.path.exists(npz_path):
            pass
        else:
            os.makedirs(npz_path)

        list_songs = os.listdir(os.path.join(dataset_path, genre))
        amount_files = len(list_songs)  # Amount of audio files per genre
        genre_songs = metavector[amount_files * i : (amount_files * i + amount_files)]
        for j, filename in enumerate(genre_songs):
            cant = i * amount_files + j
            spectrogram = spectrograms[cant]
            save_path = os.path.join(npz_path, metavector[cant])
            np.savez_compressed(
                save_path, spectrogram=spectrogram
            )  # Save Mel spectrogram as .npz file


def pre_process_song(
    genre_path: str,
    filename: str,
    samples_per_track: int,
    n: int,
    samples_per_segment: int,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> tuple:
    """Preprocesses a dataset'song.

    It splits the song into n segments and returns their names and spectrograms.

    Parameters
    ----------
    genre_path : str
        Path of the genre being processed.
    filename : str
        Name of the audio signal.
    n : int > 0
        Number of segments in which the audio will be divided. 10 by default.
    samples_per_segment : int > 0
        Number of samples per each sub segment.
    sr : int > 0
        Sample rate. 22050 by default.
    n_fft : int > 0
        Lenght of the FFT window. 2048 by default.
    hop_length : int > 0
        Number of overlapping samples between successive frames. 512 by default.

    Returns
    -------
    metavector : list
        List with the n audio segment names.
    spectrograms : list
        List with the n audio segment spectrograms.
    """

    try:
        metavector = []
        spectrograms = []

        # Load audio and check its length
        audio_path = os.path.join(genre_path, filename)
        [signal, sr] = librosa.load(path=audio_path, sr=sr)
        if len(signal) != samples_per_track:
            signal = set_audio_length(signal, samples_per_track)

        # Process all the n segments of audio file
        for N in range(n):
            metavector.append(filename[:-4] + "-" + str(N))
            start_sample = samples_per_segment * N
            finish_sample = start_sample + samples_per_segment

            # Extract Mel spectrogram
            spectrograms.append(
                extract_mel_spectrogram(signal[start_sample:finish_sample])
            )

    except Exception:
        print(f"Error with {filename}")

    return metavector, spectrograms


def extract_mel_spectrogram(
    signal, sr: int = 22050, n_fft: int = 1024, hop_length: int = 512
) -> np.ndarray:
    """Extract mel spectrogram from a audio signal.

    Parameters
    ----------
    signal : narray
        Audio signal to process.
    sr : int > 0
        Sample rate. 22050 by default.
    n_fft : int > 0
        Lenght of the FFT window. 2048 by default.
    hop_length : int > 0
        Number of overlapping samples between successive frames. 512 by default.

    Returns
    -------
    spectrogram : narray
        Signal's spectrogram.
    """
    spectrogram = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    return spectrogram


def normalization(tensor: list) -> list:
    """Normalizes a dataset.

    Normalize a dataset using mean and variance normalisation.

    Parameters
    ----------
    tensor : list
        Data to be normalized. Mel spectrograms.

    Returns
    -------
    tensor_norm : list
        Data normalized. Mel spectrograms normalized.
    """

    tensor = np.array(tensor)
    tensor_mean = np.mean(tensor, axis=0)
    tensor_std = np.std(tensor, axis=0)
    tensor_std[tensor_std == 0] = 1e-8
    tensor_norm = (tensor - tensor_mean) / tensor_std
    tensor_norm = tensor_norm / np.max(np.abs(tensor_norm))

    return tensor_norm


def check_audio_features(dataset_path: str, sr: int = 22050) -> tuple:
    """Check different audio features. Among them: Mono or Stereo, sample rate, sample length and possible errors.

    Parameters
    ----------
    dataset_path : str
        Base dataset path.
    sr : int > 0
        Sample rate. 22050 by default.

    Returns
    -------
    nchannels : ndarray
        Different numbers of audio's channels in the dataset.
    sample_rates : ndarray
        Different sample rates in the dataset.
    lengths : ndarray
        Different audio's lengths in the dataset.
    """

    sample_rates = []  # Get all sample rates
    nchannels = []  # Get all number of channels of each song
    lengths = []  # Get all song's lengths

    dataset_path = os.path.join(dataset_path, "genres_original")
    genres = os.listdir(dataset_path)
    for genre in genres:
        list_songs = os.listdir(os.path.join(dataset_path, genre))
        for filename in list_songs:
            audio_path = os.path.join(dataset_path, genre, filename)
            try:
                [signal, sr] = librosa.load(path=audio_path, sr=sr)  # Load audio file
                tensorshape = np.shape(signal)
                if len(tensorshape) == 1:
                    nchannels.append(1)
                else:
                    nchannels.append(tensorshape[1])
                lengths.append(tensorshape[0])
                sample_rates.append(sr)
            except Exception:
                print(f"Error with {filename}")

    nchannels = np.unique(nchannels)
    lengths = np.unique(lengths)
    sample_rates = np.unique(sample_rates)

    return nchannels, sample_rates, lengths


def set_audio_length(signal: np.ndarray, N: int) -> np.ndarray:
    """Set audio length.

    It applies a process to the signal.
    Zero padding if the signal's number of samples is less than N.
    It cuts the signal if N is greater than the signal's number of samples is less than N.

    Parameters
    ----------
    signal : ndarray
        Audio signal to process.
    N : int > 0
        Final audio's length.

    Returns
    -------
    signal : ndarray
        Edited audio signal.
    """

    if len(signal) > N:
        signal = signal[:N]
    if len(signal) < N:
        signal = np.hstack((signal, np.zeros(N - len(signal))))

    return signal


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess audio dataset for music genre classification."
    )
    parser.add_argument(
        "--dataset_path", type=str, default="dataset", help="Path to the dataset."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of segments to divide each audio into.",
    )
    parser.add_argument(
        "--sr", type=int, default=22050, help="Sample rate of the audio."
    )
    parser.add_argument(
        "--n_fft", type=int, default=1024, help="Length of the FFT window."
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=512,
        help="Number of overlapping samples between successive frames.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pre_process_dataset(args.dataset_path, args.n, args.sr, args.n_fft, args.hop_length)
