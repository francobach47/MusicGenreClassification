import data_managment.dataset as dataset
from models.cnn import CNNNetwork
import preprocessing
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def predict_song(audio_path: str, name_song: str, artist: str, time_segments: int, model, class_mapping: dict, plot: bool = True) -> dict:
    '''Predicts the main genre of the input song. 

    Parameters
    ----------
    audio_path : str
        Path of the song.
    name_song : str
        Song's name.
    artist : str
        Song's band/artist.
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
    '''
    spectrogram_tensors = inference_song_preprocessing(audio_path, 3)

    predictions = [] # Get all spectrograms predictions
    for tensor in spectrogram_tensors:
        prediction = predict(model, tensor, class_mapping) # Spectrogram's prediction
        predictions.append(prediction)
    prediction = np.mean(predictions, axis=0) # (1, 10)
    prediction = prediction.squeeze(0) # (10, )
    
    predicted_index = np.argmax(prediction)
    genre_predicted = class_mapping[predicted_index]
    print(f'Main Genre of the song {name_song} - {artist}: {genre_predicted}')

    if plot == True:
        plot_predictions(prediction, name_song, artist)

    return genre_predicted


def predict(model: CNNNetwork, input_to_predict: torch.tensor, class_mapping: dict) -> np.array:
    '''Evaluate data input through the model and returns the probability distribution for all the genres.
    
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
    '''
    model.eval()
    with torch.no_grad():
        predictions = model(input_to_predict)
        predictions = np.array(predictions.tolist())
        
    return predictions

def inference_song_preprocessing(audio_path: str, time_segments: int) -> list:
    ''' Pre processes a song for the inference.
    
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
    '''
    signal, sr = librosa.load(audio_path)
    samples_per_segment = time_segments * sr 
    n = int(len(signal) / samples_per_segment) # Number of segments
    
    spectrograms = []
    for N in range(n):
        start_sample = samples_per_segment * N
        finish_sample = start_sample + samples_per_segment   
        # Extract Mel spectrogram
        spectrograms.append(preprocessing.extract_mel_spectrogram(signal[start_sample:finish_sample])) 
    spectrograms = preprocessing.normalization(spectrograms) # Normalization
    
    tensors = []
    for spectrogram in spectrograms:       
        tensor = torch.from_numpy(spectrogram).unsqueeze(0) # Pytorch tensor (1, H, W)
        tensor = tensor.unsqueeze(0) # Pytorch tensor (1, 1, H, W)
        tensors.append(tensor)

    return tensors

def plot_predictions(predictions, name_song, artist):
    '''Plots the probability distribution of the song. 

    Parameters
    ----------
    predictions : np.array
        Final prediction of the song.
    name_song : str
        Song's name.
    artist : str
        Song's band/artist.

    Returns
    -------
    None.
    '''
    color_data = [1,2,3,4,5,6,7,8,9,10]
    my_cmap = cm.get_cmap('jet')
    my_norm = Normalize(vmin=0,vmax=10)
    fig,ax = plt.subplots(figsize=(6,6))
    class_mapping = ['Blues', 'Classical', 'Country', 'Disco', 'HipHop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    ax.bar(x=class_mapping, height=predictions, color=my_cmap(my_norm(color_data)))
    ax.set_ylim([0, 1.05])
    plt.xticks(rotation=45)
    plt.title(f'Probability distribution of: {name_song} - {artist}')
    plt.show()
    
# if __name__ == '__main__':

#     # Load the model
#     cnn = CNNNetwork()
#     state_dict = torch.load('results/music_genre_classifier.pth')
#     cnn.load_state_dict(state_dict)                        
    
#     class_mapping = dataset.genres_map

#     print("Test with 5 songs of different genres:")
#     print('--------------------------------------')
#     slayer_song = predict_song('songs/slayer.wav', 'Disciple', 'Slayer', 3, cnn, class_mapping, plot=False)    
#     ledzeppeling_song = predict_song('songs/led_zeppelin.wav', 'Black Dog', 'Led Zeppelin', 3, cnn, class_mapping, plot=False)
#     abba_song = predict_song('songs/abba.wav', 'Dancing Queen', 'ABBA', 3, cnn, class_mapping, plot=False)
#     davebrubeck_song = predict_song('songs/dave_brubeck.wav', 'Take Five', 'Dave Brubeck', 3, cnn, class_mapping, plot=False)
#     getoboys_song = predict_song('songs/geto_boys.wav', 'Mind Playing Tricks on Me', 'Geto Boys', 3, cnn, class_mapping, plot=False)
