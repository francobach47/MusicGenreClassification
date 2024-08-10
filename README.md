# 🎵 Music Genre Classification

Music Genre Classification is a machine learning project focused on identifying the genre of a given music track based on its audio features. The model leverages deep learning techniques, particularly a Convolutional Neural Network (CNN), to process audio data and predict the genre from a predefined set of categories.

This project was developed as part of the MLOps Zoomcamp course, where the primary goal was to apply MLOps practices across the machine learning lifecycle, including data preprocessing, model training, experiment tracking, model deployment, and monitoring.

This project is still a work in progress. Further improvements and enhancements are planned for the future, whether it's for code and MLOps tools.

<h1></h1>

 ## 🌱 Getting Started


1. Download or clone the repository.
 ```
 git clone https://github.com/francobach47/MusicGenreClassification.git
cd MusicGenreClassification
 ```

2. Create and initialize poetry environment

```
make install
poetry shell
```

3. Install the pre-commit hooks for code formating and linting with `black` and `pylint`.

```
pre-commit install
```

<h1></h1>


## 📂 Pre Processing

Start by downloading the dataset from the following link: [GTZAN Dataset for Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?datasetId=568973&sortBy=voteCount).

Once downloaded, run the music_genre_classifier/preprocessing.py script as follows:

```
python3 music_genre_classifier/preprocessing.py --dataset_path dataset
```

Here, `dataset_path` refers to the base path where you downloaded the GTZAN dataset.

Once executed, you will obtain a folder and file structure similar to the following:

```
dataset/
   ├── genres_original/
   ├── images_original/
   ├── features_3_sec.csv
   ├── features_30_sec.csv
   └── genres_mel_npz/
           ├── blues
           |    ├── blues.00000-0.npz
           |    ├── blues.00000-1.npz
           |    ├── blues.00000-2.npz
           |    ├── blues.00000-3.npz
           |    ├── blues.00000-4.npz
           |    └── ...
           ├── classical
           └── ...
```

<h1></h1>

## 🏋🏻 Training

Once the data is preprocessed, you can train the model using the following command:

```
python3 music_genre_classifier/train.py
```

<h1></h1>

## 🚀 Inference

To perform inference locally, execute the following command:

```
python3 music_genre_classifier/inference.py --model_path <path_to_model_weight> --audio_path <path_to_audio>
```