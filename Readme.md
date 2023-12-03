Deep Learning Project

We will be comparing the performance of the following techniques:
1) CNN
2) RNN (LTSM)
3) K-nearest neighbors
4) Support Vector Machine
5) Naive Bayes Classifier 

With respect to Music Genre Classification.

The dataset has been downloaded from GTZAN website.

A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)

genres original: 10 genres with 100 audio files each length of 30 seconds each

images original: visual representation for each audio file, audio files converted into Mel spectrograms

two csv files: 
One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file.

The other file has the same structure, but the songs were split before into 3 seconds audio files


sudo docker build -t my_flask_app .
sudo docker run -p 5000:5000 my_flask_app