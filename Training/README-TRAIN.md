

# OMR Application Training for classification
### Retraining 
In order to retrain the classifier and export the classification model:
1- Run main.py using the path of the folder that contain the images to train with.
Note: Images must be in BMP or JPG format, and they should be segmented already. i.e: don't train using full image.
example: `python main.py ./training_sets/set1`
A .csv file will be generated that is called `featuresAndLabels.csv` that will contain the data to be used with the main of the OMR application. `featuresAndLabels.csv` should be copied and pasted to be in the same directory as `main.py` of the project (the main of the OMR application)
Note that the current classifier uses a function that extracts the density of regions in the picture by dividing it into segments.

### Dataset used
The currently used dataset is our own, using guidoeditor.grame.fr/ to generate notes, and segmenting them ourselves. We have 16 picture of different symbols and numbers generated in different positions to better reflect our segmenting artifacts.
### Time taken
About 1~2 minutes due to using a small dataset. (As they're printed)
### Hardware used for training
GTX 770, I5-3450, 8GBs of RAM.

