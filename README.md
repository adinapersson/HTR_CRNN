# Handwriting recognition of Swedish handwritten text
## Project in Embedded Systems (15 hp), Uppsala University
###### Co-author: Christian Böhme
###### Supervisor: Ping Wu

* The code is compatible with TensorFlow2
 
A Handwritten Text Recognition (HTR) system implemented in Python with TensorFlow and Keras. The system uses a CRNN model with CTC loss and is trained on the English IAM dataset and a Swedish selfmade dataset. The model takes segmented images as input and outputs the predicted text. For clearly handwritten text 80-90% of each test sentence is correctly recognised and approximately 78% of the words in the validation dataset are correctly recognized by the model. 

The ScrabbleGAN program from https://github.com/Nikolai10/scrabble-gan is used for creating our own Swedish dataset.  


## Run demo

Scan a paper with text and put into the *segmentation.m* script. Then create labels by running the *create_labels.py* script.

Some basic Git commands are:
```
git status
git add
git commit
```

blablabla
