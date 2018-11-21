# CNN Classification

### What's in the repo?

* **CNN_Classification_Example.ipynb**
    * *A jupyter notebook that implements simple machine learning to identify images of Santa*


### Dependencies

* [matplotlib](https://matplotlib.org/)
* [numpy](www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [sklearn](scikit-learn.org/)
* [sklearn-image](https://scikit-image.org/)
* [requests](docs.python-requests.org/en/master/)
* [keras]()
* [openCV]()

------

## Simple CNN Classification Tutorial

In this repo we've provided an example [classifier](https://github.com/as595/4IRSantaHack/blob/master/tutorials/cnn_examples/CNN_Classification_Example.ipynb) that implements the LeNet architecture. LeNet was one of the very first convolutional neural networks. It's defined here in the file **lenet.py** and we can import it as:

```python
from lenet import LeNet
```

To train the machine learning model for the CNN we first need to tell the code where to find the input training data.

```python
dataset1 = '../../santa/'
dataset2 = '../../notsanta/'
```

We're going to train in epochs with a given batch of data per epoch.

```python
EPOCHS = 25
INIT_LR = 1e-3
BS = 32
```

We need to read all of the data into data arrays - and to make a corresponding array of labels for those data.

```python
data = []
labels = []
```

We want to make sure that the data are in a completely random order, so we'll shuffle all the data as we read it in.

```python
imagePaths = sorted(list(paths.list_images(dataset1))+list(paths.list_images(dataset2)))
random.seed(42)
random.shuffle(imagePaths)
```

Training is computationally expensive, so we'll resize all our data and make the images a bit smaller. They all also need to be the same size.

```python
for imagePath in imagePaths:

    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "santa" else 0
    labels.append(label)
```

Re-scale the image array values to lie within the interval [0,1]:

```python
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
```

We can use the scikit-learn function train_test_split to divide up our dataset into training and validation sets. The training_size is automatically *1 - test_size*.

```python
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
```

```python
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)
```

We don't have a huge amount of data, so we'll augment the training dataset by rotating, shifting and generally jiggling the data that we have.

```python
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")
```

Now let's build the machine learning model from LeNet:

```python
model = LeNet.build(width=28, height=28, chan=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
```

...and train it:

```python
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)
```

Training machine learning models is expensive computationally, so we don't want to have to keep repeating this step. Instead we can save our trained model and reload it whenever we need to use it.

```python
model.save(modelname)
```

We can visualise the success of the training by looking at the variation in accuracy and loss as a function of epoch.

```python
pl.style.use("ggplot")
pl.figure()
N = EPOCHS
pl.plot(np.arange(0, N), H.history["loss"], label="train_loss")
pl.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
pl.plot(np.arange(0, N), H.history["acc"], label="train_acc")
pl.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
pl.title("Training Loss and Accuracy on Santa/Not Santa")
pl.xlabel("Epoch #")
pl.ylabel("Loss/Accuracy")
pl.legend(loc="lower left")
pl.savefig("output.png")
```

Now let's see how we did... (well, actually our accuracy already tells us how well we did, but it's nice to see the network in practice).

We feed our network a test image,

```python
testimage="../../santa/00000171.jpg"
```

We need to apply the same pre-processing to this test image as we did with our training data:

```python
# pre-process the image for classification
image = cv2.imread(testimage)
orig = image.copy()
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
```

We can now use the machine learning model to predict the class of this image. The output is a pair of probabilities, which represent the likelihood of the image being *Santa* or *Not Santa*. Normally the class is decided based on whichever is larger.

```python
(notSanta, santa) = model.predict(image)[0]
```

We can then implement the classification.

```python
label = "Santa" if santa > notSanta else "Not Santa"
proba = santa if santa > notSanta else notSanta
label = "{}: {:.2f}%".format(label, proba * 100)
```

```python
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
```

```python
cv2.imshow("Output", output)
cv2.waitKey(0)
```
