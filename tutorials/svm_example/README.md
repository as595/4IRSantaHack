# SVM Classification

### What's in the repo?

* **Classify_Santa_Images.ipynb**
    * *A jupyter notebook that implements simple machine learning to identify images of Santa*


### Dependencies

* [matplotlib](https://matplotlib.org/)
* [numpy](www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [sklearn](scikit-learn.org/)
* [sklearn-image](https://scikit-image.org/)
* [requests](docs.python-requests.org/en/master/)

------

## Simple SVM Classification Tutorial

There are many different approaches to image classification. One heavily used method is Convolutional Neural Networks (CNNs) and there's a good example of how to implement a CNN using the [keras library](https://keras.io/) in [this blog](https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/) and [this blog](https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/).

In this repo we've provided an example [classifier](https://github.com/darabigdata/IDWBotswana/blob/master/CHALLENGE-2/Classifying_Zebra_Images.ipynb) that uses a combination of [Gabor filters](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabor.html) and [Support Vector Machines (SVMs)](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72). 

The purpose of the Gabor filter is to extract machine learning **features** on multiple scales from an image. By doing this it compresses the information in each image down to a small set of numbers. First we need to define the type of Gabor filters we want to use:

```python
# first we will define a function that will use Gabor filters to reduce the images to a constant set of features
# define Gabor features
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        #feats[k, 0] = filtered.mean()
        #feats[k, 1] = filtered.var()
        feats[k, 0] = kurtosis(np.reshape(filtered,-1))
        feats[k, 1] = skew(np.reshape(filtered,-1))
    return feats
```

and then we need to define the number of **scales** we want to filter on:

```python
# prepare Gabor filter bank kernels
kernels = []
for sigma in (1,4):
    theta = np.pi
    for frequency in (0.05, 0.25):
        print('theta = {}, sigma = {} frequency = {}'.format(theta, sigma, frequency) )
        kernel = np.real(gabor_kernel(frequency,theta=theta,sigma_x=sigma, sigma_y=sigma))
        kernels.append(kernel)
                         
np.shape(kernels)
```

Once we've done that we can apply the filters to our "zebra" images. We are using **2 filters** on **4 scales**, so we will get an output of **8 features** for each image.

```python
zebra_feats = np.zeros((len(zebra_images),9))
for i, image in enumerate(zebra_images):
    im = plt.imread(image,format='jpeg')
    if len(im.shape) > 2:
        imean = im.mean(axis=2)
    else:
        imean = im
    imfeats = compute_feats(imean,kernels).reshape(-1)
    zebra_feats[i,:-1] = imfeats 
    zebra_feats[i,-1] = 1
```

We now need to do the same for our "not zebra" images:

```python
nozebra_feats = np.zeros((len(nozebra_images),9))
for i, image in enumerate(nozebra_images):
    im = plt.imread(image,format='jpeg')
    imfeats = compute_feats(im.mean(axis=2),kernels).reshape(-1)
    nozebra_feats[i,:-1] = imfeats 
    nozebra_feats[i,-1] = 0
```

We'll combine all of these features into a single dataset:

```python
#combine the datasets
ds = np.concatenate((nozebra_feats,zebra_feats), axis=0)
features = ds[:,:-1]
```

and then we need to normalise the feature values to lie between 0 and 1. We can do this using a library routine from the scikit-learn library:

```python
features = MaxAbsScaler().fit_transform(features)
```

We need to tell our machine learning classifier which column in the dataset corresponds to the target class, i.e. "zebra" or "not zebra":

```python
target = ds[:,-1]
```

and then we can split the full dataset into:

* a training data set (to train our classifier), and 
* a test dataset (to test our classifer).

If we wanted, we could also add in a *validation* dataset to test for over-fitting... we won't do that in this simple example, but you might want to think about it for your own classifier.

```python
x_train, x_test, y_train, y_test = train_test_split(features,target)

print('Training data and target sizes: \n{}, {}'.format(x_train.shape,y_train.shape))
print('Test data and target sizes: \n{}, {}'.format(x_test.shape,y_test.shape))
```

Now we have to choose a classifer. For this example we're going to use a [Support Vector Machine (SVM) from the scikit-learn library](http://scikit-learn.org/stable/modules/svm.html). There are various options for how to implement support vector machines in scikit-learn; here we're using the Support Vector Classifier (SVC) and you can find a description of the parameters [here](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC). 

To use it all we have to do is (1) call the algorithm, and (2) tell it to fit a machine learning model using our training data:

```python
# Create a classifier: a support vector machine classifier
classifier = svm.SVC(C=1, kernel='rbf', gamma=1)

# fit to the training data
classifier.fit(x_train,y_train)
```

Once we've trained the machine learning model we can test how well it works using our test data:

```python
# now predict the value of the digit on the test data
y_pred = classifier.predict(x_test)
```

To assess how well it performed there are a range of methods. A simple way to view the results is the [confusion matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/):

```python
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))
```

There are also the standard error metrics:

```python
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, y_pred)))
```

This nice [slide by Nikos Nikolaou](https://github.com/as595/4IR-ClassificationWorkshop/tree/master/NIKOS_NIKOLAOU) summarises some of the standard metrics for assessing how well machine learning algorithms perform.

<p align="center"><img width=80% src="https://github.com/darabigdata/IDWBotswana/blob/master/media/errors.png"></p>

Or you can read about it online, for example [here](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c) and [here](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall).


<p align="center"><img width=15% src="https://github.com/as595/4IRSantaHack/blob/master/media/animated-santa-claus-image-0034.gif"></p>

-----

This tutorial is based on a [JBCA hack challenge](https://github.com/hrampadarath/JBCA_Hack_Night_Dec/tree/master/google_images_webscraping)
