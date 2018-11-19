

# Google Image Web-scraping Tutorial

### What's in the repo?

* **download_images.py**
    * *Code to download images from a list of urls*

### Dependencies

* [matplotlib](https://matplotlib.org/)
* [numpy](www.numpy.org/)
* [requests](docs.python-requests.org/en/master/)

------

## Simple Web-scraping  Tutorial

This tutorial is a simpler version of [] and is divided into three parts:

1. Obtaining a text file of the image urls that will make up the training and test datasets
2. Using python to download and perfom simple checks on the images

### Step 1: Image Webscraping

For this challenge you're going to need to build your own library of training data. For image data one of the best places to get this kind of data is Google Images. So in this tutorial we'll use Google Images to web-scrape a database of images. The instructions for this step are a simplified version of the excellent blog post [here](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/). To follow this tutorial you'll need to use Google Chrome, but there are also many nice ways of scraping data from the web using the Python requests library and the BeautifulSoup library, e.g. [here](https://allofyourbases.com/2017/10/08/web-scraping-youtube-in-python/).

1. Open Chrome and navigate to google image search. Now enter your search, e.g. "Santa"
2. Open the Developer console: either use CTRL+SHIFT+I or go to 'View' --> 'Developer' --> 'Javascript Console'. 
3. The next step is to start scrolling! Keep scrolling until you have found all relevant images to your query.
4. Next is to grab all the urls of the images in your scroll. In the console enter the following commands:

```javascript
// pull down jquery into the JavaScript console
var script = document.createElement('script');
script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
document.getElementsByTagName('head')[0].appendChild(script);
```
```javascript
// grab the URLs
var urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });
```

```javascript
// write the URls to file (one per line)
var textToSave = urls.toArray().join('\n');
var hiddenElement = document.createElement('a');
hiddenElement.href = 'data:attachment/text,' + encodeURI(textToSave);
hiddenElement.target = '_blank';
hiddenElement.download = 'zebra_urls.txt';
hiddenElement.click();
```
The last step will download a text file named: 'zebra_urls.txt'.

### Step 2: Download all the images with Python

This repo includes a [script](https://github.com/as595/4IRSantaHack/blob/master/tutorials/webscraping/download_images.py) to download all the images listed in the url file. You can also always write your own!

The script provided here takes two arguments: (1) the url list file, and (2) the location of the directory where you want the images to be stored. You can run it like this:

```bash
> python download_images.py -u santa_urls.txt -o ./ZEBRA/
```

Inside the script, the first step is to grab a list of the urls from the input file:

```python
# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0
```

then to loop through each of the urls and download each one into the specified output folder:

```python
# loop the URLs and download the images
for url in rows:
	try:
		# try to download the image
		r = requests.get(url, timeout=60)
 
		# save the image to disk
		p = os.path.sep.join([args["output"], "{}.jpg".format(
			str(total).zfill(8))])
		f = open(p, "wb")
		f.write(r.content)
		f.close()
 
		# update the counter
		print("[INFO] downloaded: {}".format(p))
		total += 1
 
	# handle if any exceptions are thrown during the download process
	except:
		print("[INFO] error downloading {}...skipping".format(p))

```

Once you've got all of the images you probably want to check to see if any have been corrupted or aren't really images. So there's a check at the end that tries to open each image. If the check fails (i.e. the image can't be opened) then the image file is deleted:

```python
# open the images, if it returns an error delete the image.
images = glob.glob('{}/*.jpg'.format(args["output"]))

for image in images:
	delete = False
	try:
		im = plt.imread(image,format='jpeg')
		if im is None:
			delete = True
	except:
		print('Except')
		delete = True

	if delete:
		print('INFO deleting {}'.format(image))
		os.remove(image)		
```

For classification we're also going to need a set of images that don't contain our target class, i.e. images that are NOT of santa. A good online database for random images is the [Caltech-256 Dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech256/), it contains about 30,000 images grouped into categories. You can build a "not santa" dataset by randomly sampling images from there (make sure you avoid images from the santa category!). Remember to randomly sample approximately the same number of "not santa" images as "santa" images, otherwise you'll end up with a [class imbalance problem](https://towardsdatascience.com/dealing-with-imbalanced-classes-in-machine-learning-d43d6fa19d2). 

-----

This tutorial is based on a [JBCA hack challenge](https://github.com/hrampadarath/JBCA_Hack_Night_Dec/tree/master/google_images_webscraping)
