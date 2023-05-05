Implementing a Basic Fully Connected Neural Network in Flask Based on the Iris Dataset
==============================
This is the second project for the DSS course (CIS431) at the JUST university (2<sup>nd</sup> semester, 2023).

It contains the guidelines and code necessary to help the students to carry out the project.

The following datasets have been used (for demonstration purposes):

* [Iris](https://www.kaggle.com/datasets/uciml/iris), the famous iris dataset; it will be used for building a FCNN
  classifier, which will be deployed on Flask.
* [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10), a dataset which contains 10 types/classes of
  animals in images; it will be used as an object recognition task for building a CNN architecture.

> **Note**: The provided code based on these datasets are a mere reference for you; you are required to build and deploy
> a model based on another dataset (discussed in the **_objectives_** section).

Getting Started
------------
Clone the project from GitHub

`$ git clone https://github.com/tariqshaban/fcnn-iris-flask.git`

Make sure that you have [Python](https://www.python.org/downloads) installed.

No further configuration is required.

> **Warning**: You must install the packages in the exact version found in the `requirements.txt` file; you can use this
> command: `pip install -r requirements.txt`

## Project Structure

    ├── README.md                                                 <- README file for developers using this project
    ├── requirements.txt                                          <- Contains the dependecies required to successfully run the project
    │
    ├── dataset
    │   │── animals-10                                            <- Store the dataset images of the animals-10
    │   │── synthetic-asl-numbers                                 <- Store the dataset images of the synthetic-asl-numbers (your target dataset)
    │   └── iris.csv                                              <- Store the iris dataset
    │
    ├── out                                                       <- Store model weights/configuration/architecture/...
    │
    ├── static
    │   │── images                                                <- Store images for display in the frontend
    │   │── scripts                                               <- Store Javascript for execution in the frontend
    │   └── styles                                                <- Store CSS for styling the frontend
    │
    ├── templates                                                 <- Store HTML files which are linked to Flask
    │
    ├── backend.py                                                <- Handles Flask enpoints
    ├── cnn_animals.py                                            <- Handles the training of the animals-10 dataset using CNN
    └── fcnn_iris.py                                              <- Handles the training of the iris dataset using FCNN

Objectives
------------
Now that you have been provided with an example on how to build, export, and deploy a model, your task will be composed
of:

* Inspect the [synthetic-asl-numbers](https://www.kaggle.com/datasets/lexset/synthetic-asl-numbers) dataset, note that
  this repository contains a modified version of the dataset; by reducing the resolution and sampling the images; so
  that the computation requirements is reduced. Please work **_ONLY_** on the dataset in the repository and do not
  download the dataset from Kaggle.
* Building a Convolutional Neural Network (CNN or ConvNet)
    * Use the dataset in `dataset/synthetic-asl-numbers/` as the target dataset.
    * You are strongly recommended to use the given `cnn_animals.py` script and modify it to be able to train on the
      dataset you have been tasked with, you are expected to change the following:
        * The dataset source
        * The dataset labels
        * The model's architecture
        * Any fine-tuning (as necessary)
* Export the trained model
* Deploy the exported model on Flask
    * Frontend
        * Create a home page, which contains a form that allows the user to upload an image, and a button to submit.
        * Create a prediction page, which display the prediction to the user.
    * Backend
        * Add a default endpoint `/` which redirects the user to the home page.
        * Add a prediction endpoint `/predict` which does the following:
            * Successfully decode the image from the user's submitted form (in a way that the Keras model would
              interpret).
            * Pass the decoded image into the model for prediction.
            * Fetch the prediction of the model, and properly map it for the user to interpret.
            * Redirect the user to the prediction page while passing the calculated prediction.
        * You may copy the `backend.py` script and modify it accordingly based on the given requirements.

Submission
------------
You will need to submit the following files:

* The code used for building and exporting the CNN model.
* Basic performance indicators for your CNN model, this includes:
    * Loss history plot.
    * Accuracy history plot.
    * Confusion plot.
    * _*These plots are automatically exported to you as an image file._
* The code used for deploying the CNN model (Flask part, this include):
    * The front end used as an interface for the user, including, but not limited to:
        * Template folder (HTML files).
        * Static folder (JS, CSS, and raw images used as an asset).
    * The backend used for processing the user's requests.
    * _*These plots are automatically exported to you as an image file._

You should upload all these files to the e-elearning; however, the Flask code (which is responsible for deploying the
model) should be in a project on [Replit](https://replit.com). Once the project is available for public access, upload a
shareable link of that project to the e-learning.

Notes
------------
> **Note**: When you run the model for prediction, **DON'T** return the raw prediction as is, it must be read easily
> by the user (i.e. Map the numerical value that is received from the model into a pre-defined textual value).

> **Note**: Your target dataset is the synthetic-asl-numbers, **DO NOT BUILD A MODEL WHICH IS BASED ON ANOTHER
DATASET**.

> **Note**: You are not allowed to add, modify, delete the target dataset in any way.

> **Note**: If your computational resources are limited (very long training times), it is recommended to use Colab, and
> download the exported model.

> **Note**: You are expected to deliver a properly formatted website, this includes structuring and styling of the
> webpages.

> **Warning**: Any form of cheating will result in zero for both parties.

> **Warning**: If you were not able to defend your code or understand the general intuition of the project, you will get
> zero.

> **Warning**: Late submissions are not allowed, no exceptions!

Useful Links
------------
[CNN intuition (IBM topic)](https://www.ibm.com/topics/convolutional-neural-networks)

[Minimal CNN Code using Keras (Official TensorFlow Docs)](https://www.tensorflow.org/tutorials/images/cnn)

[Official Flask Docs](https://flask.palletsprojects.com/en/2.3.x)

[Creating a Flask project on Replit](https://replit.com/talk/learn/Flask-Tutorial-Part-1-the-basics/26272)

--------