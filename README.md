Implementing a Basic Fully Connected Neural Network in Flask Based on the Iris Dataset
==============================
This is the second project for the DSS course (CIS431) at JUST University (2<sup>nd</sup> semester, 2023).

It contains the guidelines and code necessary to help the students to carry out the project.

The following datasets have been used (for demonstration purposes):

* [Iris](https://www.kaggle.com/datasets/uciml/iris), the famous iris dataset; will be used for building an FCNN
  classifier, which will be deployed on Flask.
* [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10), a dataset which contains 10 types/classes of
  animals in images; it will be used as an object recognition task for building a CNN architecture.

> **Note**: The provided code based on these datasets is a mere reference for you; you are required to build and deploy
> a model based on another dataset (discussed in the **_objectives_** section).

Getting Started
------------
Clone the project from GitHub

`$ git clone https://github.com/tariqshaban/fcnn-iris-flask.git`

Make sure that you have [Python](https://www.python.org/downloads) installed.

No further configuration is required.

> **Note**: If you do not know how to clone a repository, you can either
> directly [download](https://stackoverflow.com/a/6466993) the project or
> read [here](https://github.com/git-guides/git-clone).

> **Warning**: You must install the packages in the exact version found in the `requirements.txt` file; you can use this
> command: `pip install -r requirements.txt`.

## Project Structure

    ├── README.md                                                 <- README file for developers using this project
    ├── requirements.txt                                          <- Contains the dependencies required to successfully run the project
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
    ├── backend.py                                                <- Handles Flask endpoints
    ├── cnn_animals.py                                            <- Handles the training of the animals-10 dataset using CNN
    └── fcnn_iris.py                                              <- Handles the training of the iris dataset using FCNN

Objectives
------------
Now that you have been provided with an example of how to build, export, and deploy a model, your task will be composed
of:

* Inspect the [synthetic-asl-numbers](https://www.kaggle.com/datasets/lexset/synthetic-asl-numbers) dataset; note that
  this repository contains a _modified_ version of the dataset. Please work **_ONLY_** on the dataset in the repository
  and do not download the dataset from Kaggle.
* Building a Convolutional Neural Network (CNN or ConvNet)
    * Use the dataset in `dataset/synthetic-asl-numbers/` as the target dataset.
    * Use the `cnn_animals.py` script and modify:
        * The dataset source (make it based on `synthetic-asl-numbers`)
        * The dataset labels
        * The model's architecture
        * Any fine-tuning (as necessary)
* Export the trained model
* Deploy the exported model on Flask
    * Frontend
        * Create a home page, which contains a form having:
            * Upload button (for images only)
            * Submit button
        * Create a prediction page; which displays the prediction to the user.
    * Backend
        * Add a default endpoint: `/`, which redirects the user to the home page.
        * Add a prediction endpoint: `/predict`, which does the following:
            * Properly read the uploaded image.
            * Pass the image into the model for prediction.
            * Fetch the prediction of the model.
            * Redirect the user to the prediction page while passing the prediction.
        * You can copy the `backend.py` script and modify it accordingly based on the given requirements.
* Verify your work by uploading a test image.

Submission
------------
You will need to submit the following files:

* The code used for **building** and **exporting** the CNN model.
* Basic performance indicators for your CNN model, which include:
    * Loss history plot.
    * Accuracy history plot.
    * Confusion plot.
    * _*These plots are automatically exported to you as an image file in `cnn_animals.py`._
* The code used for **deploying** the CNN model (using Flask), which includes:
    * The front end used as an interface for the user, including, but not limited to:
        * Template folder (HTML files).
        * Static folder (JS, CSS, and raw images used as an asset).
    * The backend is used for processing the user's requests.

You should upload all these files to the e-learning; however, the Flask code should be on [Replit](https://replit.com).
Upload a shareable link of that project to the e-learning.

Rubric
------------
The following points will dictate your grading:

| Criterion                                                                     | Weight |
|-------------------------------------------------------------------------------|--------|
| CNN model is trained and exported                                             | 1 mark |
| CNN model returns acceptable loss/accuracy values                             | 1 mark |
| The end-user can upload an image                                              | 1 mark |
| The backend can interpret "decode" the uploaded image                         | 1 mark |
| The model returns a valid prediction                                          | 1 mark |
| The backend successfully redirects the prediction and displays it to the user | 1 mark |
| The displayed prediction is in a textual form rather than numerical           | 1 mark |
| The website's visuals are appealing                                           | 1 mark |
| The web application is hosted on Replit                                       | 1 mark |
| The student understands how to construct a neural network                     | 1 mark |

Notes
------------
> **Note**: Do not return the numerical value (1, 2, 3) of the prediction directly to the user; instead, return a
> textual value (one, two, three).

> **Note**: Your target dataset is the synthetic-asl-numbers, **DO NOT BUILD A MODEL WHICH IS BASED ON ANOTHER
> DATASET**.

> **Note**: You are not allowed to add, modify, or delete the target dataset in any way.

> **Note**: If your computational resources are limited (very long training times), use Colab instead of your local
> machine.

> **Note**: You are expected to deliver a properly formatted website; this includes structuring and styling of the
> webpages.

> **Warning**: Any form of cheating will result in zero for both parties.

> **Warning**: If you were not able to defend your code or understand the general intuition of the project, you will get
> zero.

> **Warning**: Late submissions are not allowed. No exceptions!

Useful Links
------------
[CNN intuition (IBM topic)](https://www.ibm.com/topics/convolutional-neural-networks)

[Minimal CNN Code using Keras (Official TensorFlow Docs)](https://www.tensorflow.org/tutorials/images/cnn)

[Official Flask Docs](https://flask.palletsprojects.com/en/2.3.x)

[Creating a Flask project on Replit](https://replit.com/talk/learn/Flask-Tutorial-Part-1-the-basics/26272)

[Rand's Video](https://www.youtube.com/watch?v=s1Us3BM6gRg)

[Rand's Replit Project](https://replit.com/@randHani98/lastlab-dss)

--------