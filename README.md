
# [Re] IDOL: Inertial Deep Orientation-Estimation and Localization

This repository is the official implementation of [[Re] IDOL: Inertial Deep Orientation-Estimation and Localization](https://arxiv.org/abs/2030.12345). The code is being prepared for submission to: (https://paperswithcode.com/rc2021)ML Reproducibility Challenge 2021 Fall Edition, and a course project in CISC 867 Deep Learning, Queen's University.

1. Quaternion Multiplication -> See [here](https://www.sciencedirect.com/topics/computer-science/quaternion-multiplication)
2. Yury Petrov's Ellipsoid Fitting (Python Version) -> See [here](https://github.com/marksemple/pyEllipsoid_Fit)
3. Extended Kalman Filters -> See [here](https://towardsdatascience.com/extended-kalman-filter-43e52b16757d)
4. 3Blue1Brown Quaternion Explanations -> See [here](https://www.youtube.com/watch?v=d4EgbgTm0Bg)
5. IMUs and what they do -> See [here](https://www.arrow.com/en/research-and-events/articles/imu-principles-and-applications)
6. What is a random walk? -> See [here]()


>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

*Optional Dependencies*
piptools - used to modify requirements.txt without dependency hassles.

```setup
pip install pip-tools
```

Steps: 

1. (Optional) To generate new requirements (after adding new requirement to requirements.in): 
*Note*: This requires that you install pip-tools, if you haven't installed pip-tools then 
please do `pip install pip-tools` to use the command below.

```sh
pip-compile
```

2. To setup virtual environment: 

```sh
python -m venv .venv
```

3. To activate virtual environment (unix): 
   
```sh
source .venv/bin/activate
```

4. To install requirements:

```sh
pip install -r requirements.txt
```

5. To setup the datasets: 
    a. Create a folder called datasets.
    b. Create another folder within datasets called csvs. You should have datasets/csvs as part of your folder structure
    c. Download and extract the datasets from [here](https://zenodo.org/record/4484093). Extract each building into 
    datasets.


## Training

To train the model(s) in the paper, run this command:

### OrientNet 

```python
python main.py train_orient --option=<option number 1-3>
```

### PosNet 

```python
python main.py train_pos --option=<option number 1-3>
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To test the model(s) in the paper, run this command: 

### OrientNet 

```python
python main.py test_orient --option=<option number 1-3>
```

### PosNet 

```python
python main.py test_pos --option=<option number 1-3>
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name       | Top 1 Accuracy | Top 5 Accuracy |
| ---------------- | -------------- | -------------- |
| ReOrientNet      | x              | x              |
| PosNet           | y              | y              | 

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

Uses Apache License, see LICENSE for more details. 

>📋  Pick a licence and describe how to contribute to your code repository. 


