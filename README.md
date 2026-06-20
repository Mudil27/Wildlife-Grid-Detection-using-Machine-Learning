# Wildlife Grid Detection using Machine Learning

**Course:** DS203 – Data Science
**Semester:** 2025 Semester 1
**Project:** Exercise 7

---

## Project Overview

This project focuses on detecting the **presence of wildlife (animals, birds, or insects)** in different regions of an image using **machine learning and handcrafted image features**.

Each image is processed and divided into a grid, and the task of the model is to **classify each grid cell as either containing wildlife or not containing wildlife**.

The goal is to create a system that can automatically analyze images and highlight regions that contain wildlife, along with generating a structured output describing the detected regions.

---

## Problem Statement

Given a dataset of images containing wildlife, the objective is to build a machine learning pipeline that:

1. Processes and standardizes the images.
2. Divides each image into a fixed grid structure.
3. Extracts meaningful handcrafted features from each grid cell.
4. Trains machine learning models to classify grid cells.
5. Produces visual and structured outputs identifying wildlife regions.

Each grid cell must be classified as:

* **1 → Contains wildlife**
* **0 → Does not contain wildlife**

---

## Image Preprocessing

Before training the models, the images are standardized according to the following rules:

* Images must have a **4:3 aspect ratio**.
* Images larger than **800 × 600 pixels** are **scaled down**.
* Images smaller than **800 × 600 pixels are not scaled up**.
* Images that do not match the required aspect ratio are **cropped appropriately**.

After preprocessing, each image has a final resolution of **800 × 600 pixels**.

---

## Grid-Based Image Representation

Each processed image is conceptually divided into an **8 × 8 grid**, resulting in **64 grid cells**.

Each grid cell is treated as an independent sample for the machine learning model.

For every grid cell, features are extracted and used to determine whether that region contains wildlife.

---

## Feature Engineering

Handcrafted features are extracted from each grid cell to represent the visual characteristics of that region.

These features help the model identify patterns such as:

* Color distributions
* Texture patterns
* Intensity variations
* Structural features within the image region

Feature engineering plays a critical role in enabling traditional machine learning models to effectively analyze image data.

---

## Machine Learning Models

The extracted features are used to train machine learning classifiers that predict whether a grid cell contains wildlife.

The models are trained and evaluated using standard data science practices including:

* Training and testing dataset splits
* Model evaluation using classification metrics
* Selection of the best performing model

The final trained model is saved as a **pickle file** for later use in prediction.

---

## Output Generation

For every input image, the trained model produces two outputs:

### 1. Visual Output

Grid cells containing wildlife are **highlighted on the image**, making them visually distinguishable.

### 2. CSV Output

A CSV file is generated containing predictions for all grid cells.

The structure of the output file is:

| Column        | Description                    |
| ------------- | ------------------------------ |
| ImageFileName | Name of the processed image    |
| c01 – c64     | Grid cell predictions (0 or 1) |

Where:

* **0 → No wildlife detected**
* **1 → Wildlife detected**

---

## Data Science Workflow

The project follows the standard data science pipeline:

1. **Problem Definition**
2. **Data Preparation**
3. **Image Preprocessing**
4. **Feature Engineering**
5. **Model Training**
6. **Model Evaluation**
7. **Prediction and Output Generation**

---


## Key Learnings

Through this project, the following concepts were explored:

* Image preprocessing and standardization
* Grid-based spatial representation of images
* Handcrafted feature engineering for image data
* Training and evaluating machine learning classifiers
* Building an end-to-end data science pipeline for image analysis

---

## Author

Mudil Goel, Arya Patil, Arnav Pandit, Khushi Kucheria <br>
Electrical Engineering<br>
IIT Bombay
