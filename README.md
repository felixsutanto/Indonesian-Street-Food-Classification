# Indonesian Street Food Classifier API

![GitHub Workflow Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.9-blue)
![Framework](https://img.shields.io/badge/framework-FastAPI-009688)
![Libraries](https://img.shields.io/badge/libraries-TensorFlow%20%7C%20Keras-FF6F00)
![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)

An end-to-end computer vision project that classifies images of popular Indonesian street foods. This repository demonstrates the full lifecycle of a machine learning project: from data acquisition and model training to final deployment as a containerized web service.

The trained model can classify images into three categories:
*   **Sate** (Satay)
*   **Bakso** (Meatball Soup)
*   **Martabak** (Sweet Pancake)

## Table of Contents

- [Project Overview](#project-overview)
- [API Endpoints](#api-endpoints)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Guide](#step-by-step-guide)
- [Usage](#usage)
  - [Using the API Docs](#using-the-api-docs)
  - [Using cURL](#using-curl)
- [Methodology](#methodology)
  - [Data Acquisition & Preprocessing](#data-acquisition--preprocessing)
  - [Model Architecture & Training](#model-architecture--training)
  - [Deployment](#deployment)

## Project Overview

This project provides a REST API built with FastAPI and containerized with Docker. The core of the service is a Convolutional Neural Network (CNN) trained with TensorFlow/Keras on a custom dataset of Indonesian street food images scraped from the web. The API exposes an endpoint that accepts an image upload and returns the predicted food category along with a confidence score.

![Training History Plot](outputs/training_history.png)

## API Endpoints

The following endpoints are available once the service is running:

| Method | Endpoint      | Description                                            |
| :----- | :------------ | :---------------------------------------------------   |
| `GET`  | `/`           | Displays a welcome message.                            |
| `GET`  | `/docs`       | Displays the interactive Swagger UI API documentation. |
| `GET`  | `/health`     | Checks if the API and model are running correctly.     |
| `POST` | `/predict`    | Accepts an image file and returns a classification.    |

## Technology Stack

-   **Backend:** FastAPI, Uvicorn
-   **ML/DL Framework:** TensorFlow 2.x, Keras
-   **Data Scraping:** Selenium, Requests
-   **Image Processing:** Pillow, NumPy
-   **Containerization:** Docker
-   **Deployment:** Bash Script

## Project Structure

The repository is organized into a modular and scalable structure to separate concerns:

```

indonesian-food-classifier/
├── api/                  \# FastAPI application code
├── data/                 \# Raw and processed image data
├── deployment/           \# Dockerfile and deployment scripts
├── models/               \# Trained model artifacts
├── notebooks/            \# Jupyter notebooks for exploration
├── outputs/              \# Generated files like plots
├── src/                  \# Source code for data scraping and model training
├── tests/                \# Unit and integration tests
├── .gitignore
├── requirements.txt
└── README.md

```

## Getting Started

Follow these instructions to get the project running on your local machine.

### Prerequisites

-   [Python 3.9+](https://www.python.org/downloads/)
-   [Docker Desktop](https://www.docker.com/products/docker-desktop/)
-   [Git](https://git-scm.com/downloads/)
-   A Unix-like terminal (Git Bash on Windows, or any standard terminal on macOS/Linux).

### Step-by-Step Guide

1.  **Clone the Repository:**
    ```
    git clone https://github.com/felixsutanto/Indonesian-Street-Food-Classification.git
    cd indonesian-food-classifier
    ```

2.  **Install Python Dependencies:**
    It is highly recommended to use a virtual environment.
    ```
    # Create a virtual environment
    python -m venv venv
    
    # Activate it (Windows)
    .\venv\Scripts\activate
    # Activate it (macOS/Linux)
    source venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Data Collection:**
    This script will scrape ~120 images for each food category and save them to `data/raw_images/`.
    ```
    python -m src.data_scraper
    ```

4.  **Model Training:**
    This script preprocesses the raw images, splits them into training/validation sets, trains the CNN, and saves the final model to `models/indonesian_food_cnn.h5`.
    ```
    python -m src.model_training
    ```

5.  **API Deployment:**
    This script builds the Docker image and starts the containerized API.
    ```
    # Make the script executable (only needs to be done once)
    chmod +x deployment/deploy.sh
    
    # Run the deployment script
    ./deployment/deploy.sh
    ```
    Upon successful execution, the API will be available at `http://localhost:8000`.

## Usage

Once the API is running, you can interact with it in several ways.

### Using the API Docs

The easiest way to test the API is through the interactive Swagger UI. Open your web browser and navigate to:

**[http://localhost:8000/docs](http://localhost:8000/docs)**

From there, you can expand the `/predict` endpoint, upload an image file, and execute a request directly from the browser.


### Using cURL

You can also send a request using `cURL` from your terminal. Replace `path/to/your/image.jpg` with the actual path to an image file.

```

curl -X 'POST' \
'http://localhost:8000/predict' \
-H 'accept: application/json' \
-H 'Content-Type: multipart/form-data' \
-F 'file=@/path/to/your/sate_image.jpg;type=image/jpeg'

```

**Example Response:**
```

{
"prediction": "sate",
"confidence": "98.74%",
"scores": {
"bakso": "0.12%",
"martabak": "1.14%",
"sate": "98.74%"
}
}

```

## Methodology

### Data Acquisition & Preprocessing
A custom dataset was created by scraping images from Google Images using Selenium. The raw data was then automatically split into training (80%) and validation (20%) sets. `ImageDataGenerator` from Keras was used to apply real-time data augmentation (rotation, shifts, zoom, flips) to the training set, artificially expanding the dataset to prevent overfitting.

### Model Architecture & Training
A Convolutional Neural Network (CNN) was built using TensorFlow/Keras. The architecture consists of three convolutional blocks with `Conv2D`, `BatchNormalization`, and `MaxPooling2D` layers, followed by a `Flatten` layer and two `Dense` layers. The final layer uses a `softmax` activation function for multi-class classification. The model was trained using the Adam optimizer and `categorical_crossentropy` loss.

### Deployment
The trained model and FastAPI application were packaged into a Docker container. A multi-stage `Dockerfile` was created to ensure a small and secure final image. A shell script (`deploy.sh`) automates the process of building the Docker image and running the container, making deployment simple and reproducible.

```

