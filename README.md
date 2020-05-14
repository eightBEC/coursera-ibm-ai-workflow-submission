# Submission for Capstone Project

## Description

This repository contains the final submission for the Coursera IBM AI Workflow Certification.
All requirements are satisfied:

- [x] Working Docker image
- [x] Unit Testing
- [x] Performance monitoring
- [x] Functional API and API documentation

## Configuration

1. Copy and rename the `.env.example` to `.env`
2. Define an API_KEY to be used for your application in the `.env`

## Installation

To use the code locally, install the dependencies. It is recommended to use a virtualenv.

```
pip install -r requirements.txt
```

## Running

The application can either be run in your local python environment or using Docker.

### Locally

1. To start the server locally, run:

   ```
   IS_LOCAL=true uvicorn app.main:app --host 0.0.0.0 --port 8089
   ```

2. Open `http://localhost:8089/docs` in your browser to see the API documentation.
3. Enter the API key defined previously to use the API routes.

### Docker

1. Build the docker container:
   ```
   docker build -t ai-workflow-capstone .
   ```
2. Start the docker container
   ```
   docker run --rm -p 8089:8080 ai-workflow-capstone:latest
   ```

### API Documentation

The API documentation is available in the docs folder complying with the OpenAPI specification 3.0.2

## Model Training

To train the model start the application as described in the section _Running_ and use the `/api/v1/model/train` route. This will train all models with the available training data.

## Testing

The `tox` tool is used for testing. To run the linter, formatter, unit test and security test, make sure you have installed `tox` then run it from your command line:

```
tox
```
