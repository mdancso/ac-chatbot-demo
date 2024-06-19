# Graphisoft RAG Chatbot

## Run with Docker

1. Download [Docker](https://www.docker.com/get-started/)
2. Register an [account](https://login.docker.com/u/login)
3. Authenticate yourself with running:
    ```
    docker login
    ```
4. Build a docker image from the project:
    ```
    docker build -t chatbot .
    ```
5. Acquire an openai-api key.
6. Start the docker image:
    ```
    docker run -p 8501:8501 --name local_chatbot -e OPENAI_API_KEY=<your-openai-api-key> chatbot
    ```

7. Optionally remove the image after using the app:
    ```
    docker rm local_chatbot
    ```

## Run locally

1. Install python. The code was tested using [this version](https://www.python.org/downloads/release/python-3913/).
2. Install dependecies. Optionally create a virtual environment.
    ```
    pip install -r requirements.py
    ```
3. Run the app and enjoy:)
    ```
    streamlit run app.py
    ```
