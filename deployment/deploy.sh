#!/bin/bash
set -e

IMAGE_NAME="indonesian-food-classifier"
CONTAINER_NAME="indonesian_food_api"
PORT="8000"
MODEL_PATH="../models/indonesian_food_cnn.h5"

# Navigate to the script's directory to ensure correct relative paths
cd "$(dirname "$0")"

echo "--- 🍜 Indonesian Food Classifier Deployment Script ---"

if ! command -v docker &> /dev/null; then
    echo "❌ ERROR: Docker is not installed. Please install Docker first."
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "❌ ERROR: Model file not found at $MODEL_PATH"
    echo "Please train the model by running: python -m src.model_training"
    exit 1
fi
echo "✅ Model file found."

if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "--- Stopping and removing existing container: $CONTAINER_NAME ---"
    docker stop "$CONTAINER_NAME" && docker rm "$CONTAINER_NAME"
fi

echo "--- Building Docker image: $IMAGE_NAME ---"
docker build -f Dockerfile -t "$IMAGE_NAME:latest" ..

echo "--- Starting container: $CONTAINER_NAME ---"
docker run -d --name "$CONTAINER_NAME" -p "$PORT:8000" "$IMAGE_NAME:latest"

echo "--- Waiting for API to be ready... ---"
sleep 5 # Give it a moment to start
for i in {1..20}; do
    if curl -s -f "http://localhost:$PORT/health" > /dev/null; then
        echo "✅ Deployment successful! 🎉"
        echo "API is running at http://localhost:$PORT"
        echo "Access documentation at http://localhost:$PORT/docs"
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo "❌ ERROR: API failed to start. Check container logs:"
docker logs "$CONTAINER_NAME"
exit 1
