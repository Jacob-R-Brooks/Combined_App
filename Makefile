# Variables
COMPOSE_FILE = docker-compose.yml

.PHONY: build run down test clean

build: ## Builds the Docker images for API and Monitor services.
	@echo "Building Docker images..."
	docker-compose -f $(COMPOSE_FILE) build

run: ## Starts the services in detached mode (alias for 'start').
	@echo "Starting services..."
	docker-compose -f $(COMPOSE_FILE) up -d

test: build run ## Builds, runs, and executes the test client against the live API.
	@echo "Running client tests..."
	@sleep 15 # Wait for services to fully start
	python evaluate.py
down: ## Stops and removes containers, networks, and volumes.
	@echo "🗑️ Stopping and removing containers/volumes..."
	docker-compose -f $(COMPOSE_FILE) down -v
clean: down ## Stops and removes all containers, networks, and volumes.
	@echo "Clean up complete."