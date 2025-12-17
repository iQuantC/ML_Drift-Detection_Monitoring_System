#!/bin/bash

echo "Stopping all services..."
docker-compose down -v

echo "Removing images..."
docker-compose down --rmi all

echo "Removing dangling volumes..."
docker volume prune -f

echo "Removing dangling images..."
docker image prune -f

echo "Cleanup complete!"
