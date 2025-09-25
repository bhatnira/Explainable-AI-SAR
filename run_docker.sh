#!/bin/bash

# Circular Fingerprint AutoML TPOT Explainability Docker Runner
# Usage: ./run_docker.sh [command]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧬 Circular Fingerprint AutoML TPOT Explainability${NC}"
echo -e "${BLUE}=================================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p results visualizations

# Build the Docker image
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
docker build -t circular-fingerprint-tpot .

case "${1:-train}" in
    "train")
        echo -e "${GREEN}🚀 Training TPOT model...${NC}"
        docker-compose up circular-fingerprint-tpot
        ;;
    "interpret")
        echo -e "${GREEN}🔍 Generating interpretations...${NC}"
        docker-compose up interpret
        ;;
    "optimize")
        echo -e "${GREEN}⚙️ Running parameter optimization...${NC}"
        docker-compose up optimize
        ;;
    "visualize")
        echo -e "${GREEN}🎬 Generating visualizations...${NC}"
        docker-compose up visualize
        ;;
    "all")
        echo -e "${GREEN}🎯 Running complete pipeline...${NC}"
        docker-compose up circular-fingerprint-tpot
        docker-compose up interpret
        docker-compose up optimize
        docker-compose up visualize
        ;;
    "interactive")
        echo -e "${GREEN}💻 Starting interactive container...${NC}"
        docker run -it --rm \
            -v $(pwd)/data:/workspace/data:ro \
            -v $(pwd)/results:/workspace/results \
            -v $(pwd)/visualizations:/workspace/visualizations \
            circular-fingerprint-tpot bash
        ;;
    "clean")
        echo -e "${YELLOW}🧹 Cleaning up containers and images...${NC}"
        docker-compose down
        docker rmi circular-fingerprint-tpot || true
        docker system prune -f
        ;;
    "help"|"-h"|"--help")
        echo -e "${BLUE}Usage: $0 [command]${NC}"
        echo ""
        echo -e "${YELLOW}Available commands:${NC}"
        echo -e "  ${GREEN}train${NC}       - Train TPOT model (default)"
        echo -e "  ${GREEN}interpret${NC}   - Generate interpretations"
        echo -e "  ${GREEN}optimize${NC}    - Run parameter optimization"
        echo -e "  ${GREEN}visualize${NC}   - Generate GIF visualizations"
        echo -e "  ${GREEN}all${NC}         - Run complete pipeline"
        echo -e "  ${GREEN}interactive${NC} - Start interactive container"
        echo -e "  ${GREEN}clean${NC}       - Clean up Docker resources"
        echo -e "  ${GREEN}help${NC}        - Show this help message"
        echo ""
        echo -e "${BLUE}Examples:${NC}"
        echo -e "  $0 train"
        echo -e "  $0 all"
        echo -e "  $0 interactive"
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo -e "Use '$0 help' for available commands."
        exit 1
        ;;
esac

echo -e "${GREEN}✅ Done!${NC}"
