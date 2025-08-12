#!/bin/bash

# Advanced AI Trading Bot - Startup Script
# This script activates the virtual environment and runs the trading bot

set -e  # Exit on error

echo "🚀 Starting Advanced AI Trading Bot..."
echo "📁 Working directory: $(pwd)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please copy .env.template to .env and configure it."
    echo "Run: cp .env.template .env && nano .env"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ main.py not found in current directory"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Export environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "✅ Environment ready"
echo "🤖 Starting bot..."
echo "📝 Logs will be written to logs/trading_bot.log"
echo "🛑 Press Ctrl+C to stop the bot gracefully"
echo

# Run the bot
python main.py