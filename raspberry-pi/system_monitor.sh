#!/bin/bash

CSV_FILE="system_usage.csv"

# Create CSV header if the file doesn't exist
if [ ! -f "$CSV_FILE" ]; then
    echo "Timestamp,CPU (%),RAM (%)" > "$CSV_FILE"
fi

while true; do
    # Get current timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Extract CPU usage (100% - idle%)
    cpu_usage=$(top -bn1 | grep "%Cpu(s)" | awk -F',' '{print $1}')

    # Extract RAM usage percentage (used / total * 100)
    ram_usage=$(free | awk '/Mem:/ {total=$2; used=$3; printf "%.1f", (used/total)*100}')

    # Append data to CSV
    echo "$timestamp,$cpu_usage,$ram_usage" >> "$CSV_FILE"

    # Wait for 3 seconds before next iteration
    sleep 3
done