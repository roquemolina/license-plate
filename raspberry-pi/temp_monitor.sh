#!/bin/bash

CSV_FILE="temperature.csv"

# Create CSV header if the file doesn't exist
if [ ! -f "$CSV_FILE" ]; then
    echo "Timestamp,Temperature (°C)" > "$CSV_FILE"
fi

while true; do
    # Get current timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")

    # Extract temperature value (e.g., 45.7)
    temp=$(vcgencmd measure_temp | awk -F "[=']" '{print $2}')

    # Append data to CSV
    echo "$timestamp,$temp" >> "$CSV_FILE"

    # Wait for 3 seconds before next iteration
    sleep 3
done