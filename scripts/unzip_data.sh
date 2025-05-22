#!/bin/bash

DATASETS_DIR="./datasets"

if [ -f "${DATASETS_DIR}/geometry3k.zip" ]; then
    echo "Unzipping geometry3K dataset..."
    unzip -q "${DATASETS_DIR}/geometry3k.zip" -d "${DATASETS_DIR}"
    if [ $? -eq 0 ]; then
        echo "geometry3K dataset unzipped successfully"
    else
        echo "Error: Failed to unzip geometry3K dataset"
        exit 1
    fi
else
    echo "Warning: geometry3K.zip file not found"
fi

if [ -f "${DATASETS_DIR}/PGPS9K.zip" ]; then
    echo "Unzipping PGPS9K dataset..."
    unzip -q "${DATASETS_DIR}/PGPS9K.zip" -d "${DATASETS_DIR}"
    if [ $? -eq 0 ]; then
        echo "PGPS9K dataset unzipped successfully"
    else
        echo "Error: Failed to unzip PGPS9K dataset"
        exit 1
    fi
else
    echo "Warning: PGPS9K.zip file not found"
fi