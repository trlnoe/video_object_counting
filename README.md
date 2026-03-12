# Video Object Counting from YouTube

Computer Vision course project.

## Problem

Given:
- a YouTube video ID
- a question

Return the number requested in the question.

Example:

Video: N623MG6xnak  
Question: how many laptops are in the video?  
Answer: 2

## Pipeline

Youtube Video
↓
Frame + Audio Extraction
↓
Question Understanding
↓
Task Router
↓
Counting Module

## Modules

- Object counting (YOLOv8 + tracking)
- Action counting (pose estimation)
- Speech numeric reasoning (Whisper)

## Demo

Run:

streamlit run webapp/streamlit_app.py