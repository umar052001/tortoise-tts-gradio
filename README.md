# Text-to-Speech Using Tortoise TTS and Gradio

This repository contains code for a Text-to-Speech (TTS) application using the Tortoise TTS model and the Gradio library for building user interfaces. The application allows users to convert text into spoken audio using the Tortoise TTS model and provides various input options, including preset voices, uploaded audio files, recorded audio, and YouTube audio.

## Overview

This application leverages the capabilities of the Tortoise TTS model, which is known for its high-quality and realistic speech synthesis. It enables users to generate speech from text input and customize the output using different voice presets and input sources.

### Features

- Choose from various preset voices for speech synthesis.
- Input text that you want to convert into spoken audio.
- Customize the model preset for different levels of speech quality.
- Upload audio files for voice customization.
- Record audio directly through your microphone.
- Generate speech from YouTube audio by providing the URL, start time, and end time (currently disabled).

## Getting Started

To use this application, follow these steps:

1. Clone the Repository

   ```bash
   git clone https://github.com/your-username/text-to-speech-tortoise-gradio.git
   cd text-to-speech-tortoise-gradio
   ```

2. Install Dependencies

   Ensure you have the required Python packages installed. You can use the following command to install them:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Application

   Run the Gradio application using the following command:

   ```bash
   python app.py
   ```

   This will start the application, and you can access it through your web browser.

## Usage

Once the application is running, you can interact with it through a web-based user interface. Here's how to use each feature:

- **Model Preset**: Choose a model preset that defines the quality of the generated speech. Options include "ultra_fast," "fast," "standard," and "high_quality."

- **Text to Speak**: Enter the text that you want to convert into speech in the provided textbox.

- **Preset Voice**: Choose a preset voice from the available options. You can select from a list of preset voices for speech synthesis.

- **Upload Audio**: Upload one or more audio files that will be used for voice customization. You can also choose whether to split the audio into smaller chunks.

- **Record Audio**: Record audio directly through your microphone. You can also choose whether to split the audio into smaller chunks.

- **Generate**: Click the "Generate" button to initiate the speech synthesis process. The generated audio will be played back, and you can listen to the result.

### YouTube Audio (Currently Disabled)

- **YouTube URL**: Enter the URL of a YouTube video that contains the audio you want to use.

- **Start Time (seconds)**: Specify the start time in seconds for extracting audio from the video.

- **End Time (seconds)**: Specify the end time in seconds for extracting audio from the video.

- **Split audio into chunks?**: Choose whether to split the audio into smaller segments for processing.

- **Generate**: Click the "Generate" button to process the YouTube audio and generate speech.

## License

This application is available under the [MIT License](LICENSE).

## Acknowledgments

- This project makes use of the Tortoise TTS model developed by James Betker. Visit the [Tortoise TTS GitHub repository](https://github.com/neonbjb/tortoise-tts) for more information.

- Gradio is used for building the user interface. Learn more about Gradio [here](https://www.gradio.app/).

## Notes

- The YouTube audio feature is currently disabled but can be enabled in future updates.
