# MAIR-G14
Welcome to the repository for our MAIR project. This project is a chatbot which can help you find a restaurant.

## Functionality
The chatbot can help you find a restaurant based on the following criteria:
- Location
- Cuisine
- Price range

The chatbot will follow the following state-transition diagram to retrieve your preferences and provide you with a restaurant recommendation:
![State-transition diagram](figures/state-diagram-combined.jpg)
## Installing and running
In general, the following instructions should work on any system.
1. Open the root directory of the project `MAIR-G14`.
2. To install the required packages, run `pip install -r requirements.txt`.
3. Run `python src/main.py` to start a conversation.

If you want to use the speech recognition feature, you need to follow the following additional steps if your OS is listed below:
### Mac
1. `brew install portaudio`
2. `brew install flac`
3. `pip install pyaudio`
## Linting and formatting
When making changes, please make sure your code is properly linted and formatted.
Please use [Flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8) for linting and Black (`pip install black`)for formatting.

