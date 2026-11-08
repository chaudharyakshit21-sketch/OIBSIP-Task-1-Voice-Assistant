# Task 1 - Voice Assistant (Python)

Two versions:
- `voice_assistant_beginner.py` - microphone input, greeting, time/date, web search, repeat-on-error, text-to-speech.
- `voice_assistant_advanced.py` - everything above plus intent detection from free-form sentences, voice email,
  timed reminders with audible alert, live weather, Q&A, and custom commands.

## Setup
```
pip install -r requirements.txt
```
PyAudio notes: Windows `pip install pyaudio`; macOS `brew install portaudio` first; Linux `sudo apt install portaudio19-dev`.

Advanced environment variables:
```
OPENWEATHER_API_KEY=<free key from openweathermap.org>
EMAIL_USER=<TEST email address>
EMAIL_PASS=<app password for the test account>
```
## Run
```
python voice_assistant_beginner.py
python voice_assistant_advanced.py
```
## Example commands
"hello", "what time is it", "search for python tutorials", "what's the weather in London",
"remind me in 2 minutes to stretch", "send an email", "who is Albert Einstein", "add a custom command".

## How intent detection works
Each sentence is stemmed (NLTK PorterStemmer), stop words are removed, and it is scored against example
utterances for every intent using token coverage, Jaccard overlap and fuzzy string similarity. Question-style
sentences that match no intent fall back to the Q&A handler.

## Custom commands
Stored in `assistant_config.json` (created on first run) under `custom_commands`, or add one by saying
"add a custom command". Contacts for email and local Q&A answers live in the same file.

## Privacy
- Audio is captured only while listening and sent to **Google's free Web Speech API** (via `speech_recognition`) for transcription. Nothing is saved to disk.
- City names go to OpenWeatherMap; questions go to Wikipedia; email content goes to your SMTP server.
- Credentials are read from environment variables and never stored in the repo.
- `assistant_config.json` is stored locally and holds contacts, custom commands and local answers only.
- Reminders live in memory and are lost if the program exits.
