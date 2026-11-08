"""Task 1 (Beginner) - Voice Assistant
Features: microphone input, greeting, time/date, web search, error handling, text-to-speech.
"""
import datetime
import urllib.parse
import webbrowser

import pyttsx3
import speech_recognition as sr


def speak(text):
    """Print and speak a response."""
    print(f"Assistant: {text}")
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def listen():
    """Capture one spoken command. Returns lowercase text, or '' if not understood."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Please repeat.")
            return ""
    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"You: {text}")
        return text
    except sr.UnknownValueError:
        speak("Sorry, I did not understand that. Please repeat.")
    except sr.RequestError:
        speak("The speech service is not available. Check your internet connection.")
    return ""


def web_search(command):
    topic = command
    for word in ("search for", "search", "google", "look up", "find"):
        topic = topic.replace(word, "")
    topic = topic.strip()
    if not topic:
        speak("What would you like me to search for?")
        topic = listen()
    if topic:
        speak(f"Searching the web for {topic}")
        webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote_plus(topic))


def main():
    speak("Voice assistant ready. Say hello, ask the time or date, search something, or say exit.")
    while True:
        command = listen()
        if not command:
            continue
        if "hello" in command:
            speak("Hello! How can I help you today?")
        elif "time" in command:
            speak("The time is " + datetime.datetime.now().strftime("%I:%M %p"))
        elif "date" in command or "day" in command:
            speak("Today is " + datetime.datetime.now().strftime("%A, %d %B %Y"))
        elif any(w in command for w in ("search", "google", "look up", "find")):
            web_search(command)
        elif any(w in command for w in ("exit", "quit", "goodbye", "stop")):
            speak("Goodbye!")
            break
        else:
            speak("I can greet you, tell the time or date, and search the web.")


if __name__ == "__main__":
    main()
