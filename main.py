"""Task 1 (Advanced) - Voice Assistant
NLU intent scoring, email, timed reminders, live weather, Q&A (local KB + Wikipedia),
and user-defined custom commands stored in a config file or added by voice.

Environment variables (never hard-code secrets):
  OPENWEATHER_API_KEY  - OpenWeatherMap free-tier key
  EMAIL_USER, EMAIL_PASS - TEST/dummy account (use an app password), optional SMTP_HOST
"""
import datetime
import difflib
import json
import os
import re
import smtplib
import threading
import urllib.parse
import webbrowser
from email.message import EmailMessage

import pyttsx3
import requests
import speech_recognition as sr
from nltk.stem import PorterStemmer

CONFIG_FILE = "assistant_config.json"
DEFAULT_CONFIG = {
    "contacts": {"test": "test.account@example.com"},
    "knowledge": {
        "who made you": "I was built as a Python internship project.",
        "what is python": "Python is a popular high-level programming language known for readable code.",
    },
    "custom_commands": {
        "open youtube": {"type": "url", "value": "https://www.youtube.com"},
        "open github": {"type": "url", "value": "https://www.github.com"},
        "motivate me": {"type": "say", "value": "Every expert was once a beginner. Keep building!"},
    },
}

# ---------------------------------------------------------------- config
def load_config():
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


CONFIG = load_config()

# ---------------------------------------------------------------- speech I/O
_speak_lock = threading.Lock()


def speak(text):
    print(f"Assistant: {text}")
    with _speak_lock:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()


def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            return ""
    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"You: {text}")
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        speak("The speech service is unavailable. Check your internet connection.")
        return ""


def listen_retry(retries=2):
    """Listen, asking the user to repeat if nothing was understood."""
    for _ in range(retries + 1):
        text = listen()
        if text:
            return text
        speak("Sorry, I didn't catch that. Please repeat.")
    return ""


# ---------------------------------------------------------------- NLU
stemmer = PorterStemmer()
STOP = {"a", "an", "the", "is", "it", "me", "to", "of", "what", "whats", "you", "can", "could",
        "please", "tell", "for", "my", "i", "s", "do", "would", "like", "want", "now", "hey"}

INTENT_EXAMPLES = {
    "greeting": ["hello", "hi there", "good morning", "hey assistant how are you"],
    "time": ["what time is it", "tell me the current time", "what's the time now"],
    "date": ["what is today's date", "what day is it today", "tell me the date"],
    "search": ["search for python tutorials", "google something for me", "look up recipes on the web"],
    "weather": ["what's the weather in london", "how is the weather today",
                "is it going to rain", "temperature outside"],
    "email": ["send an email", "write a mail to someone", "email my friend"],
    "reminder": ["remind me in five minutes to drink water", "set a timer for ten minutes",
                 "set a reminder", "wake me up in an hour"],
    "add_command": ["add a custom command", "teach you a new command", "create a new command"],
    "exit": ["goodbye", "exit", "stop the assistant", "quit", "see you later"],
}
QUESTION_STARTS = ("who", "what", "when", "where", "why", "how", "explain", "define", "tell me about")


def stems(text):
    words = re.findall(r"[a-z]+", text.lower())
    return {stemmer.stem(w) for w in words if w not in STOP}


def classify(text):
    """Score the sentence against example utterances (stemmed overlap + fuzzy similarity)."""
    toks = stems(text)
    best_intent, best_score = None, 0.0
    for intent, examples in INTENT_EXAMPLES.items():
        for ex in examples:
            ex_toks = stems(ex)
            if not ex_toks or not toks:
                continue
            overlap = len(toks & ex_toks)
            coverage = overlap / len(ex_toks)
            jaccard = overlap / len(toks | ex_toks)
            ratio = difflib.SequenceMatcher(None, text, ex.lower()).ratio()
            score = 0.5 * coverage + 0.3 * jaccard + 0.2 * ratio
            if overlap and score > best_score:
                best_intent, best_score = intent, score
    if best_score >= 0.35:
        return best_intent
    if text.startswith(QUESTION_STARTS):
        return "qa"
    return None


# ---------------------------------------------------------------- actions
def do_time():
    speak("The time is " + datetime.datetime.now().strftime("%I:%M %p"))


def do_date():
    speak("Today is " + datetime.datetime.now().strftime("%A, %d %B %Y"))


def do_search(text):
    topic = re.sub(r"^(please |can you )?(search|google|look up|find|browse)( the web| the internet)?"
                   r"( for| about| on)?\s*", "", text).strip()
    if not topic:
        speak("What should I search for?")
        topic = listen_retry()
    if topic:
        speak(f"Searching the web for {topic}")
        webbrowser.open("https://www.google.com/search?q=" + urllib.parse.quote_plus(topic))


def do_weather(text):
    key = os.environ.get("OPENWEATHER_API_KEY")
    if not key:
        speak("Weather needs an OpenWeatherMap API key in the OPENWEATHER_API_KEY variable.")
        return
    m = re.search(r"\b(?:in|for|at|of)\s+([a-z ]+?)(?:\s+(?:today|tomorrow|now|right now))?$", text)
    city = m.group(1).strip() if m else ""
    if not city:
        speak("Which city?")
        city = listen_retry()
        if not city:
            return
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                         params={"q": city, "appid": key, "units": "metric"}, timeout=8)
        if r.status_code == 404:
            speak(f"I couldn't find a city called {city}.")
            return
        if r.status_code == 401:
            speak("The weather API key is invalid or not active yet.")
            return
        r.raise_for_status()
        d = r.json()
        speak(f"In {d['name']} it is {d['main']['temp']:.0f} degrees Celsius with "
              f"{d['weather'][0]['description']}. Humidity is {d['main']['humidity']} percent "
              f"and wind speed is {d['wind']['speed']} metres per second.")
    except requests.RequestException:
        speak("I couldn't reach the weather service right now.")


NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20,
             "thirty": 30, "sixty": 60}
DUR_RE = re.compile(r"\b(\d+|%s)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b" % "|".join(NUM_WORDS))


def parse_duration(text):
    """Return (seconds, text_without_duration) or (None, text)."""
    m = DUR_RE.search(text)
    if not m:
        return None, text
    n = m.group(1)
    n = int(n) if n.isdigit() else NUM_WORDS[n]
    unit = m.group(2)
    mult = 3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1
    return n * mult, text[:m.start()] + text[m.end():]


def beep():
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1000, 400)
    except ImportError:
        print("\a\a\a")


def fire_reminder(message):
    beep()
    speak(f"Reminder: {message}")


def do_reminder(text):
    seconds, rest = parse_duration(text)
    if seconds is None:
        speak("For how long should I set the reminder?")
        seconds, _ = parse_duration(listen_retry())
        if seconds is None:
            speak("Sorry, I couldn't understand the duration.")
            return
    m = re.search(r"\b(?:to|about|that)\s+(.+)$", rest)
    message = m.group(1).strip() if m else "your time is up"
    timer = threading.Timer(seconds, fire_reminder, args=(message,))
    timer.daemon = True
    timer.start()
    spoken = f"{seconds // 3600} hours" if seconds >= 3600 else \
        f"{seconds // 60} minutes" if seconds >= 60 else f"{seconds} seconds"
    speak(f"Okay, I will remind you in {spoken}.")


def do_email():
    user, password = os.environ.get("EMAIL_USER"), os.environ.get("EMAIL_PASS")
    if not (user and password):
        speak("Set EMAIL_USER and EMAIL_PASS for a test account first.")
        return
    speak("Who should I send it to?")
    name = listen_retry()
    match = difflib.get_close_matches(name, CONFIG["contacts"].keys(), n=1, cutoff=0.6)
    if not match:
        speak("That contact is not in my config file.")
        return
    to_addr = CONFIG["contacts"][match[0]]
    speak("What is the subject?")
    subject = listen_retry()
    speak("What should the message say?")
    body = listen_retry()
    speak(f"Send email to {match[0]} with subject {subject}? Say yes to confirm.")
    if "yes" not in listen_retry(1):
        speak("Email cancelled.")
        return
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = user, to_addr, subject or "(no subject)"
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL(os.environ.get("SMTP_HOST", "smtp.gmail.com"), 465, timeout=15) as s:
            s.login(user, password)
            s.send_message(msg)
        speak("Email sent.")
    except (smtplib.SMTPException, OSError) as exc:
        print(exc)
        speak("I could not send the email. Check your credentials and connection.")


def wikipedia_answer(query):
    headers = {"User-Agent": "OIBSIP-VoiceAssistant/1.0"}
    s = requests.get("https://en.wikipedia.org/w/api.php", headers=headers, timeout=8,
                     params={"action": "query", "list": "search", "srsearch": query,
                             "format": "json", "srlimit": 1}).json()
    hits = s.get("query", {}).get("search", [])
    if not hits:
        return None
    title = urllib.parse.quote(hits[0]["title"].replace(" ", "_"))
    summary = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                           headers=headers, timeout=8).json()
    sentences = re.split(r"(?<=[.!?])\s+", summary.get("extract", ""))
    return " ".join(sentences[:2]) or None


def do_qa(text):
    close = difflib.get_close_matches(text, CONFIG["knowledge"].keys(), n=1, cutoff=0.7)
    if close:
        speak(CONFIG["knowledge"][close[0]])
        return
    query = re.sub(r"^(who is|who was|who are|what is|what are|what was|tell me about|explain|define|"
                   r"do you know about)\s+", "", text).strip()
    try:
        answer = wikipedia_answer(query)
    except (requests.RequestException, ValueError):
        speak("I couldn't reach the knowledge service.")
        return
    speak(answer or f"Sorry, I couldn't find anything about {query}.")


def do_add_command():
    speak("What phrase should trigger the new command?")
    phrase = listen_retry()
    if not phrase:
        return
    speak("Should it open a website, or say something? Say website or say.")
    kind = listen_retry()
    if "web" in kind:
        speak("Say the website address, for example youtube dot com.")
        value = listen_retry().replace(" dot ", ".").replace(" ", "")
        if not value.startswith("http"):
            value = "https://" + value
        CONFIG["custom_commands"][phrase] = {"type": "url", "value": value}
    else:
        speak("What should I say?")
        value = listen_retry()
        CONFIG["custom_commands"][phrase] = {"type": "say", "value": value}
    save_config(CONFIG)
    speak(f"Done. Say {phrase} to use it.")


def run_custom(text):
    match = difflib.get_close_matches(text, CONFIG["custom_commands"].keys(), n=1, cutoff=0.8)
    if not match:
        return False
    cmd = CONFIG["custom_commands"][match[0]]
    if cmd["type"] == "url":
        speak(f"Opening {match[0]}")
        webbrowser.open(cmd["value"])
    else:
        speak(cmd["value"])
    return True


def handle(text):
    """Return False when the assistant should stop."""
    if run_custom(text):
        return True
    intent = classify(text)
    if intent == "greeting":
        speak("Hello! How can I help you?")
    elif intent == "time":
        do_time()
    elif intent == "date":
        do_date()
    elif intent == "search":
        do_search(text)
    elif intent == "weather":
        do_weather(text)
    elif intent == "email":
        do_email()
    elif intent == "reminder":
        do_reminder(text)
    elif intent == "qa":
        do_qa(text)
    elif intent == "add_command":
        do_add_command()
    elif intent == "exit":
        speak("Goodbye!")
        return False
    else:
        speak("I'm not sure how to help with that yet. You can teach me by adding a custom command.")
    return True


def main():
    speak("Advanced voice assistant ready.")
    while True:
        text = listen()
        if text and not handle(text):
            break


if __name__ == "__main__":
    main()
