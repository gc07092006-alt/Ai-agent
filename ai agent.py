import random
import time
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ================= TRAINING DATA =================
training_sentences = [
    "hello","hi","hey",
    "i have fever","my temperature is high",
    "i have cold","cough and sneeze",
    "my head hurts","i have headache",
    "book appointment","i need doctor",
    "chest pain","severe bleeding",
    "thank you","thanks",
    "bye","goodbye"
]

labels = [
    "greeting","greeting","greeting",
    "fever","fever",
    "cold","cold",
    "headache","headache",
    "appointment","appointment",
    "emergency","emergency",
    "thanks","thanks",
    "bye","bye"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

model = MultinomialNB()
model.fit(X, labels)

# ================= RESPONSES =================
responses = {
    "greeting":["Hello! How can I help you today?"],
    "fever":["You may have infection. Drink fluids and rest."],
    "cold":["Take steam and rest well."],
    "headache":["Drink water and relax. Stress may cause headache."],
    "appointment":["Sure. Please tell preferred date."],
    "emergency":["⚠ This may be serious. Contact doctor immediately."],
    "thanks":["You're welcome 😊"],
    "bye":["Goodbye! Stay healthy."],
    "default":["I'm not sure. Please consult a doctor."]
}

# ================= FUNCTIONS =================
def clean_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def detect_intent(text):
    text = clean_text(text)
    X_test = vectorizer.transform([text])
    return model.predict(X_test)[0]

def symptom_checker(text):
    severe_words = ["severe","worst","unbearable","emergency","bleeding","chest pain"]
    mild_words = ["mild","slight","little"]

    text = text.lower()

    if any(word in text for word in severe_words):
        return "⚠ Serious symptoms detected. Visit doctor immediately."

    if any(word in text for word in mild_words):
        return "Symptoms seem mild. Rest and monitor."

    return None

def bot_print(text):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(0.02)
    print()

# ================= CHATBOT =================
def health_assistant():
    history = []
    user_name = ""

    bot_print("Assistant: Hello! What is your name?")

    while True:
        user_input = input("You: ")

        if user_name == "":
            user_name = user_input
            bot_print(f"Assistant: Nice to meet you {user_name}! How can I help you?")
            continue

        history.append(user_input)

        intent = detect_intent(user_input)
        severity = symptom_checker(user_input)

        if severity:
            reply = severity
        else:
            reply = random.choice(responses.get(intent, responses["default"]))

        bot_print("Assistant: " + reply)

        if intent == "bye":
            break

# ================= RUN =================
health_assistant()