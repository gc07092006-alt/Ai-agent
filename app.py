import streamlit as st
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ================= ML MODEL =================
training_sentences = [
    "hello","hi","hey",
    "i have fever","my temperature is high",
    "i have cold","cough and sneeze",
    "i have headache","my head hurts",
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
def detect_intent(text):
    X_test = vectorizer.transform([text])
    return model.predict(X_test)[0]

def symptom_checker(text):
    severe = ["severe","worst","unbearable"]
    mild = ["mild","slight"]

    if any(w in text.lower() for w in severe):
        return "⚠ Serious symptoms detected. Visit doctor."

    if any(w in text.lower() for w in mild):
        return "Symptoms seem mild. Rest and monitor."

    return None

# ================= CHAT MEMORY =================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🩺 AI Virtual Health Assistant")

# display previous chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input box like chat
user_input = st.chat_input("Type your message...")

if user_input:
    # show user message
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # bot logic
    intent = detect_intent(user_input)
    severity = symptom_checker(user_input)

    if severity:
        reply = severity
    else:
        reply = random.choice(responses.get(intent,responses["default"]))

    # show bot reply
    st.session_state.messages.append({"role":"assistant","content":reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
