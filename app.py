import streamlit as st
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ================= TRAINING DATA =================
training_sentences = [
    "hello","hi","hey",

    "i have fever","my temperature is high","fever",

    "i have cold","cough and sneeze","runny nose",

    "i have headache","my head hurts","head pain",

    "which medicine should i take","suggest medicine",
    "medicine for fever","medicine for cold",

    "book appointment","i need doctor",

    "chest pain","severe bleeding","emergency help",

    "thank you","thanks",

    "bye","goodbye"
]

labels = [
    "greeting","greeting","greeting",

    "fever","fever","fever",

    "cold","cold","cold",

    "headache","headache","headache",

    "medicine","medicine",
    "medicine","medicine",

    "appointment","appointment",

    "emergency","emergency","emergency",

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

    "fever":["You may have fever. Drink fluids and rest."],

    "cold":["You may have cold. Take steam and stay warm."],

    "headache":["Drink water and rest. Stress can cause headache."],

    "appointment":["Sure. Tell me preferred date."],

    "emergency":["⚠ EMERGENCY detected. Contact doctor immediately."],

    "thanks":["You're welcome 😊"],

    "bye":["Goodbye! Stay healthy."],

    "default":["I’m not sure. Please consult a doctor."]
}

# medicine suggestions based on symptom
medicine_map = {
    "fever":"Paracetamol is commonly used for fever after doctor's advice.",
    "cold":"Antihistamines may help for cold after doctor's advice.",
    "headache":"Mild pain reliever may help after doctor's advice."
}

# ================= FUNCTIONS =================
def detect_intent(text):
    X_test = vectorizer.transform([text])
    return model.predict(X_test)[0]

def symptom_checker(text):
    severe_words = ["severe","worst","unbearable","bleeding","chest pain"]

    if any(w in text.lower() for w in severe_words):
        return "emergency"
    return None

# ================= MEMORY =================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_symptom" not in st.session_state:
    st.session_state.last_symptom = None


# ================= UI =================
st.title("🩺 Smart AI Health Assistant")

# show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input
user_input = st.chat_input("Type message...")

if user_input:

    # show user message
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    intent = detect_intent(user_input)
    emergency = symptom_checker(user_input)

    # EMERGENCY PRIORITY
    if emergency:
        reply = responses["emergency"][0]

    # if user asked medicine
    elif intent == "medicine":
        if st.session_state.last_symptom:
            reply = medicine_map.get(
                st.session_state.last_symptom,
                "Please consult doctor before taking medicine."
            )
        else:
            reply = "Please tell your symptom first."

    else:
        reply = random.choice(responses.get(intent,responses["default"]))

        # save symptom memory
        if intent in ["fever","cold","headache"]:
            st.session_state.last_symptom = intent

    # show bot reply
    st.session_state.messages.append({"role":"assistant","content":reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
