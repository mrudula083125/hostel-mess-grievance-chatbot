import csv
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KB_PATH = DATA_DIR / "knowledge_base.csv"
GRIEVANCE_PATH = DATA_DIR / "grievances.csv"

st.set_page_config(page_title="CampusCare | Hostel & Mess Helpdesk", page_icon="🏠", layout="wide")

DEPARTMENTS = {
    "Hostel Maintenance": ["room repair", "electricity", "fan", "light", "plumbing", "water leak", "furniture", "door", "window"],
    "Mess Manager": ["food", "meal", "mess", "quality", "taste", "stale", "menu", "breakfast", "lunch", "dinner"],
    "Housekeeping": ["clean", "cleanliness", "bathroom", "washroom", "garbage", "pest", "mosquito", "hygiene"],
    "IT Support": ["wifi", "wi-fi", "internet", "network", "router"],
    "Hostel Warden": ["noise", "roommate", "security", "harassment", "curfew", "ragging", "general", "other"],
    "Water & Sanitation": ["drinking water", "water supply", "water", "tap", "filter"]
}

def ensure_files():
    DATA_DIR.mkdir(exist_ok=True)
    if not KB_PATH.exists():
        rows = [
            ["Hostel maintenance", "How do I report a broken fan or light?", "Submit a maintenance grievance with your block, room number, and a short description. The hostel maintenance team should review it.", "Hostel Maintenance"],
            ["Drinking water", "What should I do if drinking water is unavailable?", "Report the location and time when water was unavailable. For an urgent supply issue, inform the hostel warden as well.", "Water & Sanitation"],
            ["Mess food quality", "How can I complain about food quality?", "Mention the meal, date, approximate time, and issue (for example, undercooked or stale food). The grievance will be routed to the mess manager.", "Mess Manager"],
            ["Mess menu", "Where can I ask about the mess menu?", "Check the notice board or official campus communication channel. If the published menu is missing or incorrect, submit a mess-related grievance.", "Mess Manager"],
            ["Room cleanliness", "How do I report an unclean room or bathroom?", "Include your hostel block, room or bathroom location, and when the cleaning was last done. The request will be routed to housekeeping.", "Housekeeping"],
            ["Wi-Fi", "The hostel Wi-Fi is not working. What should I do?", "Include your hostel block, floor, device type if relevant, and the time the connection failed. The request will be routed to IT support.", "IT Support"],
            ["Safety", "Who should I contact about a safety concern?", "Contact the hostel warden or campus security immediately for urgent safety concerns. You may also log a grievance here; do not use the chatbot as an emergency service.", "Hostel Warden"],
            ["Noise", "How do I report repeated noise at night?", "Describe the location, approximate time, and frequency. The grievance will be routed to the hostel warden for review.", "Hostel Warden"],
            ["Grievance status", "How can I follow up on my complaint?", "Keep your grievance ID after submitting a complaint. In this demo, the administrator can review the record in the grievance CSV file.", "Hostel Warden"],
            ["General", "What details should I include in a grievance?", "Include a clear description, hostel block, room or location where relevant, date/time, and the action you are requesting. Avoid sharing passwords or unnecessary sensitive personal data.", "Hostel Warden"]
        ]
        with open(KB_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["topic", "question", "answer", "department"])
            writer.writerows(rows)
    if not GRIEVANCE_PATH.exists():
        with open(GRIEVANCE_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["grievance_id", "created_at", "student_name", "student_id", "hostel_block", "room_no", "category", "description", "department", "priority", "status"])

def load_kb():
    df = pd.read_csv(KB_PATH).fillna("")
    df["search_text"] = df["topic"].astype(str) + " " + df["question"].astype(str) + " " + df["answer"].astype(str)
    return df

def retrieve_answer(question, kb):
    corpus = kb["search_text"].astype(str).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    best_idx = int(scores.argmax())
    score = float(scores[best_idx])
    if score < 0.08:
        return ("I couldn't find a reliable answer in the campus knowledge base. You can submit a grievance below, or contact the hostel warden for guidance.", None, score)
    row = kb.iloc[best_idx]
    return (str(row["answer"]), row, score)

def classify_department(text):
    lower = text.lower()
    # Prefer the most specific water-related phrase before generic "water".
    if "drinking water" in lower or "water supply" in lower or "tap" in lower or "water filter" in lower:
        return "Water & Sanitation"
    best_department, best_count = "Hostel Warden", 0
    for department, keywords in DEPARTMENTS.items():
        count = sum(1 for word in keywords if word in lower)
        if count > best_count:
            best_department, best_count = department, count
    return best_department

def save_grievance(details):
    grievance_id = "GRV-" + datetime.now().strftime("%y%m%d") + "-" + uuid.uuid4().hex[:6].upper()
    record = {
        "grievance_id": grievance_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **details,
        "status": "Submitted"
    }
    with open(GRIEVANCE_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["grievance_id", "created_at", "student_name", "student_id", "hostel_block", "room_no", "category", "description", "department", "priority", "status"])
        writer.writerow(record)
    return grievance_id

ensure_files()
kb = load_kb()

st.title("🏠 CampusCare")
st.subheader("Hostel & Mess Grievance Redressal Assistant")
st.caption("A student helpdesk demo using retrieval-augmented question answering and rule-based grievance routing.")

with st.sidebar:
    st.header("About this project")
    st.write("**Domain:** Generative AI & Agentic AI Systems")
    st.write("**Method:** Retrieve relevant campus guidance, answer from the knowledge base, and route submitted complaints.")
    st.info("Demo notice: records are saved locally in data/grievances.csv. Use fictional data during demonstrations.")
    page = st.radio("Choose a section", ["Chatbot", "Submit grievance", "Admin dashboard"])

if page == "Chatbot":
    st.markdown("Ask about hostel facilities, mess food, Wi-Fi, cleanliness, water, or grievance procedures.")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm CampusCare. What hostel or mess issue can I help you with?"}
        ]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Type your question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        answer, row, score = retrieve_answer(prompt, kb)
        response = answer
        if row is not None:
            response += f"\n\n**Related topic:** {row['topic']}  \n**Suggested authority:** {row['department']}"
        response += "\n\nIf you need to report a new issue, open **Submit grievance** from the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

elif page == "Submit grievance":
    st.markdown("Complete the form below. The system assigns a grievance ID and routes the issue using its description and category.")
    with st.form("grievance_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            student_name = st.text_input("Student name (optional)")
            student_id = st.text_input("Student ID (optional)")
            hostel_block = st.text_input("Hostel block / building")
            room_no = st.text_input("Room number (if relevant)")
        with c2:
            category = st.selectbox("Grievance category", ["Mess / food", "Maintenance", "Water supply", "Cleanliness", "Wi-Fi / internet", "Safety / security", "Noise / roommate", "Other"])
            priority = st.selectbox("Priority", ["Normal", "High", "Urgent"])
        description = st.text_area("Describe the issue", placeholder="Example: The ceiling fan in room 204 has stopped working since this morning.")
        submitted = st.form_submit_button("Submit grievance", type="primary")
    if submitted:
        if len(description.strip()) < 10:
            st.error("Please provide at least 10 characters describing the issue.")
        else:
            department = classify_department(category + " " + description)
            grievance_id = save_grievance({
                "student_name": student_name.strip(),
                "student_id": student_id.strip(),
                "hostel_block": hostel_block.strip(),
                "room_no": room_no.strip(),
                "category": category,
                "description": description.strip(),
                "department": department,
                "priority": priority
            })
            st.success(f"Grievance submitted successfully. Your reference ID is **{grievance_id}**.")
            st.write(f"**Routed to:** {department}")
            st.caption("This is a local demo record; no real campus authority has been notified.")
            if priority == "Urgent":
                st.warning("If there is immediate danger or a medical emergency, contact campus security or local emergency services directly.")

else:
    st.header("Administrator dashboard")
    st.write("Review locally stored grievance records. This demo has no authentication; do not expose it publicly with real student data.")
    if GRIEVANCE_PATH.exists():
        records = pd.read_csv(GRIEVANCE_PATH).fillna("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total grievances", len(records))
        m2.metric("Submitted", int((records["status"] == "Submitted").sum()) if "status" in records else 0)
        m3.metric("High / urgent", int(records["priority"].isin(["High", "Urgent"]).sum()) if "priority" in records else 0)
        if not records.empty:
            st.dataframe(records.sort_values("created_at", ascending=False), use_container_width=True, hide_index=True)
            st.download_button("Download grievance CSV", records.to_csv(index=False).encode("utf-8"), "grievances_export.csv", "text/csv")
        else:
            st.info("No grievances have been submitted yet.")
    else:
        st.info("The grievance file has not been created yet.")
