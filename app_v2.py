
# ============================================
# TELECOM COPILOT — DATASET TRAFFIC DASHBOARD
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import random
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)

from sentence_transformers import SentenceTransformer

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Telecom Copilot",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

QUEUE_PATH = "logs/query_queue.csv"
LOG_PATH   = "logs/query_logs.csv"

os.makedirs("logs", exist_ok=True)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📡 ZENDS Telecom")

menu = st.sidebar.radio(

    "Navigation",

    [
        "Customer Copilot",
        "Live Priority Queue",
        "Company Analytics"
    ]
)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    # Intent
    intent_model = joblib.load("models/intent_model.pkl")
    intent_vectorizer = joblib.load("models/tfidf.pkl")

    # Sentiment
    sentiment_model = joblib.load("models/sentiment_model.pkl")
    sentiment_vectorizer = joblib.load("models/sentiment_vectorizer.pkl")
    sentiment_decoder = joblib.load(
        "models/sentiment_decoder.pkl"
    )

    # Embedding
    embedding_model = SentenceTransformer(
        "models/embedding_model"
    )

    # RAG
    rag = joblib.load("models/rag_artifacts.joblib")

    index = rag["faiss_index"]   # ← correct key
    chunks = rag["chunks"]

    # LLM
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    llm_tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir="models/tinyllama"
    )

    llm_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir="models/tinyllama",
        torch_dtype=torch.float32
    )

    return (
        intent_model,
        intent_vectorizer,
        sentiment_model,
        sentiment_vectorizer,
        sentiment_decoder,
        embedding_model,
        index,
        chunks,
        llm_tokenizer,
        llm_model
    )


(
 intent_model,
 intent_vectorizer,
 sentiment_model,
 sentiment_vectorizer,
 sentiment_decoder,
 embedding_model,
 index,
 chunks,
 llm_tokenizer,
 llm_model
) = load_models()

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(
        "data/zends_customer_query_dataset.csv"
    )

    df["text"] = df["text"].astype(str)

    return df

dataset_df = load_dataset()

# ============================================================
# ML FUNCTIONS
# ============================================================

def predict_intent(query):

    vec = intent_vectorizer.transform([query])
    return intent_model.predict(vec)[0]


def predict_sentiment(query):

    vec = sentiment_vectorizer.transform([query])

    pred = sentiment_model.predict(vec)[0]

    sentiment = sentiment_decoder[pred]

    return sentiment
def retrieve_context(query, top_k=3):

    emb = embedding_model.encode([query])
    emb = np.array(emb).astype("float32")

    distances, idx = index.search(emb, top_k)

    # Guardrail
    if distances[0][0] > 1.2:
        return "NO_CONTEXT_FOUND"

    context = "\n".join(
        [chunks[i] for i in idx[0]]
    )

    return context

# ============================================================
# PRIORITY CLASSIFIER
# ============================================================

def assign_priority(sentiment):

    if sentiment == "Frustrated":
        return "High"
    elif sentiment == "Informational":
        return "Medium"
    else:
        return "Low"

# ============================================================
# RESPONSE ENGINE (Plug RAG here)
# ============================================================

def generate_response(query):

    context = retrieve_context(query)

    prompt = f"""
You are a ZENDS Communications telecom customer support assistant.

STRICT INSTRUCTIONS:

• Answer ONLY the customer query.
• Do NOT repeat the question.
• Do NOT include words like "Question" or "Answer".
• Do NOT provide unrelated information.
• If context is insufficient, say the query will be escalated.
• Keep the response professional and concise.

Context:
{context}

Customer Query:
{query}

Final Response:
"""


    inputs = llm_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = llm_model.generate(
    **inputs,
    max_new_tokens=120,
    min_new_tokens=30,
    temperature=0.2,
    top_p=0.8,
    repetition_penalty=1.2,
    do_sample=True,
    eos_token_id=llm_tokenizer.eos_token_id
    )

    text = llm_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return text.replace(prompt, "").strip()

# ============================================================
# ADD QUERY TO QUEUE
# ============================================================

def add_to_queue(user_type, query):

    intent = predict_intent(query)
    sentiment = predict_sentiment(query)
    priority = assign_priority(sentiment)

    row = pd.DataFrame([{

        "Time": datetime.now(),
        "User Type": user_type,
        "Query": query,
        "Intent": intent,
        "Sentiment": sentiment,
        "Priority": priority,
        "Status": "Pending"
    }])

    try:
        old = pd.read_csv(QUEUE_PATH)
        df = pd.concat([old, row])
    except:
        df = row

    df.to_csv(QUEUE_PATH, index=False)

# ============================================================
# DATASET TRAFFIC GENERATOR
# ============================================================

def generate_dataset_queries(n=50):

    sampled = dataset_df.sample(n)

    records = []

    for _, row in sampled.iterrows():

        query = row["text"]

        intent = predict_intent(query)
        sentiment = predict_sentiment(query)
        priority = assign_priority(sentiment)

        records.append({

            "Time": datetime.now(),
            "User Type": random.choice(
                ["New User", "Existing User"]
            ),
            "Query": query,
            "Intent": intent,
            "Sentiment": sentiment,
            "Priority": priority,
            "Status": "Pending"
        })

    df = pd.DataFrame(records)

    try:
        old = pd.read_csv(QUEUE_PATH)
        df = pd.concat([old, df])
    except:
        pass

    df.to_csv(QUEUE_PATH, index=False)

# ============================================================
# LOAD SORTED QUEUE
# ============================================================

def load_sorted_queue():

    if not os.path.exists(QUEUE_PATH):
        return pd.DataFrame()

    df = pd.read_csv(QUEUE_PATH)

    order = {
        "High": 0,
        "Medium": 1,
        "Low": 2
    }

    df["Rank"] = df["Priority"].map(order)

    df = df.sort_values(
        by=["Rank", "Time"]
    )

    return df.drop(columns=["Rank"])

# ============================================================
# LOGGING
# ============================================================

def log_query(user_type, query, response,
              intent, sentiment, status,
              response_time):

    time_now = datetime.now()

    row = pd.DataFrame([{
        "Time": time_now,
        "User Type": user_type,
        "Query": query,
        "Response": response,
        "Intent": intent,
        "Sentiment": sentiment,
        "Status": status,
        "Response Time": response_time
    }])

    if os.path.exists(LOG_PATH):
        old = pd.read_csv(LOG_PATH)
        df = pd.concat([old, row])
    else:
        df = row

    df.to_csv(LOG_PATH, index=False)

# ============================================================
# PROCESS SINGLE QUERY
# ============================================================

def process_next_query():

    df = load_sorted_queue()

    pending = df[df["Status"] == "Pending"]

    if len(pending) == 0:
        return None

    # Get next query
    row_index = pending.index[0]
    row = pending.loc[row_index]

    query = row["Query"]
    user_type = row["User Type"]
    intent = row["Intent"]
    sentiment = row["Sentiment"]

    # ⏱️ STEP-2 → START TIMER
    start_time = time.time()

    response = generate_response(query)

    # ⏱️ END TIMER
    end_time = time.time()

    response_time = round(end_time - start_time, 2)

    # Status
    status = (
        "Escalated"
        if "escalate" in response.lower()
        else "Resolved"
    )

    # Update queue
    df.loc[row_index, "Status"] = status
    df.to_csv(QUEUE_PATH, index=False)

    # Log query (Correct fields)
    log_query(
        user_type,
        query,
        response,
        intent,
        sentiment,
        status,
        response_time
    )

    return query, response, status
# ============================================================
# BULK PROCESS
# ============================================================

def process_bulk_queries(n=10):

    results = []

    for _ in range(n):

        res = process_next_query()

        if res is None:
            break

        results.append(res)

    return results

# ============================================================
# AUTO SOLVE
# ============================================================

def auto_solve_queue(interval=2, limit=50):

    processed = []

    for _ in range(limit):

        res = process_next_query()

        if res is None:
            break

        processed.append(res)

        time.sleep(interval)

    return processed

# ============================================================
# 👤 CUSTOMER COPILOT
# ============================================================

if menu == "Customer Copilot":

    st.title("👤 Customer AI Assistant")

    # -------------------------------
    # SESSION STATE INIT
    # -------------------------------
    if "query_text" not in st.session_state:
        st.session_state["query_text"] = ""

    # -------------------------------
    # USER TYPE
    # -------------------------------
    user_type = st.radio(
        "User Type",
        ["New User", "Existing User"]
    )

    # -------------------------------
    # QUERY INPUT (Auto-fill enabled)
    # -------------------------------
    query = st.text_area(
        "Enter telecom query",
        value=st.session_state["query_text"],
        key="query_box"
    )

    # ========================================================
    # 🔝 TOP 10 FAQs FROM LOGS
    # ========================================================

    st.markdown("## 🔝 Top Asked Queries")

    if os.path.exists(LOG_PATH):

     logs_df = pd.read_csv(LOG_PATH)

     top_queries = (
        logs_df["Query"]
        .value_counts()
        .head(10)
     )

     cols = st.columns(2)

     for i, q in enumerate(top_queries.index):

        if cols[i % 2].button(
            f"📌 {q}",
            key=f"topfaq_{i}"
        ):

            # 🔥 STEP 1 — Fill textbox
            st.session_state["query_text"] = q

            # 🔥 STEP 2 — Rerun UI
            st.rerun()

    else:
     st.info("No query logs available yet.")

    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    if st.button("Generate Response"):

        query = st.session_state["query_text"]

        if query:

            start_time = time.time()

            intent = predict_intent(query)
            sentiment = predict_sentiment(query)
            priority = assign_priority(sentiment)

            response = generate_response(query)

            response_time = round(
                time.time() - start_time, 2
            )

            status = (
                "Escalated"
                if "escalate" in response.lower()
                else "Resolved"
            )

            # ---------------------------
            # SHOW RESPONSE
            # ---------------------------
            st.success(response)

            # ---------------------------
            # INSIGHTS
            # ---------------------------
            st.markdown("### Insights")

            c1, c2, c3 = st.columns(3)

            c1.metric("Intent", intent)
            c2.metric("Sentiment", sentiment)
            c3.metric("Priority", priority)

            # ---------------------------
            # LOG
            # ---------------------------
            log_query(
                user_type,
                query,
                response,
                intent,
                sentiment,
                status,
                response_time
            )

        else:
            st.warning("Please enter a query.")

# ============================================================
# 🚦 LIVE PRIORITY QUEUE
# ============================================================

elif menu == "Live Priority Queue":

    st.title("🚦 Live Priority Queue")

    # ========================================================
    # 📊 TRAFFIC GENERATOR
    # ========================================================

    st.subheader("📊 Simulate Incoming Traffic")

    col1, col2 = st.columns([3, 1])

    with col1:

        traffic_n = st.slider(
            "Generate Dataset Queries",
            min_value=10,
            max_value=500,
            value=50,
            step=10
        )

    with col2:

        if st.button("Generate Traffic"):

            generate_dataset_queries(traffic_n)

            st.success(
                f"{traffic_n} queries added to queue."
            )

            st.rerun()

    st.markdown("---")

    # ========================================================
    # LOAD QUEUE
    # ========================================================

    if os.path.exists(QUEUE_PATH):

        df = load_sorted_queue()

        # ====================================================
        # KPI METRICS
        # ====================================================

        pending = (df["Status"] == "Pending").sum()
        resolved = (df["Status"] == "Resolved").sum()
        escalated = (df["Status"] == "Escalated").sum()

        high_priority = (df["Priority"] == "High").sum()

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Pending", pending)
        k2.metric("Resolved", resolved)
        k3.metric("Escalated", escalated)
        k4.metric("High Priority", high_priority)

        st.markdown("---")

        # ====================================================
        # PROCESSING MODE SELECTOR
        # ====================================================

        st.subheader("⚙️ Processing Mode")

        mode = st.radio(
            "Select Queue Processing Mode",
            [
                "Manual (Agent)",
                "Bulk (Team)",
                "Auto AI (Copilot)"
            ],
            horizontal=True
        )

        st.markdown("---")

        # ====================================================
        # QUEUE TABLE
        # ====================================================

        st.subheader("📋 Live Query Queue")

        st.dataframe(
            df,
            use_container_width=True
        )

        st.markdown("---")

        # ====================================================
        # MODE 1 — MANUAL PROCESSING
        # ====================================================

        if mode == "Manual (Agent)":

            st.markdown("### 👤 Agent Processing")

            if st.button("Process Next Query"):

                res = process_next_query()

                if res:

                    st.success(
                        f"Solved: {res[0]}"
                    )

                    st.rerun()

                else:
                    st.warning("No pending queries.")

        # ====================================================
        # MODE 2 — BULK PROCESSING
        # ====================================================

        elif mode == "Bulk (Team)":

            st.markdown("### 👥 Bulk Processing")

            bulk_n = st.slider(
                "Queries to Process",
                1, 50, 10
            )

            if st.button("Process Bulk Queries"):

                results = process_bulk_queries(
                    bulk_n
                )

                st.success(
                    f"{len(results)} queries solved."
                )

                st.rerun()

        # ====================================================
        # MODE 3 — AUTO AI PROCESSING
        # ====================================================

        elif mode == "Auto AI (Copilot)":

            st.markdown("### 🤖 Autonomous AI Processing")

            auto_limit = st.slider(
                "Max Queries to Auto-Solve",
                5, 200, 50
            )

            interval = st.slider(
                "Response Interval (sec)",
                1, 5, 2
            )

            if st.button("Start Auto Solving"):

                progress = st.progress(0)

                solved = 0

                for i in range(auto_limit):

                    res = process_next_query()

                    if res is None:
                        break

                    solved += 1

                    progress.progress(
                        (i + 1) / auto_limit
                    )

                    time.sleep(interval)

                st.success(
                    f"{solved} queries auto-solved."
                )

                st.rerun()

            # ================================================
            # CONTINUOUS AUTO MODE
            # ================================================

            auto_live = st.toggle(
                "Enable Continuous Auto Mode"
            )

            if auto_live:

                pending_df = df[
                    df["Status"] == "Pending"
                ]

                if len(pending_df) > 0:

                    st.info(
                        "AI Copilot solving pending queries..."
                    )

                    process_next_query()

                    time.sleep(1)

                    st.rerun()

    else:
        st.warning("No queries in queue yet.")


# ============================================================
# 🏢 COMPANY ANALYTICS
# ============================================================

elif menu == "Company Analytics":

    st.title("🏢 Company Analytics")

    if os.path.exists(LOG_PATH):

        df = pd.read_csv(LOG_PATH)

        # Ensure numeric
        df["Response Time"] = pd.to_numeric(
            df["Response Time"],
            errors="coerce"
        )

        # ==============================
        # KPI ROW
        # ==============================

        k1, k2, k3, k4, k5 = st.columns(5)

        k1.metric("Total Queries", len(df))

        k2.metric(
            "Resolved",
            (df["Status"] == "Resolved").sum()
        )

        k3.metric(
            "Escalated",
            (df["Status"] == "Escalated").sum()
        )

        k4.metric(
            "Avg Response Time",
            f"{df['Response Time'].mean():.2f} sec"
        )

        frustrated_pct = (
            (df["Sentiment"] == "Frustrated").sum()
            / len(df) * 100
        )

        k5.metric(
            "Frustrated %",
            f"{frustrated_pct:.1f}%"
        )

        st.markdown("---")

        # ==============================
        # RESPONSE TIME METRICS
        # ==============================

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Max Response Time",
            f"{df['Response Time'].max():.2f} sec"
        )

        r2.metric(
            "Min Response Time",
            f"{df['Response Time'].min():.2f} sec"
        )

        sla_breach = (
            df["Response Time"] > 30
        ).sum()

        r3.metric(
            "SLA Breaches (>30s)",
            sla_breach
        )

        st.markdown("---")

        # ==============================
        # INTENT DISTRIBUTION
        # ==============================

        c1, c2 = st.columns(2)

        with c1:

            st.subheader("Intent Distribution")

            intent_counts = df["Intent"].value_counts()

            fig1, ax1 = plt.subplots()

            bars = ax1.bar(
                intent_counts.index,
                intent_counts.values
            )

            for bar in bars:
                ax1.text(
                    bar.get_x()+bar.get_width()/2,
                    bar.get_height(),
                    int(bar.get_height()),
                    ha="center"
                )

            st.pyplot(fig1, use_container_width=True)

        # ==============================
        # SENTIMENT DISTRIBUTION
        # ==============================

        with c2:

            st.subheader("Sentiment Distribution")

            sent_counts = df["Sentiment"].value_counts()

            fig2, ax2 = plt.subplots()

            bars2 = ax2.bar(
                sent_counts.index,
                sent_counts.values
            )

            for bar in bars2:
                ax2.text(
                    bar.get_x()+bar.get_width()/2,
                    bar.get_height(),
                    int(bar.get_height()),
                    ha="center"
                )

            st.pyplot(fig2, use_container_width=True)

        st.markdown("---")

        # ==============================
        # RESPONSE TIME TREND
        # ==============================

        st.subheader("📈 Response Time Trend")

        df["Time"] = pd.to_datetime(df["Time"])
        df["Response Time"] = pd.to_numeric(
            df["Response Time"],
            errors="coerce"
        )

        trend_df = df[["Time", "Response Time"]].copy()
        trend_df = trend_df.set_index("Time")

        # Aggregation selector
        agg_level = st.selectbox(
            "Aggregation Level",
            ["Hourly", "Daily"]
        )

        if agg_level == "Hourly":
            trend_df = trend_df.resample("H").mean()
            date_format = "%d %b %H:%M"
        else:
            trend_df = trend_df.resample("D").mean()
            date_format = "%d %b"

        trend_df = trend_df.dropna()

        # Rolling average (Smoothed)
        trend_df["Rolling Avg"] = (
            trend_df["Response Time"]
            .rolling(window=3)
            .mean()
        )

        # ==========================================================
        # Plot
        # ==========================================================

        fig, ax = plt.subplots(figsize=(12,5))

        # Raw line
        ax.plot(
            trend_df.index,
            trend_df["Response Time"],
            alpha=0.4,
            label="Raw"
        )

        # Smoothed line
        ax.plot(
            trend_df.index,
            trend_df["Rolling Avg"],
            linewidth=3,
            label="Smoothed"
        )

        # SLA line
        ax.axhline(
            y=30,
            linestyle="--",
            label="SLA (30s)"
        )

        # -------------------------------
        # X-axis Formatting
        # -------------------------------

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter(date_format)
        )

        ax.xaxis.set_major_locator(
            mdates.AutoDateLocator()
        )

        plt.xticks(rotation=45, ha="right")

        ax.set_ylabel("Response Time (sec)")
        ax.set_xlabel("Time")

        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig, use_container_width=True)

        # ==============================
        # LOG TABLE
        # ==============================

        st.subheader("Query Logs")

        st.dataframe(
            df,
            use_container_width=True
        )

    else:
        st.warning("No logs available.")
