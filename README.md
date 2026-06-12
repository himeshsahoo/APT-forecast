# 🛡️ APT Forecast Project: AI-Powered Cyber Threat Detection

Welcome to the **APT Forecast Project**, an intelligent cybersecurity tool designed to detect **Advanced Persistent Threats (APTs)**. 

APTs are stealthy, continuous computer hacking processes, often orchestrated by sophisticated actors targeting specific organizations. Unlike traditional viruses that cause immediate chaos, APTs hide in the shadows of a network, slowly gathering data over months. 

This project serves as an AI "crystal ball." By analyzing massive streams of everyday computer activities, it learns to recognize the subtle digital footprints of these hidden attackers and forecasts malicious intent *before* critical damage is done.

---

## ✨ Why This Project is Impressive

*   **🧠 Deep Learning Engine:** Instead of relying on simple "block lists," this project uses state-of-the-art neural networks (Bidirectional LSTMs). It doesn't just look at isolated events; it reads a computer's history like a story, understanding the *context* of actions both forwards and backwards.
*   **🤝 The "Wisdom of the Crowd" (Ensemble AI):** To ensure high accuracy and reduce false alarms, the system deploys an "Ensemble" of three distinct AI models (Small, Medium, and Large). They independently review the network's activity and "vote" on whether an attack is happening, making the final decision highly robust.
*   **🔎 Finding the Needle in the Digital Haystack:** In cybersecurity, 99.9% of computer activity is completely normal. This AI is specially engineered with advanced mathematical weighting (`BCEWithLogitsLoss`) to spot the ultra-rare 0.1% of malicious behavior without getting distracted by the noise.
*   **⏱️ Time-Series Intelligence:** Hackers don't break in all at once. The AI groups system logs chronologically by computer, tracking sequences of events over time to catch slow, methodical breaches.

---

## 📖 How It Works (The Simple Version)

1.  **Data Ingestion (Reading the Logs):** The system consumes massive raw security logs detailing every process, network connection, and file access on a network.
2.  **Threat Labeling (Teaching the AI):** It cross-references this activity with a "Threat Intelligence Dictionary" of known bad actors and breached servers. This teaches the AI what a "bad guy" looks like.
3.  **Building the Story (Sequencing):** It groups actions by computer and organizes them chronologically. 
4.  **Prediction (Forecasting):** The trained AI reviews new, unseen "stories" and outputs a probability score predicting whether the computer is currently under a coordinated cyberattack.

---

## 🛠️ Technical Deep Dive (For Engineers)

### Project Files & Architecture

#### 1. Data Processing
*   `parse_optc.py`: The ingestion engine. Parses OpTC ECAR JSON-lines into a structured CSV. It applies our custom Threat Intelligence Dictionary to flag malicious events based on IoCs (Indicators of Compromise).
*   `prepare_sequences.py`: The data pipeline. Loads the CSV, encodes categorical features (like IP addresses and process names), and builds time-series tensors. It safely splits data into 80% training and 20% testing sets (`X_train.npy`, `y_test.npy`, etc.).
*   `inspect_data.py`: A utility script for sanity-checking the prepared datasets.

#### 2. Deep Learning Models
*   `train_lstm.py`: Trains the core **Bidirectional LSTM** model. Features include:
    *   **Layer Normalization** and **Dropout** for training stability.
    *   **Learning Rate Scheduling** and **Early Stopping** to prevent overfitting.
    *   **Class Weights (`pos_weight`)** to combat extreme dataset imbalance.
*   `train_ensemble.py`: The advanced pipeline. Trains three variations of the LSTM architecture (64, 96, and 128 hidden units). It evaluates predictions using Average Voting, Majority (Hard) Voting, and Weighted Voting based on AUC scores.
*   `test_model.py`: The evaluation suite. Calculates advanced metrics (ROC-AUC, Matthews Correlation Coefficient), performs dynamic threshold optimization, and generates beautiful visualizations (`model_test_results.png`, `roc_curve.png`).

#### 3. Configuration
*   `requirements.txt`: Python dependencies (PyTorch, Scikit-learn, Pandas, etc.).

---

## 🚀 Getting Started

Want to run the pipeline yourself? Follow these steps:

**1. Parse the raw ECAR log into CSV:**
```bash
python parse_optc.py
```
*(Note: Ensure the `INPUT_JSON` path in the script points to your raw ECAR file).*

**2. Prepare sequences for the AI:**
```bash
python prepare_sequences.py
```
*(This encodes the data and generates the `.npy` tensor files for PyTorch).*

**3. Train the AI Model:**
Train a single, highly optimized model:
```bash
python train_lstm.py
```
*Or, for maximum accuracy, train the Ensemble:*
```bash
python train_ensemble.py
```

**4. Evaluate the Results:**
Generate comprehensive test metrics and visualizations:
```bash
python test_model.py
```