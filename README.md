# Tardis Project

This is a 1st year Epitech project. The goal was to build a tool to predict SNCF delays from a dataset.
This means clean it, make graphics to analyse trends in delays, train prediction models and make the web interface.
(All of this is describe in the Jupyter Notebook used for the project).

---

## 🛠️ Requirements

This project uses **Streamlit** and **Python**.

- Ubuntu 22.04+ or WSL2
- Python 3.12+
- `python3.12-venv` installed

---

## 🚀 Setup Instructions

Follow these steps carefully to avoid errors.

---

### 1️⃣ Install Required System Packages

Open your terminal and run:

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-full
```

### 2️⃣ Create Your Project Directory

Important: Do this inside your Linux filesystem (e.g., ~/), not in /mnt/c/.

Example:

```bash
cd ~
mkdir tardis-project
cd tardis-project
```

### 3️⃣ Create a Virtual Environment

Run:
```bash
python3.12 -m venv streamlit-env
```

### 4️⃣ Activate the Virtual Environment

Activate it:
```bash
source streamlit-env/bin/activate
```
Your prompt should now look like:
```bash
(streamlit-env) your-username@yourmachine:~/tardis-project$
```

### 5️⃣ Install Python Packages
Inside the activated virtual environment, install all dependencies:
```bash
pip install streamlit pandas matplotlib requests folium streamlit-folium pillow scikit-learn
```

### 6️⃣ Run the Application
Start Streamlit:
```bash
streamlit run tardis_dashboard.py
```