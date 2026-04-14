# 🚀 Sentinel AI – Backend

Backend service for a **Deepfake Detection Platform** that allows users to analyze images/videos and determine whether they are AI-generated or real.

---

## 🧠 Features

* 🔐 User Authentication (Register/Login with JWT)
* 🔑 API Key System (per user)
* 🔄 API Key Regeneration
* 📡 Deepfake Detection Endpoint (extensible for ML models)
* 📜 Scan History Tracking
* 🗄️ PostgreSQL Database (via Supabase)
* ⚡ FastAPI for high-performance APIs

---

## 🏗️ Tech Stack

* **Framework:** FastAPI
* **Database:** PostgreSQL (Supabase)
* **ORM:** SQLAlchemy
* **Authentication:** JWT (python-jose)
* **Password Hashing:** Passlib (bcrypt)

---

## 📁 Project Structure

```
app/
 ├── main.py          # Entry point
 ├── db.py            # Database connection
 ├── models.py        # SQLAlchemy models
 ├── schemas.py       # Request/response schemas
 ├── auth.py          # JWT + auth utilities
 ├── utils.py         # Helper functions
 ├── routes/
 │    ├── auth.py     # Auth endpoints
 │    ├── api.py      # Core API endpoints
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```
git clone <your-repo-url>
cd deepfake-backend
```

---

### 2️⃣ Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```
pip install fastapi uvicorn sqlalchemy psycopg2-binary passlib[bcrypt] python-jose python-dotenv email-validator
```

---

### 4️⃣ Configure environment variables

Create a `.env` file in the root directory:

```
DATABASE_URL=postgresql://username:password@host:port/database
```

> ⚠️ Use Supabase **Session Pooler (IPv4 compatible)** if direct connection fails.

---

### 5️⃣ Run the server

```
uvicorn app.main:app --reload
```

---

### 6️⃣ Open API Docs

👉 http://127.0.0.1:8000/docs

---

## 🔐 Authentication Flow

### Register

* Creates user
* Returns API key

### Login

* Returns JWT token + API key

---

## 🔑 API Key Usage

All protected endpoints require:

```
x-api-key: YOUR_API_KEY
```

---

## 📡 API Endpoints

### Auth

* `POST /auth/register` → Register new user
* `POST /auth/login` → Login user
* `POST /auth/regenerate-key` → Generate new API key

---

### Core API

* `POST /detect` → Detect deepfake
* `GET /history` → Get scan history

---

## 🧪 Example Request

### Detect Deepfake

```
POST /detect
Headers:
  x-api-key: YOUR_API_KEY

Body:
{
  "input_data": "https://example.com/video.mp4"
}
```

---

## 🧠 Future Improvements

* 🤖 Integrate ML model for real detection
* ⏱️ Rate limiting (Redis)
* 🌐 Chrome extension support
* 📊 Dashboard for analytics
* 🔐 Multiple API keys per user

---

## 💀 Notes

* Ensure passwords do not exceed bcrypt limit (72 chars)
* Use URL-safe encoding for DB credentials
* Use Session Pooler if facing network issues with Supabase

---

## 👨‍💻 Author

Built as part of a hackathon project – **Sentinel AI**

---

## ⭐ Contribute

Pull requests are welcome. For major changes, please open an issue first.

---
