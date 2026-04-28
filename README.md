
---

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
.
├── app/
│   ├── __pycache__/
│   │
│   ├── detectors/
│   │   ├── image_model.py
│   │   └── video_detector.py
│   │
│   ├── models/
│   │   ├── image_detect/
│   │   │   └── xception_deepfake_base.keras
│   │   └── video-detect/
│   │       ├── model.ipynb
│   │       └── video_model.keras
│   │
│   ├── routes/
│   │   ├── __pycache__/
│   │   ├── api.py
│   │   └── auth.py
│   │
│   ├── services/
│   │   ├── __pycache__/
│   │   ├── image_scraper.py
│   │   └── ytdlp_service.py
│   │
│   ├── auth.py
│   ├── db.py
│   ├── image_model.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   └── utils.py
│
│
├── venv/
├── .env
├── .gitignore
├── cookies.txt
├── LICENSE
└── README.md                  
```

---

## ⚙️ Setup Instructions (Manual)

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

Create a `.env` file:

```
DATABASE_URL=postgresql://username:password@host:port/database
```

> ⚠️ Use Supabase Session Pooler if needed.

---

### 5️⃣ Run the server

```
uvicorn app.main:app --reload
```

---

### 6️⃣ Open API Docs

👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

# 🐳 Docker Setup (Recommended for Frontend Team)

Run the backend **without installing Python or dependencies**.

---

## ✅ Prerequisites

Install:

* Docker Desktop

---

## 🚀 Steps to Run

### 1️⃣ Clone the repository

```
git clone <your-repo-url>
cd deepfake-backend
```

---

### 2️⃣ Create `.env` file

```
DATABASE_URL=postgresql://username:password@host:port/database
```

---

### 3️⃣ Start backend

```
docker compose up --build
```

---

### 4️⃣ Access API

* Backend:
  👉 [http://localhost:8080](http://localhost:8080)

* Swagger Docs:
  👉 [http://localhost:8080/docs](http://localhost:8080/docs)

---

## 🛑 Stop the server

```
docker compose down
```

---

## 🔄 Rebuild after changes

```
docker compose up --build
```

---

## ⚠️ Notes for Frontend Team

* No need to install Python
* Make sure port **8080** is free
* `.env` file is required
* First run may take time (Docker build)

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

* `POST /auth/register`
* `POST /auth/login`
* `POST /auth/regenerate-key`

---

### Core API

* `POST /detect`
* `GET /history`

---

## 🧪 Example Request

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

* 🤖 ML model improvements
* ⏱️ Rate limiting (Redis)
* 🌐 Chrome extension
* 📊 Dashboard
* 🔐 Multiple API keys

---

## 💀 Notes

* bcrypt limit: 72 chars
* Use URL-safe DB credentials
* Use Supabase Session Pooler if needed

---

## 👨‍💻 Author

Built as part of a hackathon project – **Sentinel AI**

---

## ⭐ Contribute

Pull requests are welcome. Open an issue for major changes.

---
