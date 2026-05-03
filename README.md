# 🚀 Sentinel AI – Backend

Backend service for a **Deepfake Detection Platform** that allows users to analyze images/videos and determine whether they are AI-generated or real.

---

## 🧠 Features

* 🔐 User Authentication (JWT + API Key)
* 🔑 Per-user API Key system
* 🔄 API Key regeneration
* 📡 Deepfake detection (URL & file upload)
* 📜 Scan history tracking
* 🗄️ PostgreSQL (Supabase)
* ⚡ High-performance APIs with FastAPI

---

## 🏗️ Tech Stack

* **Framework:** FastAPI
* **Database:** PostgreSQL (Supabase)
* **ORM:** SQLAlchemy
* **Auth:** JWT (python-jose)
* **Hashing:** Passlib (bcrypt)
* **Containerization:** Docker

---

## 📁 Project Structure

```id="r8a2lx"
.
├── app/
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
├── venv/
├── .env
├── .gitignore
├── cookies.txt
├── LICENSE
└── README.md
```

---

# 🐳 Docker Setup (Recommended)

Run the backend **without installing Python or dependencies**.

---

## ✅ Prerequisites

* Install **Docker Desktop**

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash id="nqoq2n"
git clone <your-repo-url>
cd deepfake-backend
```

---

### 2️⃣ Create `.env` file

```env id="hz7n7r"
DATABASE_URL=postgresql://username:password@host:port/database
use ipv4 instead of ipv6 in case of supabase
GEMINI_API_KEY=your_gemini_api_key
SECRET_KEY=your_secret_key_for_jwt
```

> ⚠️ Use Supabase Session Pooler if required.

---

### 3️⃣ Run the backend

```bash id="1lh4wu"
docker compose up --build
```

---

### 4️⃣ Access the API

* Backend:
  👉 [http://localhost:8080](http://localhost:8080)

* Swagger Docs:
  👉 [http://localhost:8080/docs](http://localhost:8080/docs)

---

## 🛑 Stop the server

```bash id="1p3b2w"
docker compose down
```

---

## 🔄 Rebuild after changes

```bash id="qp0c8c"
docker compose up --build
```

---

## ⚠️ Notes

* First build may take time (Docker image build)
* Ensure port **8080** is free
* `.env` file is required
* No need to install Python locally

---

## 🔐 Authentication

The API supports **two authentication methods**:

### 1️⃣ JWT Token (Recommended)

Use the token received after login:

```http id="j8ye4h"
Authorization: Bearer YOUR_JWT_TOKEN
```

---

### 2️⃣ API Key

Use the API key received during registration/login:

```http id="z7dnqf"
x-api-key: YOUR_API_KEY
```

---

### ✅ You can use **either JWT OR API Key** to access protected endpoints.

---

## 🔑 Authentication Flow

### Register

* Creates a new user
* Returns API key

---

### Login

* Returns:

  * JWT token
  * API key

---

### Regenerate API Key

* Generates a new API key
* Invalidates the old one

---

## 📡 API Endpoints

### Auth

* `POST /auth/register` → Register user
* `POST /auth/login` → Login & get JWT + API key
* `POST /auth/regenerate-key` → Generate new API key

---

### Core API

* `POST /analyse/url` → Analyze media from URL
* `POST /analyse/analyse_upload` → Analyze uploaded file
* `GET /history` → Get user scan history
* `POST /factcheck` → Fact-check a claim using Gemini API'
---

## 🧪 Example Requests

### Using JWT

```http id="3o2wfc"
POST /analyse/url

Headers:
  Authorization: Bearer YOUR_JWT_TOKEN

Body:
{
  "url": "https://example.com/video.mp4"
}
```

---

### Using API Key

```http id="49kbpw"
POST /analyse/url

Headers:
  x-api-key: YOUR_API_KEY

Body:
{
  "url": "https://example.com/video.mp4"
}
```

---

## ⚡ Notes on Usage

* Swagger UI supports authentication via the 🔒 button
* JWT is recommended for frontend apps
* API keys are useful for scripts and external integrations

---

## 🧠 Future Improvements

* 🤖 Model improvements
* ⏱️ Rate limiting (Redis)
* 🌐 Chrome extension
* 📊 Dashboard
* 🔐 Multiple API keys

---

## 💀 Notes

* bcrypt limit: 72 characters
* Use URL-safe DB credentials
* Prefer Supabase Session Pooler for production

---

## 👨‍💻 Author

Built as part of a hackathon project — **Sentinel AI**

---

## ⭐ Contribute

Pull requests are welcome. Open an issue for major changes.

---

If you want next-level polish, I can:

* add **badges (Docker, FastAPI, stars, etc.)**
* or make it look like a **top-tier GitHub repo (with screenshots + demo GIF)**
