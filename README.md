# Spam Detection Full-Stack Application

### Swinburne University of Technology
Anton Chuchuva - Student ID: 104584362
Joshua Causon - Student ID:103065507 
Kim Thongdee - Student ID:104000839 

This project is a full-stack web application that uses a machine learning model to detect spam. It features a FastAPI backend that serves the AI model and a React frontend that provides a user-friendly interface for predictions and visualisations.

## Features

- **AI-Powered Spam Detection**: Classify text as "spam" or "ham" with confidence scores.
- **File Upload**: Analyse text from uploaded files.
- **Prediction History**: View and manage a history of past predictions.
- **Interactive Dashboard**: Visualise prediction statistics and model confidence over time.
- **Email Monitoring**: (Optional) Connect to an email account to monitor for spam in real-time.

## Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Frontend**: JavaScript, React, Plotly.js
- **AI Model**: Logistic Regression with TF-IDF Vectorization

---

## Setup and Installation

### Prerequisites

- Python 3.9+
- Node.js and npm

### 1. Clone the Repository

```bash
git clone <repository-url>
cd cos30049-group9
```

### 2. Backend Setup

The backend server runs on FastAPI.

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install the required Python packages
pip install -r requirements.txt
```

### 3. Frontend Setup

The frontend is a React application.

```bash
# Navigate to the frontend directory from the root
cd Frontend

# Install the required npm packages
npm install
```

### 4. Environment Configuration (Backend)

The backend uses a `.env` file for configuration, primarily for the optional email monitoring feature.

1.  In the `backend` directory, create a file named `.env`.
2.  Add the following variables if you wish to use the email monitoring feature. Otherwise, the application will run without it.

    ```env
    # .env file in the 'backend' directory

    # Set to true to enable the email monitoring service
    EMAIL_MONITORING_ENABLED=true

    # IMAP server settings for your email provider
    EMAIL_IMAP_HOST=imap.example.com
    EMAIL_IMAP_PORT=993

    # Your email address and an app-specific password
    EMAIL_ADDRESS=your-email@example.com
    EMAIL_PASSWORD=your-app-password
    ```

    **Note**: It is strongly recommended to use an **app-specific password** rather than your main email password for security reasons.

---

## Running the Application

### 1. Run the Backend Server

```bash
# Make sure you are in the 'backend' directory with the virtual environment activated
cd backend
source .venv/bin/activate

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. You can view the interactive API documentation at `http://localhost:8000/docs`.

### 2. Run the Frontend Application

```bash
# Open a new terminal and navigate to the 'Frontend' directory
cd Frontend

# Start the React development server
npm start
```

The web application will open automatically in your browser at `http://localhost:3000`.

---

## AI Model Integration

The AI model and its associated artifacts (vectorizer and scaler) are pre-trained and included in the `backend` directory. The integration is already configured and functional out of the box.

- **Model**: `backend/models/logreg.joblib`
- **Artifacts**: `backend/artifacts/`

The original model training scripts and datasets are located in the `ai4cyber` folder for reference.
