RESEARCHAI FRONTEND REDESIGN

Files:
- App.jsx
- App.css
- DeveloperDashboard.jsx
- DeveloperDashboard.css

Setup:
1. Back up your existing src files.
2. Copy these four files into your React/Vite src folder.
3. Axios is required: npm install axios
4. Run Flask on http://127.0.0.1:5000.
5. Run the frontend with npm run dev.

Optional API URL:
Create a .env file beside package.json:
VITE_API_URL=http://127.0.0.1:5000
Restart the Vite server after editing .env.

Expected existing Flask API:
GET /api/documents -> { "documents": [...] }
POST /api/upload -> multipart form field named "file"
POST /api/ask -> { "question": "...", "selected_document_ids": [...] }

The backend API and evaluation values are not changed.
The qrels count shown as 300 must be checked against your final Colab output.
Ensure the Gemini model description matches backend/services/answer_generator.py.
