import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained Scikit-learn Logistic Regression model
MODEL_PATH = "model_.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: '{MODEL_PATH}' not found. Please ensure the model file exists.")

# Inline Single Page Application HTML, CSS, and JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Retention Predictor</title>
    <!-- Font Awesome CDN -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --bg-gradient-start: #0f172a;
            --bg-gradient-end: #2e1065;
            --primary-accent: #8b5cf6;
            --primary-accent-hover: #7c3aed;
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.12);
            --input-bg: rgba(15, 23, 42, 0.6);
            --input-border: rgba(255, 255, 255, 0.15);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --danger: #ef4444;
            --danger-bg: rgba(239, 68, 68, 0.15);
            --danger-border: rgba(239, 68, 68, 0.4);
            --success: #10b981;
            --success-bg: rgba(16, 185, 129, 0.15);
            --success-border: rgba(16, 185, 129, 0.4);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: linear-gradient(-45deg, #0f172a, #1e1b4b, #311042, #2e1065);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Top Navigation Bar */
        .navbar {
            width: 100%;
            padding: 1.25rem 2rem;
            background: rgba(15, 23, 42, 0.7);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--card-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .navbar-brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        .navbar-brand i {
            color: var(--primary-accent);
            font-size: 1.5rem;
        }

        .navbar-subtitle {
            font-size: 0.875rem;
            color: var(--text-muted);
            font-weight: 400;
        }

        /* Container Layout */
        .main-container {
            width: 100%;
            max-width: 900px;
            padding: 2.5rem 1rem;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 2rem;
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Glassmorphism Card */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 1.25rem;
            padding: 2.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            animation: float 6s ease-in-out infinite;
        }

        /* Form Grid */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.5rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .form-group label {
            font-size: 0.875rem;
            font-weight: 500;
            color: #cbd5e1;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .form-group label i {
            color: var(--primary-accent);
            width: 16px;
        }

        .input-control {
            width: 100%;
            padding: 0.75rem 1rem;
            background: var(--input-bg);
            border: 1px solid var(--input-border);
            border-radius: 0.5rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-control:focus {
            border-color: var(--primary-accent);
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25);
        }

        select.input-control option {
            background-color: #0f172a;
            color: var(--text-main);
        }

        /* Validation Error Container */
        .error-banner {
            display: none;
            background: var(--danger-bg);
            border: 1px solid var(--danger-border);
            color: #fca5a5;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            font-size: 0.875rem;
        }

        .error-banner ul {
            margin-left: 1.25rem;
        }

        /* Submit Button */
        .submit-btn {
            width: 100%;
            margin-top: 2rem;
            padding: 1rem;
            background: linear-gradient(135deg, var(--primary-accent), var(--primary-accent-hover));
            border: none;
            border-radius: 0.5rem;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.4); }
            70% { box-shadow: 0 0 0 12px rgba(139, 92, 246, 0); }
            100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0); }
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #ffffff;
            animation: spin 0.8s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Result Section */
        .result-card {
            display: none;
            text-align: center;
            padding: 2rem;
            border-radius: 1rem;
            border: 1px solid transparent;
            animation: slideUp 0.5s ease-out;
            margin-top: 1rem;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-card.stay {
            background: var(--success-bg);
            border-color: var(--success-border);
            color: #6ee7b7;
        }

        .result-card.leave {
            background: var(--danger-bg);
            border-color: var(--danger-border);
            color: #fca5a5;
        }

        .result-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 0.5rem;
        }

        .result-icon {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        /* Footer */
        footer {
            margin-top: auto;
            padding: 1.5rem;
            font-size: 0.85rem;
            color: var(--text-muted);
            text-align: center;
        }

        @media (max-width: 640px) {
            .navbar { flex-direction: column; gap: 0.25rem; align-items: flex-start; }
            .glass-card { padding: 1.5rem; }
        }
    </style>
</head>
<body>

    <!-- Top Navigation Bar -->
    <nav class="navbar">
        <div class="navbar-brand">
            <i class="fa-solid fa-brain"></i>
            <span>RetainAI</span>
        </div>
        <div class="navbar-subtitle">Employee Attrition Prediction Model</div>
    </nav>

    <!-- Main Container -->
    <div class="main-container">
        <div class="glass-card">
            
            <!-- Error Banner -->
            <div id="errorBanner" class="error-banner">
                <ul id="errorList"></ul>
            </div>

            <!-- Input Form -->
            <form id="predictionForm" onsubmit="handlePrediction(event)">
                <div class="form-grid">
                    
                    <!-- Education -->
                    <div class="form-group">
                        <label><i class="fa-solid fa-user-graduate"></i> Education</label>
                        <select id="education" class="input-control" required>
                            <option value="Bachelors">Bachelors</option>
                            <option value="Masters">Masters</option>
                            <option value="PHD">PHD</option>
                        </select>
                    </div>

                    <!-- Joining Year -->
                    <div class="form-group">
                        <label><i class="fa-solid fa-calendar-days"></i> Joining Year</label>
                        <input type="number" id="joiningYear" class="input-control" placeholder="e.g. 2018" value="2018" required>
                    </div>

                    <!-- City -->
                    <div class="form-group">
                        <label><i class="fa-solid fa-city"></i> City</label>
                        <select id="city" class="input-control" required>
                            <option value="Bangalore">Bangalore</option>
                            <option value="Pune">Pune</option>
                            <option value="New Delhi">New Delhi</option>
                        </select>
                    </div>

                    <!-- Payment Tier -->
                    <div class="form-group">
                        <label><i class="fa-solid fa-credit-card"></i> Payment Tier</label>
                        <select id="paymentTier" class="input-control" required>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3" selected>3</option>
                        </select>
                    </div>

                    <!-- Age -->
                    <div class="form-group">
                        <label><i class="fa-solid fa-cake-candles"></i> Age</label>
                        <input type="number" id="age" class="input-control" placeholder="18 - 65" value="28" required>
                    </div>

                    <!-- Gender -->
                    <div class="form-group">
                        <label><i class="fa-solid fa-venus-mars"></i> Gender</label>
                        <select id="gender" class="input-control" required>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>

                    <!-- Ever Benched -->
                    <div class="form-group">
                        <label><i class="fa-solid fa-pause"></i> Ever Benched</label>
                        <select id="everBenched" class="input-control" required>
                            <option value="No">No</option>
                            <option value="Yes">Yes</option>
                        </select>
                    </div>

                    <!-- Experience In Current Domain -->
                    <div class="form-group">
                        <label><i class="fa-solid fa-briefcase"></i> Experience (Years)</label>
                        <input type="number" id="experience" class="input-control" placeholder="Years" value="5" required>
                    </div>

                </div>

                <!-- Submit Button -->
                <button type="submit" id="submitBtn" class="submit-btn">
                    <span id="btnSpinner" class="spinner"></span>
                    <span id="btnText"><i class="fa-solid fa-wand-magic-sparkles"></i> Predict Retention</span>
                </button>
            </form>

            <!-- Result Section -->
            <div id="resultCard" class="result-card">
                <div id="resultIcon" class="result-icon"></div>
                <div id="resultTitle" class="result-title"></div>
            </div>

        </div>
    </div>

    <!-- Footer -->
    <footer>
        Created with Flask + Scikit-learn
    </footer>

    <!-- JavaScript Logic -->
    <script>
        async function handlePrediction(event) {
            event.preventDefault();

            // Clear previous errors and results
            const errorBanner = document.getElementById("errorBanner");
            const errorList = document.getElementById("errorList");
            const resultCard = document.getElementById("resultCard");
            const submitBtn = document.getElementById("submitBtn");
            const btnSpinner = document.getElementById("btnSpinner");
            const btnText = document.getElementById("btnText");

            errorBanner.style.display = "none";
            errorList.innerHTML = "";
            resultCard.style.display = "none";

            // Get Input Values
            const age = parseInt(document.getElementById("age").value, 10);
            const experience = parseInt(document.getElementById("experience").value, 10);
            const joiningYear = parseInt(document.getElementById("joiningYear").value, 10);

            // Client-side Validations
            let errors = [];
            if (isNaN(age) || age < 18 || age > 65) {
                errors.push("Age must be between 18 and 65.");
            }
            if (!isNaN(experience) && !isNaN(age) && experience > age) {
                errors.push("Experience in current domain cannot exceed Age.");
            }
            if (isNaN(joiningYear) || joiningYear < 2010 || joiningYear > 2035) {
                errors.push("Joining Year must be between 2010 and 2035.");
            }

            if (errors.length > 0) {
                errors.forEach(err => {
                    const li = document.createElement("li");
                    li.textContent = err;
                    errorList.appendChild(li);
                });
                errorBanner.style.display = "block";
                return;
            }

            // Payload Payload Preparation
            const payload = {
                Education: document.getElementById("education").value,
                JoiningYear: joiningYear,
                City: document.getElementById("city").value,
                PaymentTier: parseInt(document.getElementById("paymentTier").value, 10),
                Age: age,
                Gender: document.getElementById("gender").value,
                EverBenched: document.getElementById("everBenched").value,
                ExperienceInCurrentDomain: experience
            };

            // UI Loading state
            btnSpinner.style.display = "inline-block";
            btnText.textContent = "Analyzing...";
            submitBtn.disabled = true;

            try {
                // Enforce 1 second minimum loading for UX
                const [response] = await Promise.all([
                    fetch("/predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    }),
                    new Promise(resolve => setTimeout(resolve, 1000))
                ]);

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || "An error occurred during prediction.");
                }

                // Render Result
                resultCard.className = "result-card " + (data.prediction === 0 ? "stay" : "leave");
                
                if (data.prediction === 0) {
                    document.getElementById("resultIcon").innerHTML = '<i class="fa-solid fa-circle-check"></i>';
                    document.getElementById("resultTitle").textContent = "Employee Will Stay";
                } else {
                    document.getElementById("resultIcon").innerHTML = '<i class="fa-solid fa-circle-xmark"></i>';
                    document.getElementById("resultTitle").textContent = "Employee Will Leave";
                }

                resultCard.style.display = "block";

            } catch (err) {
                const li = document.createElement("li");
                li.textContent = err.message;
                errorList.appendChild(li);
                errorBanner.style.display = "block";
            } finally {
                // Reset Button State
                btnSpinner.style.display = "none";
                btnText.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Predict Retention';
                submitBtn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

# Feature Label Encoding Maps
EDUCATION_MAP = {"Bachelors": 0, "Masters": 1, "PHD": 2}
GENDER_MAP = {"Female": 0, "Male": 1}
BENCHED_MAP = {"No": 0, "Yes": 1}
CITY_MAP = {"Bangalore": 0, "New Delhi": 1, "Pune": 2}


@app.route("/", methods=["GET"])
def home():
    """Renders the single page dashboard."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    """API Endpoint to accept form inputs and output predictions."""
    if model is None:
        return jsonify({"error": "Model file 'model_.pkl' was not loaded."}), 500

    try:
        data = request.get_json()

        # Extract features
        education_str = data.get("Education")
        joining_year = int(data.get("JoiningYear"))
        city_str = data.get("City")
        payment_tier = int(data.get("PaymentTier"))
        age = int(data.get("Age"))
        gender_str = data.get("Gender")
        ever_benched_str = data.get("EverBenched")
        experience = int(data.get("ExperienceInCurrentDomain"))

        # Backend Validations
        if not (18 <= age <= 65):
            return jsonify({"error": "Validation Failed: Age must be between 18 and 65."}), 400
        if experience > age:
            return jsonify({"error": "Validation Failed: Experience cannot exceed Age."}), 400
        if not (2010 <= joining_year <= 2035):
            return jsonify({"error": "Validation Failed: Joining Year must be between 2010 and 2035."}), 400

        # Encode categorical variables
        education = EDUCATION_MAP.get(education_str, 0)
        gender = GENDER_MAP.get(gender_str, 1)
        ever_benched = BENCHED_MAP.get(ever_benched_str, 0)
        city = CITY_MAP.get(city_str, 0)

        # Construct feature array maintaining order matching model expectation:
        # [Education, JoiningYear, City, PaymentTier, Age, Gender, EverBenched, ExperienceInCurrentDomain]
        features = np.array([[
            education,
            joining_year,
            city,
            payment_tier,
            age,
            gender,
            ever_benched,
            experience
        ]])

        # Execute prediction
        prediction_result = model.predict(features)[0]

        return jsonify({"prediction": int(prediction_result)}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
