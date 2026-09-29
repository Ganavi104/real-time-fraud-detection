const API_URL = "http://127.0.0.1:8000";


// =====================================================
// LOAD STATISTICS
// =====================================================

async function loadStatistics() {

    try {

        const response = await fetch(`${API_URL}/statistics`);

        if (!response.ok) {
            throw new Error("Failed to load statistics");
        }

        const data = await response.json();

        document.getElementById("totalTransactions").textContent =
            data.total_transactions;

        document.getElementById("fraudTransactions").textContent =
            data.fraud_transactions;

        document.getElementById("legitimateTransactions").textContent =
            data.legitimate_transactions;

        document.getElementById("fraudRate").textContent =
            (data.fraud_rate * 100).toFixed(2) + "%";

    } catch (error) {

        console.error("Statistics error:", error);

    }
}


// =====================================================
// LOAD TRANSACTIONS
// =====================================================

async function loadTransactions() {

    try {

        const response = await fetch(`${API_URL}/transactions`);

        if (!response.ok) {
            throw new Error("Failed to load transactions");
        }

        const transactions = await response.json();

        const table =
            document.getElementById("transactionTable");

        table.innerHTML = "";

        transactions.forEach(transaction => {

            const row = document.createElement("tr");

            row.innerHTML = `
                <td>${transaction.id}</td>

                <td>
                    ₹${Number(transaction.amount).toFixed(2)}
                </td>

                <td>
                    ${(transaction.fraud_probability * 100).toFixed(2)}%
                </td>

                <td>
                    ${transaction.prediction}
                </td>

                <td>
                    ${transaction.risk_level}
                </td>
            `;

            table.appendChild(row);

        });

    } catch (error) {

        console.error("Transaction error:", error);

    }
}


// =====================================================
// GET FORM DATA
// =====================================================

function getTransactionData() {

    const data = {};

    // Time
    data.Time = Number(document.getElementById("Time").value);

    // PCA features V1 - V28
    for (let i = 1; i <= 28; i++) {

        const field = `V${i}`;

        data[field] =
            Number(document.getElementById(field).value);
    }

    // Amount
    data.Amount =
        Number(document.getElementById("Amount").value);

    return data;
}


// =====================================================
// DISPLAY PREDICTION RESULT
// =====================================================

function displayPredictionResult(data) {

    const result =
        document.getElementById("predictionResult");

    result.style.display = "block";

    result.innerHTML = `
        <div class="result-item">
            <span class="result-label">
                Transaction ID
            </span>
            <span>
                ${data.transaction_id}
            </span>
        </div>

        <div class="result-item">
            <span class="result-label">
                Fraud Probability
            </span>
            <span>
                ${(data.fraud_probability * 100).toFixed(2)}%
            </span>
        </div>

        <div class="result-item">
            <span class="result-label">
                Decision Threshold
            </span>
            <span>
                ${(data.threshold * 100).toFixed(2)}%
            </span>
        </div>

        <div class="result-item">
            <span class="result-label">
                Prediction
            </span>
            <span>
                ${data.prediction}
            </span>
        </div>

        <div class="result-item">
            <span class="result-label">
                Risk Level
            </span>
            <span>
                ${data.risk_level}
            </span>
        </div>
    `;
}


// =====================================================
// PREDICTION FORM
// =====================================================

document
    .getElementById("predictionForm")
    .addEventListener("submit", async function(event) {

        event.preventDefault();

        const button =
            document.querySelector(".predict-button");

        const result =
            document.getElementById("predictionResult");

        try {

            // Show loading state
            button.disabled = true;
            button.textContent = "Checking...";

            result.style.display = "block";

            result.innerHTML = `
                <div class="result-item">
                    <span>
                        Analyzing transaction...
                    </span>
                </div>
            `;

            // Get transaction data
            const transactionData =
                getTransactionData();

            // Send data to FastAPI
            const response = await fetch(
                `${API_URL}/predict`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify(transactionData)
                }
            );

            // Handle API error
            if (!response.ok) {

                const errorData =
                    await response.json();

                throw new Error(
                    errorData.detail ||
                    "Prediction failed"
                );
            }

            // Get prediction response
            const data =
                await response.json();

            // Display prediction
            displayPredictionResult(data);

            // Refresh dashboard
            await loadStatistics();

            await loadTransactions();

        } catch (error) {

            console.error("Prediction error:", error);

            result.style.display = "block";

            result.innerHTML = `
                <div class="result-item">
                    <span class="result-label">
                        Error
                    </span>
                    <span>
                        ${error.message}
                    </span>
                </div>
            `;

        } finally {

            button.disabled = false;
            button.textContent = "Check Transaction";

        }

    });


// =====================================================
// LOAD DASHBOARD WHEN PAGE OPENS
// =====================================================

loadDashboard();


// =====================================================
// DASHBOARD LOADER
// =====================================================

async function loadDashboard() {

    await loadStatistics();

    await loadTransactions();

}