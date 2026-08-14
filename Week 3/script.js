// ============================================================
// ML MODEL PARAMETERS (EXTRACTED FROM sugarcane_model.pkl)
// ============================================================
// === MODEL_PARAMS_START ===
const MODEL_PARAMS = {
    "features": [
        "Rainfall_mm",
        "Irrigation_liters_ha",
        "Fertilizer_kg_ha",
        "Temperature_C",
        "Water_Usage_liters_ha",
        "Soil_Type_Clayey",
        "Soil_Type_Loamy",
        "Soil_Type_Sandy"
    ],
    "mean": [
        734.9575,
        1494.4875,
        250.33875,
        29.54625,
        2229.445,
        0.32625,
        0.34625,
        0.3275
    ],
    "scale": [
        202.164693242292,
        389.50847531696974,
        58.411569902182045,
        2.8845035859745436,
        436.32305631836573,
        0.46883999136165744,
        0.47577404038051174,
        0.46930134242297167
    ],
    "coefficients": [
        4.1741133901022875,
        3.2497367665029784,
        2.8926732569527775,
        -1.1060711418694051,
        4.835083398919263,
        -0.012189184344081977,
        -0.07113346341894711,
        0.08429175202321039
    ],
    "intercept": 63.06625
};
// === MODEL_PARAMS_END ===

// ============================================================
// DOM ELEMENTS REFERENCE
// ============================================================
const form = document.getElementById("prediction-form");
const predictBtn = document.getElementById("predict-btn");
const themeToggle = document.getElementById("theme-toggle");
const mathBtn = document.getElementById("math-info-btn");
const mathOverlay = document.getElementById("math-overlay");
const closeMathBtn = document.getElementById("close-math-btn");

const resultsPanel = document.getElementById("results-panel");
const emptyState = document.getElementById("results-empty-state");
const contentState = document.getElementById("results-content-state");
const yieldValueEl = document.getElementById("yield-value");
const yieldRatingEl = document.getElementById("yield-rating");
const gaugeFill = document.getElementById("gauge-fill-ring");
const insightsList = document.getElementById("insights-list");
const summaryTableBody = document.getElementById("summary-table-body");

// ============================================================
// STATE & CONFIG
// ============================================================
let activeTheme = localStorage.getItem("cane-theme") || "dark";
document.documentElement.setAttribute("data-theme", activeTheme);

// ============================================================
// THEME SWITCHER
// ============================================================
themeToggle.addEventListener("click", () => {
    activeTheme = activeTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", activeTheme);
    localStorage.setItem("cane-theme", activeTheme);
});

// ============================================================
// MATH DETAILS DRAWER HANDLERS
// ============================================================
mathBtn.addEventListener("click", () => {
    mathOverlay.classList.add("open");
});

closeMathBtn.addEventListener("click", () => {
    mathOverlay.classList.remove("open");
});

mathOverlay.addEventListener("click", (e) => {
    if (e.target === mathOverlay) {
        mathOverlay.classList.remove("open");
    }
});

window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && mathOverlay.classList.contains("open")) {
        mathOverlay.classList.remove("open");
    }
});

// ============================================================
// ML SCALER & RUNNER
// ============================================================
function runPrediction(rainfall, irrigation, fertilizer, temperature, waterUsage, soilType) {
    // One-hot encode soil
    const isClayey = soilType === "Clayey" ? 1 : 0;
    const isLoamy = soilType === "Loamy" ? 1 : 0;
    const isSandy = soilType === "Sandy" ? 1 : 0;

    const inputVector = [
        rainfall,
        irrigation,
        fertilizer,
        temperature,
        waterUsage,
        isClayey,
        isLoamy,
        isSandy
    ];

    // Compute dot product of scaled inputs
    let predictedYield = MODEL_PARAMS.intercept;

    for (let i = 0; i < inputVector.length; i++) {
        // Standardize: (x - mean) / scale
        const scaledVal = (inputVector[i] - MODEL_PARAMS.mean[i]) / MODEL_PARAMS.scale[i];
        predictedYield += MODEL_PARAMS.coefficients[i] * scaledVal;
    }

    return predictedYield;
}

// ============================================================
// FORM SUBMISSION AND ANIMATIONS
// ============================================================
form.addEventListener("submit", (e) => {
    e.preventDefault();

    // Reset validation states
    let isFormValid = true;
    const inputs = form.querySelectorAll("input, select");
    
    inputs.forEach(input => {
        const group = input.closest(".input-group");
        if (group) group.classList.remove("invalid");

        if (!input.checkValidity()) {
            if (group) group.classList.add("invalid");
            isFormValid = false;
        }
    });

    if (!isFormValid) return;

    // Grab inputs
    const soilType = document.getElementById("soil_type").value;
    const rainfall = parseFloat(document.getElementById("rainfall").value);
    const irrigation = parseFloat(document.getElementById("irrigation").value);
    const fertilizer = parseFloat(document.getElementById("fertilizer").value);
    const temperature = parseFloat(document.getElementById("temperature").value);
    const waterUsage = parseFloat(document.getElementById("water_usage").value);

    // Show loading state on button
    predictBtn.classList.add("loading");
    predictBtn.disabled = true;

    // Simulate AI pipeline calculation time (600ms)
    setTimeout(() => {
        const yieldResult = runPrediction(rainfall, irrigation, fertilizer, temperature, waterUsage, soilType);

        // Render dashboard results
        displayResults(yieldResult, {
            soilType,
            rainfall,
            irrigation,
            fertilizer,
            temperature,
            waterUsage
        });

        predictBtn.classList.remove("loading");
        predictBtn.disabled = false;
    }, 600);
});

// ============================================================
// RENDER OUTPUTS
// ============================================================
function displayResults(prediction, inputs) {
    // Hide empty state, show results content
    emptyState.classList.add("hidden");
    contentState.classList.remove("hidden");

    // Scroll to results panel on mobile
    if (window.innerWidth <= 900) {
        resultsPanel.scrollIntoView({ behavior: "smooth" });
    }

    // 1. Counter Animation for Yield value
    animateNumber(0, prediction, 800, yieldValueEl);

    // 2. Classify Yield & Style Badge
    yieldRatingEl.className = "yield-label-badge"; // reset classes
    if (prediction < 60) {
        yieldRatingEl.textContent = "Low Yield Range";
        yieldRatingEl.classList.add("low");
    } else if (prediction >= 60 && prediction <= 75) {
        yieldRatingEl.textContent = "Optimal Yield Range";
        yieldRatingEl.classList.add("optimal");
    } else {
        yieldRatingEl.textContent = "High Yield Range";
        yieldRatingEl.classList.add("high");
    }

    // 3. Gauge Circle Animation
    // Circumference of gauge-fill is 314
    // Scale 0 to 100 tons/ha
    const clampedVal = Math.min(Math.max(prediction, 0), 100);
    const strokeOffset = 314 - (clampedVal / 100) * 314;
    gaugeFill.style.strokeDashoffset = strokeOffset;

    // 4. Generate Dynamic Insights
    generateCropInsights(prediction, inputs);

    // 5. Populate input summary table
    populateSummaryTable(inputs);
}

// Custom Counter Function
function animateNumber(start, end, duration, element) {
    const range = end - start;
    let current = start;
    const increment = range / (duration / 16); // ~60fps
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            clearInterval(timer);
            element.textContent = end.toFixed(2);
        } else {
            element.textContent = current.toFixed(2);
        }
    }, 16);
}

// Generate Expert Advisory Advice
function generateCropInsights(prediction, inputs) {
    insightsList.innerHTML = "";
    const insights = [];

    // Diagnostic 1: Soil choice recommendation
    let bestSoil = inputs.soilType;
    let bestYield = prediction;

    ["Clayey", "Loamy", "Sandy"].forEach(soil => {
        if (soil !== inputs.soilType) {
            const tempYield = runPrediction(
                inputs.rainfall, inputs.irrigation, inputs.fertilizer,
                inputs.temperature, inputs.waterUsage, soil
            );
            if (tempYield > bestYield) {
                bestYield = tempYield;
                bestSoil = soil;
            }
        }
    });

    if (bestSoil !== inputs.soilType) {
        const gain = bestYield - prediction;
        insights.push({
            icon: "🌱",
            text: `Under these exact climatic factors, planting in <strong>${bestSoil} Soil</strong> rather than ${inputs.soilType} could increase yield density by <strong>+${gain.toFixed(2)} tons/ha</strong>.`
        });
    }

    // Diagnostic 2: Irrigation Reduction (Negative Weight Weight)
    // Model Coefficient for Irrigation is negative. Reducing irrigation will raise predicted yield.
    if (inputs.irrigation > 1000) {
        const idealIrrigation = Math.max(inputs.irrigation - 300, 800);
        const lowerIrrigYield = runPrediction(
            inputs.rainfall, idealIrrigation, inputs.fertilizer,
            inputs.temperature, inputs.waterUsage, inputs.soilType
        );
        const difference = lowerIrrigYield - prediction;
        if (difference > 0) {
            insights.push({
                icon: "💧",
                text: `The model flags excess irrigation. Decreasing supplementary watering by <strong>300 L/ha</strong> could increase yield weight by <strong>+${difference.toFixed(2)} tons/ha</strong>.`
            });
        }
    }

    // Diagnostic 3: Fertilizer Boosting (Positive Weight Weight)
    // Model Coefficient for Fertilizer is positive. Increasing it raises predicted yield.
    if (inputs.fertilizer < 300) {
        const targetFertilizer = inputs.fertilizer + 50;
        const higherFertYield = runPrediction(
            inputs.rainfall, inputs.irrigation, targetFertilizer,
            inputs.temperature, inputs.waterUsage, inputs.soilType
        );
        const difference = higherFertYield - prediction;
        if (difference > 0) {
            insights.push({
                icon: "🧪",
                text: `Nutrient input is positive. Elevating fertilizer by <strong>+50 kg/ha</strong> would optimize cell growth, generating an extra <strong>+${difference.toFixed(2)} tons/ha</strong> yield.`
            });
        }
    }

    // Diagnostic 4: Temperature thermal warning
    if (inputs.temperature > 30) {
        insights.push({
            icon: "⚠️",
            text: `Growing season temperature is high (${inputs.temperature.toFixed(1)}°C). Sugarcane transpiration increases rapidly; consider heavy organic mulching to prevent rapid soil evaporation.`
        });
    } else {
        insights.push({
            icon: "📈",
            text: "Environmental conditions match moderate optimal physiological rates. Good crop development expected."
        });
    }

    // Render list
    insights.forEach(item => {
        const itemEl = document.createElement("div");
        itemEl.className = "insight-item";
        itemEl.innerHTML = `
            <span class="insight-icon">${item.icon}</span>
            <span class="insight-text">${item.text}</span>
        `;
        insightsList.appendChild(itemEl);
    });
}

// Populate table
function populateSummaryTable(inputs) {
    summaryTableBody.innerHTML = `
        <tr>
            <td>Soil Type</td>
            <td><strong>${inputs.soilType}</strong></td>
        </tr>
        <tr>
            <td>Yearly Rainfall</td>
            <td><strong>${inputs.rainfall.toFixed(0)} mm</strong></td>
        </tr>
        <tr>
            <td>Supplemental Irrigation</td>
            <td><strong>${inputs.irrigation.toFixed(0)} L/ha</strong></td>
        </tr>
        <tr>
            <td>Fertilizer Level</td>
            <td><strong>${inputs.fertilizer.toFixed(0)} kg/ha</strong></td>
        </tr>
        <tr>
            <td>Average Temperature</td>
            <td><strong>${inputs.temperature.toFixed(1)} °C</strong></td>
        </tr>
        <tr>
            <td>Water Absorbed</td>
            <td><strong>${inputs.waterUsage.toFixed(0)} L/ha</strong></td>
        </tr>
    `;
}

// ============================================================
// DYNAMIC MATH MODAL SETUP
// ============================================================
function initMathModalTable() {
    const mathTableBody = document.getElementById("math-table-body");
    const mathInterceptVal = document.getElementById("math-intercept-val");
    if (!mathTableBody || !mathInterceptVal) return;

    const featureLabels = {
        "Rainfall_mm": "Rainfall (mm)",
        "Irrigation_liters_ha": "Irrigation (L/ha)",
        "Fertilizer_kg_ha": "Fertilizer (kg/ha)",
        "Temperature_C": "Temperature (°C)",
        "Water_Usage_liters_ha": "Water Usage (L/ha)",
        "Soil_Type_Clayey": "Soil Type - Clayey",
        "Soil_Type_Loamy": "Soil Type - Loamy",
        "Soil_Type_Sandy": "Soil Type - Sandy"
    };

    mathTableBody.innerHTML = "";
    MODEL_PARAMS.features.forEach((feature, index) => {
        const mean = MODEL_PARAMS.mean[index];
        const scale = MODEL_PARAMS.scale[index];
        const coef = MODEL_PARAMS.coefficients[index];
        
        const label = featureLabels[feature] || feature;
        const coefClass = coef >= 0 ? "positive" : "negative";
        const coefSign = coef >= 0 ? "+" : "";

        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${label}</td>
            <td>${mean.toFixed(2)}</td>
            <td>${scale.toFixed(2)}</td>
            <td class="${coefClass}">${coefSign}${coef.toFixed(4)}</td>
        `;
        mathTableBody.appendChild(row);
    });

    mathInterceptVal.textContent = `${MODEL_PARAMS.intercept.toFixed(2)} tons/ha`;
}

// Initialize when elements load
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initMathModalTable);
} else {
    initMathModalTable();
}
