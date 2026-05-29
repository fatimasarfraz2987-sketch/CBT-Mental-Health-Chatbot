/**
 * CBT Chatbot Frontend JavaScript
 * Handles chat interactions, API calls, and UI updates
 */

const API_BASE_URL = 'http://localhost:5000';
const chatWindow = document.getElementById('chatWindow');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const messageCount = document.getElementById('messageCount');

let messageCounter = 1;
let isLoading = false;

// Event Listeners
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !isLoading) {
        sendMessage();
    }
});

/**
 * Send user message to backend
 */
async function sendMessage() {
    const message = userInput.value.trim();

    if (!message || isLoading) return;

    // Add user message to chat
    addMessageToChat(message, 'user');
    userInput.value = '';
    isLoading = true;
    sendBtn.disabled = true;

    try {
        // Send to backend
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const data = await response.json();

        // Add bot response
        addMessageToChat(data.therapist_response, 'bot');

        // Update analysis panel
        updateAnalysis(data);

        // Increment message counter
        messageCounter++;
        messageCount.textContent = messageCounter;

    } catch (error) {
        console.error('Error:', error);
        addMessageToChat('Sorry, I encountered an error. Please try again.', 'bot');
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

/**
 * Add message to chat window
 * @param {string} text - Message content
 * @param {string} sender - 'user' or 'bot'
 */
function addMessageToChat(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;

    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timeDiv);
    chatWindow.appendChild(messageDiv);

    // Auto scroll to bottom
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

/**
 * Update analysis panel with sentiment, risk, and distortions
 * @param {object} data - Response data from API
 */
function updateAnalysis(data) {
    // Update Sentiment
    const sentimentDisplay = document.getElementById('sentimentDisplay');
    const sentiment = data.sentiment?.overall_sentiment || 'NEUTRAL';
    sentimentDisplay.textContent = sentiment;
    sentimentDisplay.className = `sentiment ${sentiment.toLowerCase()}`;

    // Update Risk Level
    const riskDisplay = document.getElementById('riskDisplay');
    const riskLevel = data.risk_level || 'low';
    riskDisplay.textContent = riskLevel.toUpperCase();
    riskDisplay.className = `risk ${riskLevel.toLowerCase()}`;

    // Show crisis warning if needed
    if (riskLevel === 'high' || riskLevel === 'critical') {
        showCrisisWarning(riskLevel);
    }

    // Update Distortions
    const distortionsDisplay = document.getElementById('distortionsDisplay');
    const distortions = data.distortions || [];

    if (distortions.length === 0) {
        distortionsDisplay.innerHTML = '<span class="no-distortion">None detected</span>';
    } else {
        distortionsDisplay.innerHTML = distortions
            .map(d => `<span class="distortion-tag">${d}</span>`)
            .join('');
    }
}

/**
 * Show crisis warning modal
 * @param {string} riskLevel - Risk level detected
 */
function showCrisisWarning(riskLevel) {
    const message = riskLevel === 'critical'
        ? '⚠️ CRITICAL: Please contact emergency services immediately. Call 911 or 988.'
        : '⚠️ HIGH RISK: Please consider contacting a mental health professional. Call 988.';

    // You can implement a modal here
    console.warn(message);
}

/**
 * Format timestamp
 * @returns {string} Formatted time
 */
function getTimestamp() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Initialize
console.log('CBT Chatbot frontend loaded');
userInput.focus();
