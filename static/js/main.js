const API_URL = "http://127.0.0.1:8000/chat";

// Generate a unique session ID for the user
let sessionId = sessionStorage.getItem('session_id');
if (!sessionId) {
    sessionId = 'user_' + Date.now() + Math.random().toString(36).substring(2, 15);
    sessionStorage.setItem('session_id', sessionId);
}

let isBotTyping = false;

// --- Refactored and Centralized Message Handling ---

function appendMessage(sender, text, responseType = 'text', choices = []) {
    const chatbox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message', `${sender}`);

    // Use marked.parse() to convert bot messages from Markdown to HTML
    const messageHtml = sender === 'bot' ? marked.parse(text) : text;
    messageDiv.innerHTML = `<strong>${sender === 'bot' ? 'Bot' : 'You'}:</strong> ${messageHtml}`;
    
    chatbox.appendChild(messageDiv);

    // Handle multiple choice buttons
    if (responseType === 'multiple_choice' && choices && choices.length > 0) {
        const choicesContainer = document.createElement('div');
        choicesContainer.classList.add('choices-container');
        
        choices.forEach(choice => {
            const button = document.createElement('button');
            button.classList.add('choice-button');
            button.innerText = choice;
            button.onclick = () => selectChoice(choice);
            choicesContainer.appendChild(button);
        });
        chatbox.appendChild(choicesContainer);
    }

    chatbox.scrollTop = chatbox.scrollHeight;
}

// --- Simplified API Interaction Functions ---

async function sendMessage() {
    const userInputField = document.getElementById("user-input");
    const userInput = userInputField.value.trim();
    if (!userInput) return;

    appendMessage('user', userInput);
    userInputField.value = ""; // Clear input field
    
    showTypingIndicator();
    isBotTyping = true;

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                query: userInput, 
                session_id: sessionId 
            })
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        handleBotResponse(data);

    } catch (error) {
        console.error("Error sending message:", error);
        appendMessage('bot', 'Sorry, something went wrong. Please try again later.');
    } finally {
        isBotTyping = false;
        hideTypingIndicator();
    }
}

async function selectChoice(choice) {
    appendMessage('user', choice);

    // Disable all choice buttons after a selection is made
    const buttons = document.querySelectorAll('.choice-button');
    buttons.forEach(button => {
        button.disabled = true;
        button.style.cursor = 'not-allowed';
        button.style.backgroundColor = '#dcdcdc';
    });
    
    showTypingIndicator();
    isBotTyping = true;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: choice,
                session_id: sessionId,
                is_choice: true
            })
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        
        const data = await response.json();
        handleBotResponse(data);

    } catch (error) {
        console.error('Error in selectChoice:', error);
        appendMessage('bot', 'An error occurred. Please try again.');
    } finally {
        isBotTyping = false;
        hideTypingIndicator();
    }
}

// --- Centralized Bot Response Handler ---

function handleBotResponse(data) {
    console.log("🔍 Server response data:", data);

    if (data.response_type === "diagnostic") {
        const { diagnosis_title, diagnosis_description, recommendations, final_explanation } = data;
        let fullResponseMarkdown = `### ${diagnosis_title}\n\n`;
        if (diagnosis_description) fullResponseMarkdown += `${diagnosis_description}\n\n`;
        if (final_explanation) fullResponseMarkdown += `**Explanation:** ${final_explanation}\n\n`;
        if (recommendations && recommendations.length > 0) {
            fullResponseMarkdown += `**Recommendations:**\n${recommendations.map(r => `* ${r}`).join('\n')}`;
        }
        appendMessage('bot', fullResponseMarkdown);

    } else if (data.response_type === "multiple_choice") {
        appendMessage('bot', data.response_text, 'multiple_choice', data.choices);

    } else if (data.response_type === "services") {
        appendMessage('bot', "Here are some services that might help. Let me know if you'd like to book one.");
        // Here you could also render the services list if needed
    } else {
        const responseText = data.response_text || data.response || "Sorry, I couldn't process your request.";
        appendMessage('bot', responseText);
    }
}

// --- UI Helpers ---

function showTypingIndicator() {
    const chatbox = document.getElementById('chat-box');
    let typingIndicator = document.getElementById('typing-indicator');
    if (!typingIndicator) {
        typingIndicator = document.createElement('p');
        typingIndicator.id = 'typing-indicator';
        typingIndicator.className = 'chat-message bot';
        typingIndicator.innerHTML = `<strong>Bot:</strong> is typing...`;
        chatbox.appendChild(typingIndicator);
    }
    typingIndicator.style.display = 'block';
    chatbox.scrollTop = chatbox.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }
}

// --- Event Listeners ---

document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});

// Initial greeting
window.onload = function() {
    const chatBox = document.getElementById('chat-box');
    if (chatBox.children.length === 0) {
        appendMessage('bot', "Hello! I am your AI Health Consultant. How can I help you today?");
    }
}; 