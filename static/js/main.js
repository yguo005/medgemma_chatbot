const API_URL = "http://127.0.0.1:8000/chat";

async function sendMessage() {
    let userInput = document.getElementById("user-input").value;
    if (!userInput.trim()) return;

    let chatBox = document.getElementById("chat-box");

    // Display user message
    let userMessage = document.createElement("p");
    userMessage.className = "chat-message user";
    userMessage.innerHTML = `<strong>You:</strong> ${userInput}`;
    chatBox.appendChild(userMessage);
    document.getElementById("user-input").value = ""; // Clear input field

    // Call API
    try {
        // Generate or get session ID
        let sessionId = sessionStorage.getItem('session_id') || 'user_' + Date.now();
        sessionStorage.setItem('session_id', sessionId);

        let response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                query: userInput, 
                session_id: sessionId 
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        let data = await response.json();

        // Display bot response
        let botMessage = document.createElement("p");
        botMessage.className = "chat-message bot";
        
        // Handle different response types with better debugging
        console.log("Server response data:", data);
        let responseText = data.response_text || data.response || "Sorry, I couldn't process your request.";
        console.log("Using response text:", responseText);
        
        // If it's a multiple choice question, display choices
        if (data.response_type === "multiple_choice" && data.choices) {
            responseText += "<br><br><strong>Please select:</strong><br>";
            data.choices.forEach((choice, index) => {
                responseText += `${index + 1}. ${choice}<br>`;
            });
        }
        
        botMessage.innerHTML = `<strong>Bot:</strong> ${responseText}`;
        chatBox.appendChild(botMessage);

        // Scroll to latest message
        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (error) {
        console.error("Error sending message:", error);
        // Optionally display an error message in the chat
        let errorMessage = document.createElement("p");
        errorMessage.className = "chat-message bot";
        errorMessage.innerHTML = `<strong>Bot:</strong> Sorry, something went wrong. Please try again later.`;
        chatBox.appendChild(errorMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

// Send message when Enter key is pressed
document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
}); 