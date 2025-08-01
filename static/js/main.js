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
        let response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: userInput })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        let data = await response.json();

        // Display bot response
        let botMessage = document.createElement("p");
        botMessage.className = "chat-message bot";
        botMessage.innerHTML = `<strong>Bot:</strong> ${data.response}`;

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