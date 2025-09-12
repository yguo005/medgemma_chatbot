const API_URL = "http://127.0.0.1:8000/chat";

async function selectChoice(choice, sessionId) {
    // Remove all choice buttons after selection
    let choicesContainers = document.querySelectorAll(".choices-container");
    choicesContainers.forEach(container => container.remove());
    
    // Display the user's choice
    let chatBox = document.getElementById("chat-box");
    let userMessage = document.createElement("p");
    userMessage.className = "chat-message user";
    userMessage.innerHTML = `<strong>You:</strong> ${choice}`;
    chatBox.appendChild(userMessage);
    
    // Send the choice to the server
    try {
        let response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                query: choice, 
                session_id: sessionId,
                is_choice: true
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        let data = await response.json();
        
        // DEBUG: Log the response to see what we're getting
        console.log("🔍 Server response data:", data);
        console.log("🔍 response_type:", data.response_type);
        console.log("🔍 choices:", data.choices);

        // Display bot response based on response type
        let botMessage = document.createElement("p");
        botMessage.className = "chat-message bot";
        
        let responseText;
        if (data.response_type === "diagnostic") {
            // Handle diagnostic response
            responseText = data.diagnosis_description || data.final_explanation || "Diagnosis complete.";
            console.log("🔍 Diagnostic response received:", data.diagnosis_title);
        } else if (data.response_type === "services") {
            // Handle services response
            responseText = "Thank you for providing your symptoms. Here are the available services:";
            console.log("🔍 Services response received");
        } else {
            // Handle regular responses
            responseText = data.response_text || data.response || "Sorry, I couldn't process your request.";
        }
        
        console.log("🔍 Using response text:", responseText);
        
        botMessage.innerHTML = `<strong>Bot:</strong> ${responseText}`;
        chatBox.appendChild(botMessage);
        
        // If it's a diagnostic response, show additional information
        if (data.response_type === "diagnostic") {
            if (data.diagnosis_title) {
                let titleMessage = document.createElement("p");
                titleMessage.className = "chat-message bot diagnosis-title";
                titleMessage.innerHTML = `<strong>Diagnosis:</strong> ${data.diagnosis_title}`;
                chatBox.appendChild(titleMessage);
            }
            
            if (data.recommendations && data.recommendations.length > 0) {
                let recMessage = document.createElement("p");
                recMessage.className = "chat-message bot recommendations";
                recMessage.innerHTML = `<strong>Recommendations:</strong> ${data.recommendations.join(', ')}`;
                chatBox.appendChild(recMessage);
            }
        }
        
        // If it's a services response, show available services
        if (data.response_type === "services" && data.services) {
            let servicesContainer = document.createElement("div");
            servicesContainer.className = "services-container";
            
            data.services.forEach(service => {
                let serviceItem = document.createElement("p");
                serviceItem.className = "service-item";
                serviceItem.innerHTML = `<strong>${service.name}:</strong> ${service.description}`;
                servicesContainer.appendChild(serviceItem);
            });
            
            chatBox.appendChild(servicesContainer);
        }
        
        // If it's a multiple choice question, create clickable buttons
        console.log("🔍 Checking condition:", data.response_type === "multiple_choice", "&&", !!data.choices);
        console.log("🔍 data.response_type === 'multiple_choice':", data.response_type === "multiple_choice");
        console.log("🔍 data.choices exists:", !!data.choices);
        console.log("🔍 data.choices length:", data.choices ? data.choices.length : "undefined");
        
        if (data.response_type === "multiple_choice" && data.choices) {
            console.log("✅ Creating buttons for multiple choice response");
            let choicesContainer = document.createElement("div");
            choicesContainer.className = "choices-container";
            choicesContainer.innerHTML = "<strong>Please select:</strong>";
            
            data.choices.forEach((choice, index) => {
                let choiceButton = document.createElement("button");
                choiceButton.className = "choice-button";
                choiceButton.textContent = `${index + 1}. ${choice}`;
                choiceButton.onclick = () => selectChoice(choice, sessionId);
                choicesContainer.appendChild(choiceButton);
            });
            
            chatBox.appendChild(choicesContainer);
        }

        // Scroll to latest message
        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (error) {
        console.error("Error sending choice:", error);
        let errorMessage = document.createElement("p");
        errorMessage.className = "chat-message bot";
        errorMessage.innerHTML = `<strong>Bot:</strong> Sorry, something went wrong. Please try again later.`;
        chatBox.appendChild(errorMessage);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

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
        console.log("🔍 Server response data (selectChoice):", data);
        console.log("🔍 response_type (selectChoice):", data.response_type);
        console.log("🔍 choices (selectChoice):", data.choices);
        
        let responseText;
        if (data.response_type === "diagnostic") {
            // Handle diagnostic response
            responseText = data.diagnosis_description || data.final_explanation || "Diagnosis complete.";
            console.log("🔍 Diagnostic response received (selectChoice):", data.diagnosis_title);
        } else if (data.response_type === "services") {
            // Handle services response
            responseText = "Thank you for providing your symptoms. Here are the available services:";
            console.log("🔍 Services response received (selectChoice)");
        } else {
            // Handle regular responses
            responseText = data.response_text || data.response || "Sorry, I couldn't process your request.";
        }
        console.log("🔍 Using response text (selectChoice):", responseText);
        
        botMessage.innerHTML = `<strong>Bot:</strong> ${responseText}`;
        chatBox.appendChild(botMessage);
        
        // If it's a diagnostic response, show additional information
        if (data.response_type === "diagnostic") {
            if (data.diagnosis_title) {
                let titleMessage = document.createElement("p");
                titleMessage.className = "chat-message bot diagnosis-title";
                titleMessage.innerHTML = `<strong>Diagnosis:</strong> ${data.diagnosis_title}`;
                chatBox.appendChild(titleMessage);
            }
            
            if (data.recommendations && data.recommendations.length > 0) {
                let recMessage = document.createElement("p");
                recMessage.className = "chat-message bot recommendations";
                recMessage.innerHTML = `<strong>Recommendations:</strong> ${data.recommendations.join(', ')}`;
                chatBox.appendChild(recMessage);
            }
        }
        
        // If it's a services response, show available services
        if (data.response_type === "services" && data.services) {
            let servicesContainer = document.createElement("div");
            servicesContainer.className = "services-container";
            
            data.services.forEach(service => {
                let serviceItem = document.createElement("p");
                serviceItem.className = "service-item";
                serviceItem.innerHTML = `<strong>${service.name}:</strong> ${service.description}`;
                servicesContainer.appendChild(serviceItem);
            });
            
            chatBox.appendChild(servicesContainer);
        }
        
        // If it's a multiple choice question, create clickable buttons
        if (data.response_type === "multiple_choice" && data.choices) {
            console.log("✅ Creating buttons for multiple choice response (selectChoice)");
            let choicesContainer = document.createElement("div");
            choicesContainer.className = "choices-container";
            choicesContainer.innerHTML = "<strong>Please select:</strong>";
            
            data.choices.forEach((choice, index) => {
                let choiceButton = document.createElement("button");
                choiceButton.className = "choice-button";
                choiceButton.textContent = `${index + 1}. ${choice}`;
                choiceButton.onclick = () => selectChoice(choice, sessionId);
                choicesContainer.appendChild(choiceButton);
            });
            
            chatBox.appendChild(choicesContainer);
        }

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