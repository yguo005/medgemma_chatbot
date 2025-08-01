class MobileHealthConsultant {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.conversationState = 'initial';
        this.mediaRecorder = null;
        this.recordingChunks = [];
        this.isRecording = false;
        
        this.initializeEventListeners();
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    initializeEventListeners() {
        // Photo upload
        document.getElementById('photo-btn').addEventListener('click', () => {
            document.getElementById('photo-input').click();
        });

        document.getElementById('photo-input').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handlePhotoUpload(e.target.files[0]);
            }
        });

        // Voice recording
        document.getElementById('voice-btn').addEventListener('click', () => {
            this.toggleVoiceRecording();
        });

        // Text input
        document.getElementById('text-btn').addEventListener('click', () => {
            this.focusTextInput();
        });

        // Send message
        document.getElementById('send-btn').addEventListener('click', () => {
            this.sendMessage();
        });

        document.getElementById('message-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });

        // Choice button clicks (delegated)
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('choice-btn')) {
                this.handleChoiceSelection(e.target);
            }
        });
    }

    async handlePhotoUpload(file) {
        this.addUserMessage(`📷 Photo uploaded: ${file.name}`);
        
        // Convert to base64 for sending to backend
        const base64 = await this.fileToBase64(file);
        
        try {
            const response = await fetch('/analyze_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    image_data: base64,
                    filename: file.name
                })
            });

            const data = await response.json();
            this.handleBotResponse(data);
        } catch (error) {
            console.error('Error uploading photo:', error);
            this.addBotMessage('Sorry, there was an error processing your photo. Please try again.');
        }
    }

    async toggleVoiceRecording() {
        if (!this.isRecording) {
            await this.startRecording();
        } else {
            this.stopRecording();
        }
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.recordingChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                this.recordingChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.recordingChunks, { type: 'audio/wav' });
                this.handleVoiceRecording(audioBlob);
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            
            // Update UI to show recording state
            const voiceBtn = document.getElementById('voice-btn');
            voiceBtn.innerHTML = '<div class="recording-indicator"><div class="recording-dot"></div>Stop Recording</div>';
            voiceBtn.style.background = '#ef4444';
            voiceBtn.style.color = 'white';

        } catch (error) {
            console.error('Error starting recording:', error);
            this.addBotMessage('Sorry, I couldn\'t access your microphone. Please check your permissions.');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            this.isRecording = false;

            // Reset button
            const voiceBtn = document.getElementById('voice-btn');
            voiceBtn.innerHTML = '🎤 Voice';
            voiceBtn.style.background = '';
            voiceBtn.style.color = '';
        }
    }

    async handleVoiceRecording(audioBlob) {
        this.addUserMessage('🎤 Voice message recorded');

        try {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');
            formData.append('session_id', this.sessionId);

            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            this.handleBotResponse(data);
        } catch (error) {
            console.error('Error processing voice:', error);
            this.addBotMessage('Sorry, there was an error processing your voice message. Please try again.');
        }
    }

    focusTextInput() {
        document.getElementById('message-input').focus();
    }

    async sendMessage() {
        const input = document.getElementById('message-input');
        const message = input.value.trim();
        
        if (!message) return;

        this.addUserMessage(message);
        input.value = '';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: message,
                    session_id: this.sessionId
                })
            });

            const data = await response.json();
            this.handleBotResponse(data);
        } catch (error) {
            console.error('Error sending message:', error);
            this.addBotMessage('Sorry, something went wrong. Please try again later.');
        }
    }

    handleChoiceSelection(button) {
        // Mark as selected
        button.parentElement.querySelectorAll('.choice-btn').forEach(btn => {
            btn.classList.remove('selected');
        });
        button.classList.add('selected');

        // Send the choice as a message
        const choice = button.textContent.trim();
        this.addUserMessage(choice);

        // Send to backend
        this.sendChoiceToBackend(choice);
    }

    async sendChoiceToBackend(choice) {
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: choice,
                    session_id: this.sessionId,
                    is_choice: true
                })
            });

            const data = await response.json();
            this.handleBotResponse(data);
        } catch (error) {
            console.error('Error sending choice:', error);
            this.addBotMessage('Sorry, something went wrong. Please try again.');
        }
    }

    handleBotResponse(data) {
        if (data.response_type === 'multiple_choice') {
            this.addBotMessageWithChoices(data.response_text, data.choices);
        } else if (data.response_type === 'diagnostic') {
            this.addDiagnosticResult(data);
        } else if (data.response_type === 'services') {
            this.addServiceOptions(data);
        } else {
            this.addBotMessage(data.response || data.response_text);
        }
    }

    addUserMessage(message) {
        const chatContainer = document.getElementById('chat-container');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user';
        messageDiv.innerHTML = `
            <div class="message-content">${message}</div>
        `;
        chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addBotMessage(message) {
        const chatContainer = document.getElementById('chat-container');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        messageDiv.innerHTML = `
            <div class="bot-avatar">🤖</div>
            <div class="message-content">${message}</div>
        `;
        chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addBotMessageWithChoices(message, choices) {
        const chatContainer = document.getElementById('chat-container');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        
        const choiceButtons = choices.map(choice => 
            `<button class="choice-btn">${choice}</button>`
        ).join('');

        messageDiv.innerHTML = `
            <div class="bot-avatar">🤖</div>
            <div class="message-content">
                ${message}
                <div class="choice-buttons">
                    ${choiceButtons}
                </div>
            </div>
        `;
        chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addDiagnosticResult(data) {
        const chatContainer = document.getElementById('chat-container');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        
        const recommendations = data.recommendations ? 
            data.recommendations.map(rec => `<li>${rec}</li>`).join('') : '';

        messageDiv.innerHTML = `
            <div class="bot-avatar">🤖</div>
            <div class="message-content">
                <div class="diagnostic-card">
                    <div class="diagnostic-title">${data.diagnosis_title || 'Diagnostic Result'}</div>
                    <div class="diagnostic-description">${data.diagnosis_description}</div>
                    ${recommendations ? `
                        <div style="margin-top: 12px;">
                            <strong>You may need help from:</strong>
                            <ul class="recommendation-list">
                                ${recommendations}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
        chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addServiceOptions(data) {
        const chatContainer = document.getElementById('chat-container');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        
        const services = data.services.map(service => `
            <div class="service-card">
                <div class="service-header">
                    <div class="service-title">${service.title}</div>
                    <div class="service-price">$${service.price}</div>
                </div>
                <div class="service-description">${service.description}</div>
                <button class="book-btn" onclick="bookService('${service.id}')">Book Service →</button>
            </div>
        `).join('');

        messageDiv.innerHTML = `
            <div class="bot-avatar">🤖</div>
            <div class="message-content">
                <div>You can book with one of them now:</div>
                <div class="service-options">
                    ${services}
                </div>
            </div>
        `;
        chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    scrollToBottom() {
        const chatContainer = document.getElementById('chat-container');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }
}

// Global function for booking services
function bookService(serviceId) {
    console.log('Booking service:', serviceId);
    // Implement booking logic here
    alert('Booking functionality would be implemented here');
}

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    new MobileHealthConsultant();
}); 