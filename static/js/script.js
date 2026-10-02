document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const msgInput = document.getElementById('msg');
    const chatMain = document.getElementById('chat');

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevents page reload

        const message = msgInput.value.trim();
        if (!message) return;

        // 1. Display User Message
        appendMessage('User', message);
        msgInput.value = '';

        try {
            // Send POST request to Flask backend
            const response = await fetch('/get', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({ msg: message })
            });

            const textData = await response.text();
            let cleanAnswer = textData;

            // Extract just the answer string if the server returned JSON
            try {
                const jsonData = JSON.parse(textData);
                cleanAnswer = jsonData.answer || textData;
            } catch (e) {
                // If it's already plain text, keep it as is
            }

            // Display Bot Response
            appendMessage('Bot', cleanAnswer);
        } catch (error) {
            console.error('Error fetching response:', error);
            appendMessage('Bot', 'Error connecting to the server. Check your Flask terminal.');
        }
    });

    function appendMessage(sender, text) {
        const msgDiv = document.createElement('div');
        msgDiv.style.margin = '10px 0';
        msgDiv.style.padding = '10px 14px';
        msgDiv.style.borderRadius = '8px';
        msgDiv.style.color = '#ffffff';
        msgDiv.style.backgroundColor = sender === 'User' ? '#1f293d' : '#00a884';
        msgDiv.style.maxWidth = '80%';
        msgDiv.style.alignSelf = sender === 'User' ? 'flex-end' : 'flex-start';
        msgDiv.innerHTML = `<strong>${sender}:</strong> ${text}`;

        chatMain.appendChild(msgDiv);
        chatMain.scrollTop = chatMain.scrollHeight;
    }
});