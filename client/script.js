document.addEventListener("DOMContentLoaded", function () {
  const questionInput = document.getElementById("question-input");
  const askButton = document.getElementById("ask-button");
  const chatMessages = document.getElementById("chat-messages");

  askButton.addEventListener("click", askQuestion);
  questionInput.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      askQuestion();
    }
  });

  function askQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    // Add user message
    addMessage(question, "user");
    questionInput.value = "";
    askButton.disabled = true;
    askButton.textContent = "සිතා බලමින්...";

    // Call API
    fetch("http://127.0.0.1:5000/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question: question }),
    })
      .then((response) => response.json())
      .then((data) => {
        let answer = formatText(data.answer);
        addMessage(answer, "assistant", true);
        if (data.sources && data.sources.length > 0) {
          const sourcesHtml =
            "<div class='sources'><strong>උල්පත්:</strong><br>" +
            data.sources
              .map((source, i) => `${i + 1}. ${source}`)
              .join("<br>") +
            "</div>";
          addMessage(sourcesHtml, "assistant", true);
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        addMessage("සමාවන්න, දෝෂයක් සිදු විය. නැවත උත්සාහ කරන්න.", "assistant");
      })
      .finally(() => {
        askButton.disabled = false;
        askButton.textContent = "අසන්න";
      });
  }

  function addMessage(text, sender, isHtml = false) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${sender}-message`;
    if (isHtml) {
      messageDiv.innerHTML = text;
    } else {
      messageDiv.textContent = text;
    }
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function formatText(text) {
    // Simple markdown: **bold** to <strong>
    return text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  }
});
