// This script handles the form submission and toggling of the form.
document.addEventListener("DOMContentLoaded", () => {
  const toggleBtn = document.getElementById("toggleForm");
  const form = document.getElementById("preferencesForm");
  const successSection = document.getElementById("successSection");
  const goBackBtn = document.getElementById("goBackBtn");

  document.getElementById("submitPrompt").addEventListener("click", () => {
    const username = document.getElementById("usernameInput").value.trim();
    const prompt = document.getElementById("promptInput").value.trim();

    if (!username) {
      alert("Please enter your name.");
      return;
    }

    const data = {
      username,
      timestamp: new Date().toISOString()
    };
    if (prompt) data.prompt = prompt;

    fetch("/submit-preferences", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    })
      .then(res => res.json())
      .then(response => {
        console.log("Saved:", data);
        document.querySelector(".container").classList.add("hidden");
        successSection.classList.remove("hidden");
        setTimeout(() => location.reload(), 60000);
      })
      .catch(err => {
        alert("Error submitting data.");
        console.error(err);
      });
  });

  document.getElementById("goToPreferences").addEventListener("click", () => {
    const username = document.getElementById("usernameInput").value.trim();
    if (!username) {
      alert("Please enter your name before customizing preferences.");
      return;
    }
    window.location.href = `/preferences/${encodeURIComponent(username)}`;
  });

  goBackBtn.addEventListener("click", () => location.reload());
});
