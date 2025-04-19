/* static/cineprompt.js
   Front‑end controller for CinePrompt
   – Saves username & prompt
   – Shows the top‑5 movies returned by Flask/OpenAI
*/

document.addEventListener("DOMContentLoaded", () => {
  // ----- DOM shortcuts -----
  const formSection     = document.getElementById("formSection");
  const successSection  = document.getElementById("successSection");
  const suggestionsBox  = document.getElementById("suggestionsBox");

  const submitBtn       = document.getElementById("submitPrompt");
  const prefsBtn        = document.getElementById("goToPreferences");
  const goBackBtn       = document.getElementById("goBackBtn");

  // ========== SUBMIT ==========
  submitBtn.addEventListener("click", async () => {      // ← async!
    const username = document.getElementById("usernameInput").value.trim();
    const prompt   = document.getElementById("promptInput").value.trim();
    if (!username) { alert("Please enter your name."); return; }

    try {
      // POST to Flask
      const res = await fetch("/submit-preferences", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, prompt, timestamp: Date.now() })
      });
      const { status, message, suggestions } = await res.json();
      if (status !== "success") throw new Error(message);

      // Insert the top‑5 lines into the success box
      suggestionsBox.innerHTML = suggestions
        .split("\n")                                 // split on newline
        .map(t => t.replace(/^[\-\d.\• ]+/, ""))     // strip bullets/numbers
        .slice(0, 5)                                 // keep max 5
        .map(title => `• ${title}`)                  // re‑bullet uniformly
        .join("<br>");

      // Swap views
      formSection.classList.add("hidden");
      successSection.classList.remove("hidden");

    } catch (err) {
      console.error("Submission error:", err);
      alert("Error submitting data – see console.");
    }
  });

  // ========== GO TO PREFERENCES ==========
  prefsBtn.addEventListener("click", () => {
    const username = document.getElementById("usernameInput").value.trim();
    if (!username) { alert("Please enter your name first."); return; }
    window.location.href = `/preferences/${encodeURIComponent(username)}`;
  });

  // ========== GO BACK ==========
  goBackBtn.addEventListener("click", () => location.reload());
});
