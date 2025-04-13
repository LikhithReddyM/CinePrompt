// It just checks the input fields and sends the data to the server
// when the user clicks the "Save Preferences" button. It also handles errors and shows success messages.
document.addEventListener("DOMContentLoaded", () => {
    const username = document.getElementById("usernameInput").value;
  
    document.getElementById("savePreferences").addEventListener("click", () => {
      const genres = [...document.querySelectorAll('input[name="genre"]:checked')].map(input => input.value);
      const favMovies = [1, 2, 3].map(i => document.querySelector(`input[name="favMovie${i}\"]`)?.value.trim()).filter(Boolean);
      const worstMovies = [1, 2, 3].map(i => document.querySelector(`input[name="worstMovie${i}\"]`)?.value.trim()).filter(Boolean);
      const era = document.querySelector('input[name="era"]:checked')?.value || '';
  
      const data = {
        username,
        timestamp: new Date().toISOString()
      };
      if (genres.length) data.genres = genres;
      if (favMovies.length) data.favMovies = favMovies;
      if (worstMovies.length) data.worstMovies = worstMovies;
      if (era) data.era = era;
  
      fetch('/submit-preferences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      })
        .then(res => res.json())
        .then(response => {
          alert("Preferences saved successfully!");
        })
        .catch(err => {
          alert("Failed to save preferences.");
          console.error(err);
        });
    });
  });
  