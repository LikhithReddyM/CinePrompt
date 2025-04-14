// It just checks the input fields and sends the data to the server
// when the user clicks the "Save Preferences" button. It also handles errors and shows success messages.
document.addEventListener("DOMContentLoaded", () => {
    const username = document.getElementById("usernameInput").value;

    const getMoviesWithRatings = (type) => {
      return Array.from({ length: 5 }, (_, i) => {
        const movieName = document.querySelector(`input[name="${type}${i + 1}"]`)?.value.trim();
        const rating = document.querySelector(`input[name="${type}${i + 1}Rating"]:checked`)?.value;
  
        if (movieName) {
          return {
            title: movieName,
            rating: rating ? parseInt(rating) : null
          };
        }
  
        return null;
      }).filter(Boolean);
    };
  
    document.getElementById("savePreferences").addEventListener("click", () => {
      const genres = [...document.querySelectorAll('input[name="genre"]:checked')].map(input => input.value);
      const favMovies = getMoviesWithRatings("favMovie");     // updated line
      const worstMovies = getMoviesWithRatings("worstMovie"); // updated line
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
  