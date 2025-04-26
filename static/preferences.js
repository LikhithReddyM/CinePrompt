// It just checks the input fields and sends the data to the server
// when the user clicks the "Save Preferences" button. It also handles errors and shows success messages.
document.addEventListener("DOMContentLoaded", () => {
    const username = document.getElementById("usernameInput").value;
  
    // Load existing preferences
    fetch('/api/get-preferences')
        .then(res => res.json())
        .then(response => {
            if (response.status === 'success' && response.preferences) {
                const prefs = response.preferences;
                
                // Set genres
                prefs.genres.forEach(genre => {
                    const checkbox = document.querySelector(`input[name="genre"][value="${genre}"]`);
                    if (checkbox) checkbox.checked = true;
                });

                // Set favorite movies
                prefs.favorite_movies.forEach((movie, index) => {
                    const input = document.querySelector(`input[name="favMovie${index + 1}"]`);
                    if (input) input.value = movie;
                });

                // Set worst movies
                prefs.worst_movies.forEach((movie, index) => {
                    const input = document.querySelector(`input[name="worstMovie${index + 1}"]`);
                    if (input) input.value = movie;
                });

                // Set era preference
                const eraInput = document.querySelector(`input[name="era"][value="${prefs.era}"]`);
                if (eraInput) eraInput.checked = true;
            }
        })
        .catch(err => {
            console.error('Error loading preferences:', err);
        });
  
    document.getElementById("savePreferences").addEventListener("click", () => {
        const genres = [...document.querySelectorAll('input[name="genre"]:checked')].map(input => input.value);
        const favMovies = [1, 2, 3, 4, 5].map(i => 
            document.querySelector(`input[name="favMovie${i}"]`)?.value.trim()
        ).filter(Boolean);
        const worstMovies = [1, 2, 3, 4, 5].map(i => 
            document.querySelector(`input[name="worstMovie${i}"]`)?.value.trim()
        ).filter(Boolean);
        const era = document.querySelector('input[name="era"]:checked')?.value || 'new';
  
        const data = {
            genres,
            favorite_movies: favMovies,
            worst_movies: worstMovies,
            era
        };
  
        fetch('/save-preferences', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        })
        .then(res => res.json())
        .then(response => {
            if (response.status === 'success') {
                alert("Preferences saved successfully!");
                // Redirect to home page after successful save
                window.location.href = '/';
            } else {
                alert(response.message || "Failed to save preferences.");
            }
        })
        .catch(err => {
            console.error(err);
            alert("An error occurred while saving preferences.");
        });
    });
  
    // Debounce function to limit API calls
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Search for movies
    const searchMovies = debounce(async (input, dropdown) => {
        const query = input.value.toLowerCase();
        if (query.length < 2) {
            dropdown.style.display = 'none';
            return;
        }

        try {
            const response = await fetch(`/search-movies?q=${encodeURIComponent(query)}`);
            const movies = await response.json();
            
            dropdown.innerHTML = '';
            if (movies.length > 0) {
                movies.forEach(movie => {
                    const div = document.createElement('div');
                    div.className = 'suggestion-item';
                    div.innerHTML = `
                        <span class="title">${movie.title}</span>
                        ${movie.original_title !== movie.title ? 
                            `<span class="original-title">(${movie.original_title})</span>` : ''}
                        <span class="release-date">${movie.release_date || 'Unknown year'}</span>
                    `;
                    div.addEventListener('click', () => {
                        input.value = movie.title;
                        dropdown.style.display = 'none';
                    });
                    dropdown.appendChild(div);
                });
                dropdown.style.display = 'block';
            } else {
                dropdown.style.display = 'none';
            }
        } catch (error) {
            console.error('Search error:', error);
            dropdown.style.display = 'none';
        }
    }, 300);

    // Initialize autocomplete for all movie inputs
    document.querySelectorAll('.movie-input').forEach(input => {
        const container = input.parentElement;
        const dropdown = container.querySelector('.suggestions-dropdown');

        input.addEventListener('input', () => {
            searchMovies(input, dropdown);
        });

        input.addEventListener('blur', () => {
            setTimeout(() => {
                dropdown.style.display = 'none';
            }, 200);
        });

        // Show dropdown when input is focused
        input.addEventListener('focus', () => {
            if (input.value.length >= 2) {
                searchMovies(input, dropdown);
            }
        });
    });
});
  