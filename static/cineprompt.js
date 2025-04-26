/* static/cineprompt.js
   Front‑end controller for CinePrompt
   – Saves username & prompt
   – Shows the top‑5 movies returned by Flask/OpenAI
*/

document.addEventListener('DOMContentLoaded', function() {
    const movieForm = document.getElementById('movieForm');
    const formSection = document.getElementById('formSection');
    const loadingSection = document.getElementById('loadingSection');
    const successSection = document.getElementById('successSection');
    const recommendationsDiv = document.getElementById('recommendations');
    const newSearchBtn = document.getElementById('newSearch');

    movieForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const userPrompt = document.getElementById('userPrompt').value;
        
        // Show loading section and hide form
        formSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        successSection.classList.add('hidden');
        
        try {
            const response = await fetch('/submit-preferences', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: userPrompt,
                    genres: [],  // These will be filled by the server from user preferences
                    favorite_movies: [],
                    worst_movies: [],
                    era: 'new'
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Hide loading and show success section
                loadingSection.classList.add('hidden');
                successSection.classList.remove('hidden');
                
                // Display recommendations
                recommendationsDiv.innerHTML = ''; // Clear previous recommendations
                const recommendations = data.suggestions; // Now an array of objects
                
                if (recommendations && recommendations.length > 0) {
                    recommendations.forEach(movie => {
                        const row = document.createElement('div');
                        row.className = 'movie-row';
                        
                        // Create image element
                        const posterImg = document.createElement('img');
                        posterImg.className = 'movie-poster';
                        posterImg.src = movie.poster_url || 'static/placeholder.png'; // Use placeholder if no poster
                        posterImg.alt = movie.title ? `Poster for ${movie.title}` : 'Movie Poster';
                        // Add error handling for broken images
                        posterImg.onerror = function() { 
                            this.onerror=null; // prevent infinite loop
                            this.src='static/placeholder.png'; 
                            console.warn(`Failed to load poster for ${movie.title || 'Unknown Movie'}`);
                        };
                        
                        // Create details container
                        const detailsDiv = document.createElement('div');
                        detailsDiv.className = 'movie-details';
                        
                        const titleEl = document.createElement('h3');
                        titleEl.className = 'movie-title';
                        titleEl.textContent = movie.title || 'Unknown Title';
                        
                        const dateEl = document.createElement('p');
                        dateEl.className = 'movie-date';
                        dateEl.textContent = `Released: ${movie.release_date || 'Unknown'}`;
                        
                        const summaryEl = document.createElement('p');
                        summaryEl.className = 'movie-summary';
                        summaryEl.textContent = movie.summary || 'No summary available.';
                        
                        // Append elements
                        detailsDiv.appendChild(titleEl);
                        detailsDiv.appendChild(dateEl);
                        detailsDiv.appendChild(summaryEl);
                        
                        row.appendChild(posterImg);
                        row.appendChild(detailsDiv);
                        
                        recommendationsDiv.appendChild(row);
                    });
                } else {
                     recommendationsDiv.innerHTML = '<p class="no-results">No recommendations found.</p>';
                }
            } else {
                throw new Error(data.message || 'Failed to get recommendations');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while getting recommendations. Please try again.');
            // Reset to form view
            loadingSection.classList.add('hidden');
            formSection.classList.remove('hidden');
        }
    });

    newSearchBtn.addEventListener('click', function() {
        successSection.classList.add('hidden');
        formSection.classList.remove('hidden');
        document.getElementById('userPrompt').value = '';
    });
});
