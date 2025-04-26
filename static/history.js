document.addEventListener('DOMContentLoaded', function() {
    // Get the history list container
    const historyList = document.getElementById('historyList');
    
    // Fetch user's search history from the server
    fetch('/api/history')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(history => {
            if (!history || history.length === 0) {
                historyList.innerHTML = '<p class="no-history">No search history available</p>';
                return;
            }

            // Create history items for each search
            history.forEach(item => {
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                
                const prompt = document.createElement('div');
                prompt.className = 'history-prompt';
                prompt.textContent = item.prompt;
                
                const timestamp = document.createElement('div');
                timestamp.className = 'history-timestamp';
                timestamp.textContent = new Date(item.timestamp).toLocaleString();
                
                const suggestions = document.createElement('div');
                suggestions.className = 'history-suggestions';
                
                // Use the pre-processed list of titles from the backend
                const movieTitles = item.suggestions_titles || []; 
                if (movieTitles.length > 0) {
                    movieTitles.forEach(title => {
                        const movieItem = document.createElement('div');
                        movieItem.className = 'history-movie-item';
                        movieItem.textContent = title; // Display only the title
                        suggestions.appendChild(movieItem);
                    });
                } else {
                    const noMovies = document.createElement('div');
                    noMovies.className = 'history-movie-item';
                    noMovies.textContent = 'No specific movie titles found for this entry.';
                    suggestions.appendChild(noMovies);
                }
                
                historyItem.appendChild(prompt);
                historyItem.appendChild(timestamp);
                historyItem.appendChild(suggestions);
                
                // Add click event to toggle movie details
                historyItem.addEventListener('click', function() {
                    this.classList.toggle('expanded');
                });
                
                historyList.appendChild(historyItem);
            });
        })
        .catch(error => {
            console.error('Error fetching history:', error);
            historyList.innerHTML = '<p class="error">Error loading history. Please try again later.</p>';
        });
}); 