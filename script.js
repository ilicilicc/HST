document.getElementById('analyze').addEventListener('click', () => {
    const filepath = document.getElementById('filepath').value;
    const resultsDiv = document.getElementById('results');

    fetch('/api/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filepath }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultsDiv.innerHTML = `<p>Error: ${data.error}</p>`;
            return;
        }

        let html = `
            <h2>Analysis of ${filepath}</h2>
            <h3>Explanation</h3>
            <p>${data.explanation}</p>
            <h3>Suggestions</h3>
            <ul>
        `;
        data.suggestions.forEach(suggestion => {
            html += `<li>${suggestion}</li>`;
        });
        html += '</ul><h3>New Inventions</h3><ul>';
        data.new_inventions.forEach(invention => {
            html += `<li>${invention}</li>`;
        });
        html += '</ul>';
        resultsDiv.innerHTML = html;
    });
});
