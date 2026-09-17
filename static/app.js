const $ = (selector) => document.querySelector(selector);

function setText(selector, value) { $(selector).textContent = value; }

async function loadOverview() {
  const response = await fetch('/api/overview');
  const data = await response.json();
  const total = data.total;
  setText('#accuracy', `${data.metrics.accuracy}%`);
  $('#accuracy-meter').style.width = `${data.metrics.accuracy}%`;
  setText('#dataset-total', `${total} STORIES`);
  setText('#real-count', data.labels.REAL);
  setText('#fake-count', data.labels.FAKE);
  $('#real-bar').style.width = `${(data.labels.REAL / total) * 100}%`;
  $('#fake-bar').style.width = `${(data.labels.FAKE / total) * 100}%`;
  setText('#average-length', data.average_length);
  setText('#vocabulary', data.vocabulary.toLocaleString());
  const matrix = data.metrics.confusion_matrix;
  setText('#matrix-00', matrix[0][0]); setText('#matrix-01', matrix[0][1]);
  setText('#matrix-10', matrix[1][0]); setText('#matrix-11', matrix[1][1]);
}

$('#article-text').addEventListener('input', (event) => {
  setText('#char-count', `${event.target.value.length.toLocaleString()} / 12,000`);
});

$('#predict-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = $('#article-text').value.trim();
  const result = $('#prediction-result');
  const button = $('.primary-button');
  button.disabled = true;
  button.querySelector('span').textContent = 'Reading...';
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text}) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    result.className = `result ${data.label === 'FAKE' ? 'fake' : ''}`;
    result.innerHTML = `<span class="result-label">${data.label === 'FAKE' ? 'Likely fake' : 'Likely real'}</span><small>${data.confidence}% confidence<br>FAKE ${data.probabilities.FAKE}% &nbsp; / &nbsp; REAL ${data.probabilities.REAL}%</small>`;
  } catch (error) {
    result.className = 'result fake';
    result.innerHTML = `<span>${error.message}</span>`;
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Run analysis';
  }
});

loadOverview().catch(() => setText('#accuracy', 'N/A'));