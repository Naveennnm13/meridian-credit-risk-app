const form = document.getElementById('applicant-form');
const submitBtn = form.querySelector('.submit-btn');
const btnLabel = submitBtn.querySelector('.btn-label');
const btnSpinner = submitBtn.querySelector('.btn-spinner');

const placeholder = document.getElementById('placeholder');
const resultsEl = document.getElementById('results');
const decisionBanner = document.getElementById('decision-banner');
const decisionText = document.getElementById('decision-text');
const decisionSub = document.getElementById('decision-sub');
const probabilityValue = document.getElementById('probability-value');
const probabilityFill = document.getElementById('probability-fill');
const confidenceValue = document.getElementById('confidence-value');
const explanationText = document.getElementById('explanation-text');
const formError = document.getElementById('form-error');

let shapChart = null;

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  formError.hidden = true;
  btnLabel.textContent = 'Analyzing...';
  btnSpinner.hidden = false;
  submitBtn.disabled = true;

  const formData = new FormData(form);
  const payload = {};
  formData.forEach((value, key) => { payload[key] = value; });

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Something went wrong.');
    }

    renderResults(data);
  } catch (err) {
    formError.textContent = 'Error: ' + err.message;
    formError.hidden = false;
  } finally {
    btnLabel.textContent = 'Get Decision';
    btnSpinner.hidden = true;
    submitBtn.disabled = false;
  }
});

function renderResults(data) {
  placeholder.hidden = true;
  resultsEl.hidden = false;

  const isGood = data.decision === 'Good Credit';
  decisionBanner.classList.remove('good', 'bad');
  decisionBanner.classList.add(isGood ? 'good' : 'bad');
  decisionText.textContent = data.decision;
  decisionSub.textContent = isGood
    ? 'This applicant is assessed as a favorable credit risk.'
    : 'This applicant is assessed as an elevated credit risk.';

  confidenceValue.textContent = data.confidence;
  confidenceValue.style.color = data.is_borderline ? 'var(--amber)' : 'var(--blue)';

  // Animate probability count-up
  animateNumber(probabilityValue, data.probability);
  requestAnimationFrame(() => {
    probabilityFill.style.width = data.probability + '%';
  });

  explanationText.innerHTML = formatExplanation(data.explanation);

  renderChart(data.factors);

  resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function animateNumber(el, target) {
  let current = 0;
  const duration = 800;
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    current = progress * target;
    el.textContent = current.toFixed(1) + '%';
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function formatExplanation(text) {
  return text
    .split('\n')
    .filter(line => line.trim().length > 0)
    .map(line => {
      const [labelPart, ...rest] = line.split(':');
      if (rest.length === 0) return `<p>${line}</p>`;
      return `<p><strong>${labelPart.trim()}:</strong>${rest.join(':')}</p>`;
    })
    .join('');
}

function renderChart(factors) {
  const ctx = document.getElementById('shap-chart').getContext('2d');
  const labels = factors.map(f => f.feature).reverse();
  const values = factors.map(f => f.impact).reverse();
  const colors = values.map(v => v < 0 ? '#c0392b' : '#1a9e5c');

  if (shapChart) shapChart.destroy();

  shapChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
        barThickness: 18
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { color: '#eef1f5' },
          title: { display: true, text: 'Impact on approval likelihood', color: '#6b7686' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#1c232e', font: { size: 12 } }
        }
      }
    }
  });
}
