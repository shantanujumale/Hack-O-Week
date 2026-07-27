// State
let polls = [];
let activePollId = null;
let ws = null;
let voterId = localStorage.getItem('voter_id');

// Setup Voter ID
if (!voterId) {
  voterId = 'voter_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  localStorage.setItem('voter_id', voterId);
}

// DOM Elements
const createPollForm = document.getElementById('create-poll-form');
const addOptionBtn = document.getElementById('add-option-btn');
const optionsInputsContainer = document.getElementById('options-inputs-container');
const pollsList = document.getElementById('polls-list');
const activePollCard = document.getElementById('active-poll-card');
const noPollSelected = document.getElementById('no-poll-selected');
const activeQuestion = document.getElementById('active-question');
const pollIdDisplay = document.getElementById('poll-id-display');
const pollStatusBadge = document.getElementById('poll-status-badge');
const pollExpirationDisplay = document.getElementById('poll-expiration-display');
const votingOptionsContainer = document.getElementById('voting-options-container');
const totalVotesCount = document.getElementById('total-votes-count');
const resultsBarsContainer = document.getElementById('results-bars-container');

// Simulation DOM
const simRaceConditionBtn = document.getElementById('sim-race-condition');
const simDuplicateCheckBtn = document.getElementById('sim-duplicate-check');
const simLog = document.getElementById('sim-log');
const simLogText = document.getElementById('sim-log-text');

// Init
document.addEventListener('DOMContentLoaded', () => {
  loadPolls();
  setupWebSocket();
  setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
  // Add Dynamic Options Inputs
  addOptionBtn.addEventListener('click', () => {
    const row = document.createElement('div');
    row.className = 'option-input-row';
    const optCount = optionsInputsContainer.querySelectorAll('.option-input').length + 1;
    row.innerHTML = `
      <input type="text" class="option-input" placeholder="Option ${optCount}" required>
      <button type="button" class="btn-secondary remove-opt-btn" style="margin-top:0">×</button>
    `;
    optionsInputsContainer.appendChild(row);

    // Remove Option Event
    row.querySelector('.remove-opt-btn').addEventListener('click', () => {
      row.remove();
      // Re-index placeholders
      optionsInputsContainer.querySelectorAll('.option-input').forEach((input, idx) => {
        input.placeholder = `Option ${idx + 1}`;
      });
    });
  });

  // Handle Poll Creation Submit
  createPollForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = document.getElementById('poll-question').value.trim();
    const optionInputs = Array.from(document.querySelectorAll('.option-input'));
    const options = optionInputs.map(input => input.value.trim()).filter(val => val.length > 0);
    const expiresAt = document.getElementById('poll-expiration').value;

    try {
      const response = await fetch('/api/polls', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          options,
          expires_at: expiresAt ? new Date(expiresAt).toISOString() : null
        })
      });

      const resData = await response.json();
      if (!response.ok) {
        alert(resData.error || 'Failed to create poll');
        return;
      }

      // Reset form
      createPollForm.reset();
      optionsInputsContainer.innerHTML = `
        <div class="option-input-row"><input type="text" class="option-input" placeholder="Option 1" required></div>
        <div class="option-input-row"><input type="text" class="option-input" placeholder="Option 2" required></div>
      `;

      // Reload lists and select the new poll
      await loadPolls();
      selectPoll(resData.poll_id);

    } catch (err) {
      console.error('Error creating poll:', err);
      alert('Error creating poll. Check console.');
    }
  });

  // Simulation Triggers
  simRaceConditionBtn.addEventListener('click', runRaceConditionSimulation);
  simDuplicateCheckBtn.addEventListener('click', runDuplicateSimulation);
}

// Load List of Polls
async function loadPolls() {
  try {
    const res = await fetch('/api/polls');
    polls = await res.json();

    if (polls.length === 0) {
      pollsList.innerHTML = '<div class="loading-spinner">No polls launched yet. Create one!</div>';
      return;
    }

    pollsList.innerHTML = '';
    polls.forEach(poll => {
      const isExpired = poll.expires_at && new Date(poll.expires_at) < new Date();
      const statusBadge = isExpired 
        ? '<span class="badge expired-badge">Expired</span>' 
        : '<span class="badge active-badge">Active</span>';

      const item = document.createElement('div');
      item.className = `poll-item ${poll.id === activePollId ? 'selected' : ''}`;
      item.innerHTML = `
        <div class="poll-item-question">${escapeHTML(poll.question)}</div>
        <div class="poll-item-meta">
          <span>${poll.options.length} options</span>
          ${statusBadge}
        </div>
      `;
      item.addEventListener('click', () => selectPoll(poll.id));
      pollsList.appendChild(item);
    });
  } catch (err) {
    console.error('Error loading polls:', err);
    pollsList.innerHTML = '<div class="loading-spinner" style="color:var(--danger)">Error loading polls</div>';
  }
}

// Select Active Poll
async function selectPoll(pollId) {
  activePollId = pollId;

  // Highlight in sidebar list
  document.querySelectorAll('.poll-item').forEach((item, idx) => {
    const poll = polls[idx];
    if (poll && poll.id === pollId) {
      item.classList.add('selected');
    } else {
      item.classList.remove('selected');
    }
  });

  try {
    const res = await fetch(`/api/polls/${pollId}`);
    if (!res.ok) throw new Error('Poll not found');
    const poll = await res.json();

    noPollSelected.classList.add('hidden');
    activePollCard.classList.remove('hidden');

    // Update Headers
    activeQuestion.textContent = poll.question;
    pollIdDisplay.textContent = `ID: ${poll.id}`;

    const isExpired = poll.expires_at && new Date(poll.expires_at) < new Date();
    if (isExpired) {
      pollStatusBadge.textContent = 'Expired';
      pollStatusBadge.className = 'badge expired-badge';
      pollExpirationDisplay.textContent = `Poll closed on ${new Date(poll.expires_at).toLocaleString()}`;
    } else {
      pollStatusBadge.textContent = 'Active';
      pollStatusBadge.className = 'badge active-badge';
      pollExpirationDisplay.textContent = poll.expires_at 
        ? `Closes on ${new Date(poll.expires_at).toLocaleString()}`
        : 'Open run (no expiration)';
    }

    // Build Voting Section
    buildVotingButtons(poll, isExpired);

    // Initial results load
    const resultsRes = await fetch(`/api/polls/${pollId}/results`);
    const initialResults = await resultsRes.json();
    renderResults(initialResults);

    // Join room on websocket
    joinWebSocketRoom(pollId);

    // Reset Sim Log
    simLog.classList.add('hidden');
    simLogText.textContent = '';

  } catch (err) {
    console.error('Error selecting poll:', err);
  }
}

// Build Voting Options UI
function buildVotingButtons(poll, isExpired) {
  votingOptionsContainer.innerHTML = '';
  
  // Check if this user has already voted locally
  const alreadyVotedOption = localStorage.getItem(`voted_poll_${poll.id}`);

  poll.options.forEach(opt => {
    const btn = document.createElement('button');
    btn.className = `vote-option-btn ${alreadyVotedOption === opt.id ? 'voted' : ''}`;
    btn.disabled = isExpired || !!alreadyVotedOption;
    btn.innerHTML = `
      <span>${escapeHTML(opt.text)}</span>
      <span class="vote-check-icon">✓</span>
    `;

    if (!isExpired && !alreadyVotedOption) {
      btn.addEventListener('click', () => castVote(poll.id, opt.id));
    }

    votingOptionsContainer.appendChild(btn);
  });
}

// Cast Vote
async function castVote(pollId, optionId) {
  try {
    const res = await fetch(`/api/polls/${pollId}/vote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        option_id: optionId,
        voter_id: voterId
      })
    });

    const data = await res.json();
    if (!res.ok) {
      alert(data.error || 'Failed to cast vote');
      return;
    }

    // Cache vote locally
    localStorage.setItem(`voted_poll_${pollId}`, optionId);

    // Reload active poll to reflect voted state on buttons
    selectPoll(pollId);

  } catch (err) {
    console.error('Error casting vote:', err);
  }
}

// Render Results Live Chart
function renderResults(results) {
  totalVotesCount.textContent = `${results.total_votes} votes`;
  resultsBarsContainer.innerHTML = '';

  // Find max votes to highlight the leader/winner
  let maxVotes = 0;
  if (results.total_votes > 0) {
    maxVotes = Math.max(...results.options.map(o => o.votes));
  }

  results.options.forEach(opt => {
    const isWinner = results.total_votes > 0 && opt.votes === maxVotes;
    const barRow = document.createElement('div');
    barRow.className = `result-bar-item ${isWinner ? 'winner' : ''}`;
    barRow.innerHTML = `
      <div class="result-bar-info">
        <span class="result-bar-text">${escapeHTML(opt.text)}</span>
        <span class="result-bar-stats">${opt.votes} (${opt.percentage}%)</span>
      </div>
      <div class="result-bar-track">
        <div class="result-bar-fill" style="width: ${opt.percentage}%"></div>
      </div>
    `;
    resultsBarsContainer.appendChild(barRow);
  });
}

// WebSockets Networking
function setupWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}`;
  
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('WebSocket connected');
    // Re-join active room if connection dropped and reconnected
    if (activePollId) {
      joinWebSocketRoom(activePollId);
    }
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.event === 'results_updated' && msg.data.poll_id === activePollId) {
        renderResults(msg.data);
        // Refresh sidebar total/states
        loadPolls();
      }
    } catch (err) {
      console.error('Error parsing WS message:', err);
    }
  };

  ws.onclose = () => {
    console.log('WebSocket disconnected. Reconnecting in 3s...');
    setTimeout(setupWebSocket, 3000);
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
    ws.close();
  };
}

function joinWebSocketRoom(pollId) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      action: 'join_poll',
      poll_id: pollId
    }));
  }
}

// SIMULATION: Race Condition Voting (100 Simultaneous Requests)
async function runRaceConditionSimulation() {
  if (!activePollId) return;
  
  // Pick first option to spam votes on
  const res = await fetch(`/api/polls/${activePollId}`);
  const poll = await res.json();
  if (poll.options.length === 0) return;
  const targetOption = poll.options[0];

  simLog.classList.remove('hidden');
  simLogText.textContent = `Initializing Race Condition Test...\n`;
  simLogText.textContent += `Targeting: Option "${targetOption.text}" (${targetOption.id})\n`;
  simLogText.textContent += `Spawning 100 concurrent HTTP POST vote requests with unique voter IDs...\n`;

  const totalRequests = 100;
  const promises = [];

  const startTime = performance.now();

  for (let i = 0; i < totalRequests; i++) {
    const simVoterId = `sim_voter_${activePollId}_${Date.now()}_${i}_${Math.random().toString(36).substring(2, 6)}`;
    
    // Create an API call promise
    const promise = fetch(`/api/polls/${activePollId}/vote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        option_id: targetOption.id,
        voter_id: simVoterId
      })
    })
    .then(async r => {
      const data = await r.json();
      return { status: r.status, ok: r.ok, data };
    })
    .catch(err => ({ status: 500, ok: false, error: err.message }));

    promises.push(promise);
  }

  simLogText.textContent += `Sent 100 requests in parallel! Waiting for execution...\n`;
  
  const results = await Promise.all(promises);
  const duration = (performance.now() - startTime).toFixed(1);

  const successes = results.filter(r => r.ok && r.status === 201).length;
  const failures = results.length - successes;

  simLogText.textContent += `Finished 100 operations in ${duration}ms!\n`;
  simLogText.textContent += `Successful database inserts: ${successes}/100\n`;
  simLogText.textContent += `Failed operations: ${failures}/100\n`;
  simLogText.textContent += `Verifying database integrity...\n`;

  // Fetch final results from database
  const resultsRes = await fetch(`/api/polls/${activePollId}/results`);
  const finalResults = await resultsRes.json();
  const optionFinal = finalResults.options.find(o => o.id === targetOption.id);

  simLogText.textContent += `\nVerification Report:\n`;
  simLogText.textContent += `- Original count: ${targetOption.vote_count || 0}\n`;
  simLogText.textContent += `- Expected increase: +${successes}\n`;
  simLogText.textContent += `- Current DB count: ${optionFinal.votes}\n`;

  const matches = (targetOption.vote_count || 0) + successes === optionFinal.votes;
  if (matches) {
    simLogText.textContent += `✅ INTEGRITY VERIFIED: Database vote count is mathematically exact. Atomic counter increments succeeded with 0 race condition updates lost!\n`;
  } else {
    simLogText.textContent += `❌ INTEGRITY ERROR: Race condition caused data loss or counter desynchronization.\n`;
  }
}

// SIMULATION: Unique Constraints Double Vote Rejection
async function runDuplicateSimulation() {
  if (!activePollId) return;

  const res = await fetch(`/api/polls/${activePollId}`);
  const poll = await res.json();
  if (poll.options.length === 0) return;
  const targetOption = poll.options[0];

  simLog.classList.remove('hidden');
  simLogText.textContent = `Initializing Database Unique Constraint Test...\n`;
  
  const uniqueSimVoterId = `dup_test_voter_${activePollId}_${Date.now()}`;
  simLogText.textContent += `Voter ID: ${uniqueSimVoterId}\n\n`;

  // Vote 1
  simLogText.textContent += `1. Casting first vote... `;
  try {
    const res1 = await fetch(`/api/polls/${activePollId}/vote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        option_id: targetOption.id,
        voter_id: uniqueSimVoterId
      })
    });
    
    simLogText.textContent += `Status ${res1.status} (Expected: 201)\n`;
    if (!res1.ok) {
      simLogText.textContent += `❌ Setup failed: first vote was rejected.\n`;
      return;
    }
  } catch (err) {
    simLogText.textContent += `Error: ${err.message}\n`;
    return;
  }

  // Vote 2 (Duplicate)
  simLogText.textContent += `2. Casting duplicate vote (identical Voter ID)... `;
  try {
    const res2 = await fetch(`/api/polls/${activePollId}/vote`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        option_id: targetOption.id,
        voter_id: uniqueSimVoterId
      })
    });

    const data2 = await res2.json();
    simLogText.textContent += `Status ${res2.status} (Expected: 409 Conflict)\n`;
    simLogText.textContent += `Response message: "${data2.error}"\n`;

    if (res2.status === 409) {
      simLogText.textContent += `✅ DUPLICATE BLOCKED: Database UNIQUE(poll_id, voter_identity) constraint successfully rejected the duplicate vote and rolled back the transaction!\n`;
    } else {
      simLogText.textContent += `❌ CONSTRAINT FAIL: Database failed to prevent duplicate voting.\n`;
    }
  } catch (err) {
    simLogText.textContent += `Error: ${err.message}\n`;
  }
}

// Helpers
function escapeHTML(str) {
  return str.replace(/[&<>'"]/g, 
    tag => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      "'": '&#39;',
      '"': '&quot;'
    }[tag] || tag)
  );
}
