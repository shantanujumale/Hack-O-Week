const { DatabaseSync } = require('node:sqlite');
const crypto = require('crypto');
const path = require('path');

const dbPath = path.join(__dirname, 'polls.db');
const db = new DatabaseSync(dbPath);

// Enable foreign key constraints
db.exec('PRAGMA foreign_keys = ON;');

// Initialize tables
db.exec(`
  CREATE TABLE IF NOT EXISTS polls (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,
    is_active INTEGER DEFAULT 1
  );

  CREATE TABLE IF NOT EXISTS options (
    id TEXT PRIMARY KEY,
    poll_id TEXT NOT NULL,
    text TEXT NOT NULL,
    vote_count INTEGER DEFAULT 0,
    FOREIGN KEY (poll_id) REFERENCES polls(id) ON DELETE CASCADE
  );

  CREATE TABLE IF NOT EXISTS votes (
    id TEXT PRIMARY KEY,
    poll_id TEXT NOT NULL,
    option_id TEXT NOT NULL,
    voter_identity TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (poll_id) REFERENCES polls(id) ON DELETE CASCADE,
    FOREIGN KEY (option_id) REFERENCES options(id) ON DELETE CASCADE,
    UNIQUE(poll_id, voter_identity)
  );
`);

// Prepared statements for performance and safety
const insertPollStmt = db.prepare('INSERT INTO polls (id, question, expires_at) VALUES (?, ?, ?)');
const insertOptionStmt = db.prepare('INSERT INTO options (id, poll_id, text) VALUES (?, ?, ?)');
const insertVoteStmt = db.prepare('INSERT INTO votes (id, poll_id, option_id, voter_identity) VALUES (?, ?, ?, ?)');
const incrementOptionVoteStmt = db.prepare('UPDATE options SET vote_count = vote_count + 1 WHERE id = ? AND poll_id = ?');

const selectPollStmt = db.prepare('SELECT * FROM polls WHERE id = ?');
const selectOptionsStmt = db.prepare('SELECT * FROM options WHERE poll_id = ?');
const checkOptionExistsStmt = db.prepare('SELECT 1 FROM options WHERE id = ? AND poll_id = ?');

function createPoll(question, optionsList, expiresAt = null) {
  const pollId = 'poll_' + crypto.randomBytes(8).toString('hex');
  
  // Use transaction to ensure poll and all options are inserted atomically
  db.exec('BEGIN TRANSACTION;');
  try {
    insertPollStmt.run(pollId, question, expiresAt);
    for (const optText of optionsList) {
      const optId = 'opt_' + crypto.randomBytes(8).toString('hex');
      insertOptionStmt.run(optId, pollId, optText);
    }
    db.exec('COMMIT;');
    return pollId;
  } catch (err) {
    db.exec('ROLLBACK;');
    throw err;
  }
}

function getPoll(pollId) {
  const poll = selectPollStmt.all(pollId)[0];
  if (!poll) return null;
  const options = selectOptionsStmt.all(pollId);
  return { ...poll, options };
}

function getAllPolls() {
  const polls = db.prepare('SELECT * FROM polls ORDER BY created_at DESC').all();
  return polls.map(poll => {
    const options = selectOptionsStmt.all(poll.id);
    return { ...poll, options };
  });
}

function castVote(pollId, optionId, voterIdentity) {
  // Check if poll exists and is active
  const poll = selectPollStmt.all(pollId)[0];
  if (!poll) {
    throw new Error('Poll not found');
  }
  
  // Check expiration
  if (poll.expires_at && new Date(poll.expires_at) < new Date()) {
    throw new Error('Poll has expired');
  }
  if (!poll.is_active) {
    throw new Error('Poll is inactive');
  }

  // Check if option belongs to the poll
  const optionValid = checkOptionExistsStmt.all(optionId, pollId)[0];
  if (!optionValid) {
    throw new Error('Option does not belong to this poll');
  }

  // Cast vote and increment vote count inside a transaction
  db.exec('BEGIN TRANSACTION;');
  try {
    const voteId = 'vote_' + crypto.randomBytes(8).toString('hex');
    insertVoteStmt.run(voteId, pollId, optionId, voterIdentity);
    incrementOptionVoteStmt.run(optionId, pollId);
    db.exec('COMMIT;');
  } catch (err) {
    db.exec('ROLLBACK;');
    // Check for unique constraint violation
    if (err.message && err.message.includes('UNIQUE constraint failed')) {
      throw new Error('DUPLICATE_VOTE');
    }
    throw err;
  }
}

function getResults(pollId) {
  const poll = selectPollStmt.all(pollId)[0];
  if (!poll) return null;

  const options = selectOptionsStmt.all(pollId);
  const totalVotes = options.reduce((sum, opt) => sum + opt.vote_count, 0);

  const formattedOptions = options.map(opt => {
    const percentage = totalVotes > 0 ? parseFloat(((opt.vote_count / totalVotes) * 100).toFixed(1)) : 0;
    return {
      id: opt.id,
      text: opt.text,
      votes: opt.vote_count,
      percentage
    };
  });

  return {
    poll_id: poll.id,
    question: poll.question,
    expires_at: poll.expires_at,
    is_active: poll.is_active,
    total_votes: totalVotes,
    options: formattedOptions
  };
}

module.exports = {
  createPoll,
  getPoll,
  getAllPolls,
  castVote,
  getResults
};
