const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const db = require('./database');

const PORT = 3000;

// Set of active WebSocket connections
const clients = new Set();

const server = http.createServer((req, res) => {
  // CORS Headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // Parse URL
  const parsedUrl = new URL(req.url, `http://${req.headers.host}`);
  const pathname = parsedUrl.pathname;

  // Serve Static Files
  if (pathname === '/' || pathname === '/index.html' || pathname === '/style.css' || pathname === '/client.js') {
    const filename = pathname === '/' ? 'index.html' : pathname.slice(1);
    const filePath = path.join(__dirname, 'public', filename);
    
    let contentType = 'text/html';
    if (filename.endsWith('.css')) contentType = 'text/css';
    if (filename.endsWith('.js')) contentType = 'application/javascript';

    fs.readFile(filePath, (err, content) => {
      if (err) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('File not found');
        return;
      }
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content);
    });
    return;
  }

  // API Endpoints
  
  // GET /api/polls
  if (pathname === '/api/polls' && req.method === 'GET') {
    try {
      const polls = db.getAllPolls();
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(polls));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // GET /api/polls/:id/results
  const resultsMatch = pathname.match(/^\/api\/polls\/([^\/]+)\/results$/);
  if (resultsMatch && req.method === 'GET') {
    const pollId = resultsMatch[1];
    try {
      const results = db.getResults(pollId);
      if (!results) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Poll not found' }));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(results));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // GET /api/polls/:id
  const pollMatch = pathname.match(/^\/api\/polls\/([^\/]+)$/);
  if (pollMatch && req.method === 'GET') {
    const pollId = pollMatch[1];
    try {
      const poll = db.getPoll(pollId);
      if (!poll) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Poll not found' }));
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(poll));
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // POST /api/polls
  if (pathname === '/api/polls' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        if (!data.question || !Array.isArray(data.options) || data.options.length < 2) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Question and at least 2 options are required.' }));
          return;
        }

        // Validate duplicates
        const uniqueOptions = new Set(data.options.map(o => o.trim()).filter(o => o.length > 0));
        if (uniqueOptions.size < 2) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'At least 2 non-empty unique options are required.' }));
          return;
        }

        const expiresAt = data.expires_at ? new Date(data.expires_at).toISOString() : null;
        const pollId = db.createPoll(data.question.trim(), Array.from(uniqueOptions), expiresAt);

        res.writeHead(201, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ poll_id: pollId, message: 'Poll created successfully' }));
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid JSON payload' }));
      }
    });
    return;
  }

  // POST /api/polls/:id/vote
  const voteMatch = pathname.match(/^\/api\/polls\/([^\/]+)\/vote$/);
  if (voteMatch && req.method === 'POST') {
    const pollId = voteMatch[1];
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const data = JSON.parse(body);
        if (!data.option_id) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Option ID is required.' }));
          return;
        }

        // Get voter identity
        // Use client-provided tracking ID or fallback to remote IP
        const voterIdentity = (data.voter_id || req.socket.remoteAddress || 'unknown_ip').trim();

        try {
          db.castVote(pollId, data.option_id, voterIdentity);

          // Get updated results
          const results = db.getResults(pollId);

          // Broadcast results via WS
          broadcastPollResults(pollId, results);

          res.writeHead(201, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(results));
        } catch (err) {
          if (err.message === 'DUPLICATE_VOTE') {
            res.writeHead(409, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: "You have already voted on this poll." }));
          } else {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: err.message }));
          }
        }
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Invalid JSON payload' }));
      }
    });
    return;
  }

  // Catch-all 404
  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not Found');
});

// Broadcast helper for WebSocket rooms
function broadcastPollResults(pollId, results) {
  const payload = JSON.stringify({ event: 'results_updated', data: results });
  for (const client of clients) {
    if (client.currentPollId === pollId) {
      sendWSMessage(client.socket, payload);
    }
  }
}

// WS Helper: Format and send a text frame
function sendWSMessage(socket, message) {
  try {
    const payload = Buffer.from(message, 'utf8');
    const length = payload.length;
    let frame;

    if (length <= 125) {
      frame = Buffer.alloc(2 + length);
      frame[0] = 0x81; // FIN bit + text opcode (1)
      frame[1] = length;
      payload.copy(frame, 2);
    } else if (length <= 65535) {
      frame = Buffer.alloc(4 + length);
      frame[0] = 0x81;
      frame[1] = 126;
      frame.writeUInt16BE(length, 2);
      payload.copy(frame, 4);
    } else {
      frame = Buffer.alloc(10 + length);
      frame[0] = 0x81;
      frame[1] = 127;
      frame.writeBigUInt64BE(BigInt(length), 2);
      payload.copy(frame, 10);
    }
    socket.write(frame);
  } catch (err) {
    console.error('Error writing WS frame:', err);
  }
}

// WebSocket Upgrade Handler
server.on('upgrade', (req, socket, head) => {
  if (req.headers['upgrade'] && req.headers['upgrade'].toLowerCase() === 'websocket') {
    const key = req.headers['sec-websocket-key'];
    if (!key) {
      socket.end('HTTP/1.1 400 Bad Request\r\n\r\n');
      return;
    }

    // Handshake
    const accept = crypto
      .createHash('sha1')
      .update(key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11')
      .digest('base64');

    const headers = [
      'HTTP/1.1 101 Switching Protocols',
      'Upgrade: websocket',
      'Connection: Upgrade',
      `Sec-WebSocket-Accept: ${accept}`
    ];

    socket.write(headers.join('\r\n') + '\r\n\r\n');

    // Connection object
    const connection = {
      socket,
      currentPollId: null
    };

    clients.add(connection);

    // Frame reader
    socket.on('data', (buffer) => {
      try {
        if (buffer.length < 2) return;
        const byte1 = buffer[0];
        const byte2 = buffer[1];
        const opcode = byte1 & 0x0F;
        const isMasked = (byte2 & 0x80) !== 0;
        let payloadLength = byte2 & 0x7F;
        let offset = 2;

        if (payloadLength === 126) {
          if (buffer.length < 4) return;
          payloadLength = buffer.readUInt16BE(2);
          offset = 4;
        } else if (payloadLength === 127) {
          if (buffer.length < 10) return;
          payloadLength = Number(buffer.readBigUInt64BE(2));
          offset = 10;
        }

        // Connection close frame
        if (opcode === 8) {
          socket.end();
          clients.delete(connection);
          return;
        }

        // Text frame
        if (opcode === 1) {
          if (buffer.length < offset + (isMasked ? 4 : 0) + payloadLength) return;
          let payload;
          if (isMasked) {
            const maskKeys = buffer.slice(offset, offset + 4);
            offset += 4;
            const rawData = buffer.slice(offset, offset + payloadLength);
            payload = Buffer.alloc(payloadLength);
            for (let i = 0; i < payloadLength; i++) {
              payload[i] = rawData[i] ^ maskKeys[i % 4];
            }
          } else {
            payload = buffer.slice(offset, offset + payloadLength);
          }

          const messageString = payload.toString('utf8');
          try {
            const msg = JSON.parse(messageString);
            if (msg.action === 'join_poll' && msg.poll_id) {
              connection.currentPollId = msg.poll_id;
            }
          } catch (e) {
            // Ignore malformed text packets
          }
        }
      } catch (err) {
        console.error('Error parsing WS frame:', err);
      }
    });

    socket.on('close', () => {
      clients.delete(connection);
    });

    socket.on('error', () => {
      clients.delete(connection);
      socket.destroy();
    });
  } else {
    socket.end('HTTP/1.1 400 Bad Request\r\n\r\n');
  }
});

server.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
