/**
 * SENTINEL — Operator Dashboard Client Application
 * 
 * Implements WCAG 2.1 AA accessibility guidelines, session token management,
 * SSE streaming consumption, and panel status managers.
 */

// ── Application State ────────────────────────────────────────────────────────
const state = {
    sessionToken: null,
    sessionId: `session_${Math.random().toString(36).substring(2, 11)}`,
    activeTab: 'tab-audit',
    ingestPoller: null,
    statusPoller: null
};

// ── DOM References ───────────────────────────────────────────────────────────
const DOM = {
    queryForm: document.getElementById('query-form'),
    queryInput: document.getElementById('query-input'),
    btnSubmit: document.getElementById('btn-submit'),
    btnCancel: document.getElementById('btn-cancel'),
    btnSubmitText: document.querySelector('#btn-submit .btn-text'),
    btnSubmitLoader: document.querySelector('#btn-submit .loader'),
    responseBox: document.getElementById('response-box'),
    stepTimeline: document.getElementById('step-timeline'),
    crisisBanner: document.getElementById('crisis-alert-banner'),
    
    // Metrics
    confidenceBadge: document.getElementById('metric-confidence'),
    confidenceVal: document.getElementById('confidence-val'),
    faithfulnessBadge: document.getElementById('metric-faithfulness'),
    faithfulnessVal: document.getElementById('faithfulness-val'),
    conditionBadge: document.getElementById('metric-condition'),
    conditionVal: document.getElementById('condition-val'),
    
    // Lists
    citationsList: document.getElementById('citations-list'),
    clinicalAlertsContainer: document.getElementById('clinical-alerts-container'),
    clinicalAlertsList: document.getElementById('clinical-alerts-list'),
    sessionHistory: document.getElementById('session-history'),
    
    // System Status
    statusOllama: document.querySelector('#status-ollama .val'),
    statusLanceDB: document.querySelector('#status-lancedb .val'),
    statusAudit: document.querySelector('#status-audit .val'),
    statusEscalations: document.getElementById('escalation-val'),
    statusEscalationsDot: document.getElementById('escalation-dot'),
    
    // Tabs & Panels
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabPanels: document.querySelectorAll('.tab-panel'),
    auditTableBody: document.getElementById('audit-table-body'),
    escalationsTableBody: document.getElementById('escalations-table-body'),
    corpusTableBody: document.getElementById('corpus-table-body'),
    
    // Actions
    btnClearSession: document.getElementById('btn-clear-session'),
    btnRotateToken: document.getElementById('btn-rotate-token'),
    btnVerifyAudit: document.getElementById('btn-verify-audit'),
    btnExportAudit: document.getElementById('btn-export-audit'),
    
    // Ingest
    pdfDropZone: document.getElementById('pdf-drop-zone'),
    pdfFileInput: document.getElementById('pdf-file-input'),
    ingestProgressContainer: document.getElementById('ingest-progress-container'),
    ingestFilename: document.getElementById('ingest-filename'),
    ingestPercentage: document.getElementById('ingest-percentage'),
    ingestBarFill: document.getElementById('ingest-bar-fill'),
    
    // Modal
    resolveModal: document.getElementById('resolve-modal'),
    resolveForm: document.getElementById('resolve-form'),
    resolveEscalationId: document.getElementById('resolve-escalation-id'),
    resolutionText: document.getElementById('resolution-text'),
    btnCloseModal: document.getElementById('btn-close-modal')
};

// ── Authentication Management ───────────────────────────────────────────────
async function initializeAuthToken() {
    // Check sessionStorage first to isolate token to tab (Finding #21)
    let storedToken = sessionStorage.getItem('sentinel_session_token');
    if (!storedToken) {
        try {
            // First load: request token from server
            const res = await fetch('/auth/token');
            const data = await res.json();
            storedToken = data.token;
            sessionStorage.setItem('sentinel_session_token', storedToken);
        } catch (e) {
            console.error("Auth token initialization failed:", e);
        }
    }
    state.sessionToken = storedToken;
}

function rotateToken() {
    sessionStorage.removeItem('sentinel_session_token');
    window.location.reload();
}

// Helper: Authenticated fetch wrapper
async function authFetch(url, options = {}) {
    options.headers = options.headers || {};
    if (state.sessionToken) {
        options.headers['X-Session-Token'] = state.sessionToken;
    }
    const response = await fetch(url, options);
    if (response.status === 401) {
        console.warn("Session token invalid or expired. Reloading to obtain a new token.");
        sessionStorage.removeItem('sentinel_session_token');
        window.location.reload();
    }
    return response;
}

// ── Tab Management ───────────────────────────────────────────────────────────
function setupTabs() {
    DOM.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const panelId = btn.getAttribute('aria-controls');
            
            // Update Active button state
            DOM.tabBtns.forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
            
            // Update panels visibility
            DOM.tabPanels.forEach(panel => {
                if (panel.id === panelId) {
                    panel.classList.add('active');
                    panel.removeAttribute('hidden');
                } else {
                    panel.classList.remove('active');
                    panel.setAttribute('hidden', 'true');
                }
            });
            
            state.activeTab = btn.id;
            refreshTabData();
        });
    });
}

function refreshTabData() {
    if (state.activeTab === 'tab-audit') {
        loadSessionAuditTable();
    } else if (state.activeTab === 'tab-escalations') {
        loadEscalations();
    } else if (state.activeTab === 'tab-corpus') {
        loadCorpus();
    }
}

// ── Status Polling ───────────────────────────────────────────────────────────
async function pollSystemStatus() {
    try {
        const res = await authFetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();
        
        DOM.statusOllama.textContent = data.ollama;
        DOM.statusLanceDB.textContent = data.index_rows;
        DOM.statusEscalations.textContent = data.escalations_pending;
        
        // Dynamic colors for status bar
        if (data.ollama === 'HEALTHY') {
            document.querySelector('#status-ollama .status-dot').className = 'status-dot green';
        } else {
            document.querySelector('#status-ollama .status-dot').className = 'status-dot red';
        }

        if (data.escalations_pending > 0) {
            DOM.statusEscalationsDot.className = 'status-dot red';
        } else {
            DOM.statusEscalationsDot.className = 'status-dot green';
        }
    } catch (e) {
        console.error("System status poll failed:", e);
    }
}

// ── Ingestion & Drag and Drop ────────────────────────────────────────────────
function setupDragAndDrop() {
    const zone = DOM.pdfDropZone;
    
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });
    
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type === 'application/pdf') {
            uploadPDF(files[0]);
        }
    });
    
    zone.addEventListener('click', () => {
        DOM.pdfFileInput.click();
    });
    
    DOM.pdfFileInput.addEventListener('change', () => {
        const files = DOM.pdfFileInput.files;
        if (files.length > 0) {
            uploadPDF(files[0]);
        }
    });
}

async function uploadPDF(file) {
    const formData = new FormData();
    formData.append('file', file);

    DOM.ingestProgressContainer.classList.remove('hidden');
    DOM.ingestFilename.textContent = `Uploading ${file.name}...`;
    DOM.ingestPercentage.textContent = '0%';
    DOM.ingestBarFill.style.width = '0%';

    try {
        const res = await authFetch('/api/ingest', {
            method: 'POST',
            body: formData
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
        }
        
        // Start polling ingestion status
        if (state.ingestPoller) clearInterval(state.ingestPoller);
        state.ingestPoller = setInterval(pollIngestStatus, 1000);
        
    } catch (e) {
        DOM.ingestFilename.textContent = `Error: ${e.message}`;
        DOM.ingestBarFill.style.backgroundColor = 'var(--color-red)';
    }
}

async function pollIngestStatus() {
    try {
        const res = await authFetch('/api/ingest/status');
        if (!res.ok) return;
        const data = await res.json();
        
        if (data.status === 'IDLE') return;
        
        DOM.ingestFilename.textContent = `Processing: ${data.current_file}`;
        DOM.ingestPercentage.textContent = `${data.progress}%`;
        DOM.ingestBarFill.style.width = `${data.progress}%`;
        
        if (data.status === 'COMPLETED') {
            clearInterval(state.ingestPoller);
            DOM.ingestFilename.textContent = `Ingested: ${data.current_file} (Success)`;
            DOM.ingestBarFill.style.backgroundColor = 'var(--color-green)';
            setTimeout(() => {
                DOM.ingestProgressContainer.classList.add('hidden');
            }, 5000);
            loadCorpus();
            pollSystemStatus();
        } else if (data.status === 'FAILED') {
            clearInterval(state.ingestPoller);
            DOM.ingestFilename.textContent = `Failed: ${data.error}`;
            DOM.ingestBarFill.style.backgroundColor = 'var(--color-red)';
        }
    } catch (e) {
        console.error("Ingest status poll failed:", e);
    }
}

async function loadCorpus() {
    try {
        const res = await authFetch('/api/corpus');
        if (!res.ok) return;
        const data = await res.json();
        
        DOM.corpusTableBody.innerHTML = '';
        if (data.documents.length === 0) {
            DOM.corpusTableBody.innerHTML = '<tr role="row"><td colspan="4" class="table-empty">No documents indexed in VectorStore.</td></tr>';
            return;
        }

        data.documents.forEach(doc => {
            const tr = document.createElement('tr');
            tr.setAttribute('role', 'row');
            
            const nameTd = document.createElement('td');
            nameTd.textContent = doc.source_doc;
            
            const verTd = document.createElement('td');
            verTd.textContent = doc.doc_version;
            
            const dateTd = document.createElement('td');
            dateTd.textContent = doc.effective_date;
            
            const statusTd = document.createElement('td');
            statusTd.textContent = doc.superseded ? 'Superseded' : 'Active';
            statusTd.className = doc.superseded ? 'color-amber' : 'color-green';
            
            tr.appendChild(nameTd);
            tr.appendChild(verTd);
            tr.appendChild(dateTd);
            tr.appendChild(statusTd);
            
            DOM.corpusTableBody.appendChild(tr);
        });
    } catch (e) {
        console.error("Failed to load corpus documents:", e);
    }
}

// ── Agent Integration / Query Streaming ──────────────────────────────────────
const reasoningStepsMap = {
    'PHI_SCRUB': 'Scrubbing PHI',
    'CRISIS_DETECT': 'Screening for Crises',
    'INTENT_CLASSIFY': 'Classifying Clinical Intent',
    'RETRIEVAL': 'Searching Knowledge Base',
    'CLINICAL_ALERTS': 'Scanning Clinical Warnings',
    'SYNTHESIS': 'Generating Answer',
    'SENTENCE_SPLIT': 'Checking Sentences',
    'NLI_FAITHFULNESS': 'Auditing Faithfulness (NLI)',
    'CONFIDENCE_CALIBRATE': 'Calibrating Confidence',
    'LOOP_DECISION': 'Verification Check'
};

function resetConsoleUI() {
    DOM.stepTimeline.innerHTML = '';
    DOM.responseBox.innerHTML = '<div class="loader-container"><span class="loader"></span> Generating response...</div>';
    DOM.citationsList.innerHTML = '<li class="empty-citations-msg">No citations referenced.</li>';
    DOM.clinicalAlertsContainer.classList.add('hidden');
    DOM.clinicalAlertsList.innerHTML = '';
    DOM.crisisBanner.classList.add('hidden');
    
    // Reset badges
    DOM.confidenceBadge.className = 'metric-badge';
    DOM.confidenceVal.textContent = '--';
    DOM.faithfulnessBadge.className = 'metric-badge';
    DOM.faithfulnessVal.textContent = '--';
    DOM.conditionBadge.className = 'metric-badge';
    DOM.conditionVal.textContent = '--';
}

function updateTimelineStep(stepName, status, details = {}) {
    let indicator = document.getElementById(`step-ind-${stepName}`);
    
    if (!indicator) {
        indicator = document.createElement('span');
        indicator.id = `step-ind-${stepName}`;
        indicator.className = 'step-indicator';
        indicator.textContent = reasoningStepsMap[stepName] || stepName;
        DOM.stepTimeline.appendChild(indicator);
    }

    if (status === 'START' || status.startsWith('START_ITER')) {
        indicator.className = 'step-indicator active';
    } else if (status === 'COMPLETE' || status === 'ACCEPTED') {
        indicator.className = 'step-indicator complete';
    } else if (status === 'FAILED' || status === 'ESCALATED' || status === 'CRISIS_BLOCK') {
        indicator.className = 'step-indicator failed';
    }
    
    // Log to memory ledger
    addAuditLedgerRow(stepName, status, details);
}

// Simple local state to track sequence
let auditRowSequence = 0;

function addAuditLedgerRow(step, status, details) {
    // If it's the start, reset the audit body on first logged step
    if (step === 'PHI_SCRUB' && status === 'START') {
        DOM.auditTableBody.innerHTML = '';
        auditRowSequence = 0;
    }
    
    auditRowSequence++;
    
    const tr = document.createElement('tr');
    tr.setAttribute('role', 'row');
    
    const seqTd = document.createElement('td');
    seqTd.textContent = auditRowSequence;
    
    const tsTd = document.createElement('td');
    tsTd.textContent = new Date().toLocaleTimeString();
    
    const stepTd = document.createElement('td');
    stepTd.textContent = `${reasoningStepsMap[step] || step} (${status})`;
    
    const qhTd = document.createElement('td');
    qhTd.textContent = details.original_query_hash || details.scrubbed_query_preview || '--';
    qhTd.style.fontFamily = 'monospace';
    
    const sigTd = document.createElement('td');
    sigTd.textContent = details.record_hmac || 'Calculating...';
    sigTd.style.fontFamily = 'monospace';
    
    const statusTd = document.createElement('td');
    statusTd.textContent = status;
    
    tr.appendChild(seqTd);
    tr.appendChild(tsTd);
    tr.appendChild(stepTd);
    tr.appendChild(qhTd);
    tr.appendChild(sigTd);
    tr.appendChild(statusTd);
    
    DOM.auditTableBody.appendChild(tr);
}

let queryAbortController = null;

async function submitQuery(queryText) {
    if (!queryText.strip) queryText = queryText.trim();
    if (!queryText) return;
    
    resetConsoleUI();
    DOM.btnSubmit.classList.add('hidden');
    DOM.btnCancel.classList.remove('hidden');
    
    queryAbortController = new AbortController();
    
    try {
        const response = await authFetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: queryText, session_id: state.sessionId }),
            signal: queryAbortController.signal
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Query failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        
        let buffer = '';
        let fullResponseText = '';
        
        DOM.responseBox.innerHTML = ''; // Clear loader
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // Split buffer into events
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep partial line
            
            for (const line of lines) {
                if (!line.trim()) continue;
                
                // Parse EventSource line (format: event: NAME\ndata: CONTENT)
                const parts = line.split('\n');
                let eventName = '';
                let eventData = '';
                
                for (const part of parts) {
                    if (part.startsWith('event: ')) {
                        eventName = part.substring(7).trim();
                    } else if (part.startsWith('data: ')) {
                        eventData = part.substring(6).trim();
                    }
                }
                
                if (eventName === 'token') {
                    fullResponseText += eventData;
                    // Simple Markdown replacement for rendering citations and line breaks
                    DOM.responseBox.innerHTML = formatMarkdown(fullResponseText);
                } else if (eventName === 'error') {
                    DOM.responseBox.innerHTML = `<div class="error-msg">⛔ Error: ${eventData}</div>`;
                } else if (eventName === 'status') {
                    // connected
                } else {
                    // System Reasoning Step
                    try {
                        // Parse JSON state data sent by EventSourceResponse
                        const cleanData = eventData.replace(/'/g, '"').replace(/True/g, 'true').replace(/False/g, 'false').replace(/None/g, 'null');
                        const data = JSON.parse(cleanData);
                        
                        updateTimelineStep(eventName, data.status, data);
                        
                        // Handle Specific step updates
                        if (eventName === 'CRISIS_DETECT' && data.level && data.level !== 'NONE') {
                            DOM.crisisBanner.classList.remove('hidden');
                        }
                        
                        if (eventName === 'RETRIEVAL' && data.citations) {
                            renderCitations(data.citations);
                        }
                        
                        if (eventName === 'CLINICAL_ALERTS' && data.alerts_found && data.alerts_found.length > 0) {
                            renderClinicalAlerts(data.alerts_found);
                        }
                        
                        if (eventName === 'NLI_FAITHFULNESS' && data.score !== undefined) {
                            updateFaithfulnessBadge(data.score, data.blocked);
                        }
                        
                        if (eventName === 'CONFIDENCE_CALIBRATE' && data.confidence_score !== undefined) {
                            updateConfidenceBadge(data.confidence_score);
                        }
                        
                        if (eventName === 'INTENT_CLASSIFY' && data.condition_codes) {
                            DOM.conditionVal.textContent = data.condition_codes.join(', ');
                            DOM.conditionBadge.className = 'metric-badge active';
                        }
                        
                    } catch (err) {
                        console.warn("Failed to parse reasoning step metadata:", eventData, err);
                    }
                }
            }
        }
        
        // Refresh audit table
        loadSessionAuditTable();
        // Load Session History turns
        loadSessionHistory();
        // Refresh escalations count
        pollSystemStatus();
        
    } catch (e) {
        if (e.name === 'AbortError') {
            DOM.responseBox.innerHTML = '<div class="warning-msg">Consultation generation cancelled by operator.</div>';
        } else {
            DOM.responseBox.innerHTML = `<div class="error-msg">⛔ Pipeline Execution Error: ${e.message}</div>`;
        }
    } finally {
        DOM.btnSubmit.classList.remove('hidden');
        DOM.btnCancel.classList.add('hidden');
        queryAbortController = null;
    }
}

function cancelQuery() {
    if (queryAbortController) {
        queryAbortController.abort();
    }
}

// ── Metadata Badge Rendering ────────────────────────────────────────────────
function updateConfidenceBadge(score) {
    DOM.confidenceVal.textContent = `${Math.round(score * 100)}%`;
    if (score >= 0.70) {
        DOM.confidenceBadge.className = 'metric-badge green';
    } else if (score >= 0.50) {
        DOM.confidenceBadge.className = 'metric-badge amber';
    } else {
        DOM.confidenceBadge.className = 'metric-badge red';
    }
}

function updateFaithfulnessBadge(score, blocked) {
    if (blocked) {
        DOM.faithfulnessVal.textContent = 'CONTRADICTED (BLOCKED)';
        DOM.faithfulnessBadge.className = 'metric-badge contradicted';
    } else if (score >= 0.70) {
        DOM.faithfulnessVal.textContent = 'Grounded';
        DOM.faithfulnessBadge.className = 'metric-badge grounded';
    } else {
        DOM.faithfulnessVal.textContent = 'Partially Grounded';
        DOM.faithfulnessBadge.className = 'metric-badge partial';
    }
}

function renderCitations(citations) {
    DOM.citationsList.innerHTML = '';
    if (citations.length === 0) {
        DOM.citationsList.innerHTML = '<li class="empty-citations-msg">No citations referenced.</li>';
        return;
    }

    citations.forEach(cit => {
        const li = document.createElement('li');
        li.className = `citation-card ${cit.superseded ? 'superseded' : ''}`;
        
        const sourceDiv = document.createElement('div');
        sourceDiv.className = 'citation-source';
        sourceDiv.textContent = `${cit.source} (p.${cit.page})`;
        
        const secDiv = document.createElement('div');
        secDiv.className = 'citation-section';
        secDiv.textContent = cit.section;
        
        li.appendChild(sourceDiv);
        li.appendChild(secDiv);
        
        if (cit.superseded) {
            const warningDiv = document.createElement('div');
            warningDiv.className = 'citation-warning';
            warningDiv.textContent = '⚠ Superseded Guideline Version';
            li.appendChild(warningDiv);
        }
        
        DOM.citationsList.appendChild(li);
    });
}

function renderClinicalAlerts(alerts) {
    DOM.clinicalAlertsList.innerHTML = '';
    DOM.clinicalAlertsContainer.classList.remove('hidden');
    
    alerts.forEach(alert => {
        const li = document.createElement('li');
        li.textContent = alert;
        DOM.clinicalAlertsList.appendChild(li);
    });
}

// ── Session History ─────────────────────────────────────────────────────────
async function loadSessionHistory() {
    try {
        const res = await authFetch(`/api/session/${state.sessionId}/context`);
        if (!res.ok) return;
        const data = await res.json();
        
        DOM.sessionHistory.innerHTML = '';
        if (data.turns.length === 0) {
            DOM.sessionHistory.innerHTML = '<div class="empty-history-msg">No active turns in this session yet.</div>';
            return;
        }

        data.turns.forEach(turn => {
            const div = document.createElement('div');
            div.className = 'history-turn';
            
            const q = document.createElement('div');
            q.className = 'history-q';
            q.textContent = `Q: ${turn.query}`;
            
            const a = document.createElement('div');
            a.className = 'history-a';
            a.textContent = turn.answer;
            
            div.appendChild(q);
            div.appendChild(a);
            
            DOM.sessionHistory.appendChild(div);
        });
        
        // Scroll session history to bottom
        DOM.sessionHistory.scrollTop = DOM.sessionHistory.scrollHeight;
    } catch (e) {
        console.error("Failed to load session history:", e);
    }
}

async function clearSession() {
    try {
        await authFetch(`/api/session/${state.sessionId}`, { method: 'DELETE' });
        state.sessionId = `session_${Math.random().toString(36).substring(2, 11)}`;
        loadSessionHistory();
        resetConsoleUI();
        DOM.responseBox.innerHTML = '<div class="response-placeholder">Awaiting clinician query...</div>';
    } catch (e) {
        console.error("Failed to clear session:", e);
    }
}

// ── Escalation Management ────────────────────────────────────────────────────
async function loadEscalations() {
    try {
        const res = await authFetch('/api/escalations');
        if (!res.ok) return;
        const data = await res.json();
        
        DOM.escalationsTableBody.innerHTML = '';
        if (data.escalations.length === 0) {
            DOM.escalationsTableBody.innerHTML = '<tr role="row"><td colspan="5" class="table-empty">No pending escalations found.</td></tr>';
            return;
        }

        data.escalations.forEach(esc => {
            const tr = document.createElement('tr');
            tr.setAttribute('role', 'row');
            
            const idTd = document.createElement('td');
            idTd.textContent = esc.escalation_id.substring(0, 8) + '...';
            idTd.style.fontFamily = 'monospace';
            
            const tsTd = document.createElement('td');
            tsTd.textContent = new Date(esc.timestamp).toLocaleString();
            
            const qTd = document.createElement('td');
            qTd.textContent = esc.scrubbed_query;
            
            const rTd = document.createElement('td');
            rTd.textContent = esc.reason;
            rTd.className = 'color-red';
            
            const actTd = document.createElement('td');
            const btn = document.createElement('button');
            btn.className = 'btn btn-primary btn-sm';
            btn.textContent = 'Resolve';
            btn.addEventListener('click', () => openResolveModal(esc.escalation_id));
            actTd.appendChild(btn);
            
            tr.appendChild(idTd);
            tr.appendChild(dateCell(esc.timestamp));
            tr.appendChild(qTd);
            tr.appendChild(rTd);
            tr.appendChild(actTd);
            
            DOM.escalationsTableBody.appendChild(tr);
        });
    } catch (e) {
        console.error("Failed to load escalations:", e);
    }
}

function dateCell(ts) {
    const td = document.createElement('td');
    td.textContent = new Date(ts).toLocaleString();
    return td;
}

function openResolveModal(id) {
    DOM.resolveEscalationId.value = id;
    DOM.resolutionText.value = '';
    DOM.resolveModal.classList.remove('hidden');
    DOM.resolutionText.focus();
}

function closeModal() {
    DOM.resolveModal.classList.add('hidden');
}

async function handleResolveSubmit(e) {
    e.preventDefault();
    const id = DOM.resolveEscalationId.value;
    const note = DOM.resolutionText.value.trim();
    
    if (!note) return;

    try {
        const res = await authFetch(`/api/escalations/${id}/resolve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ resolution: note })
        });
        
        if (res.ok) {
            closeModal();
            loadEscalations();
            pollSystemStatus();
        } else {
            alert('Failed to resolve escalation');
        }
    } catch (err) {
        console.error("Resolve failed:", err);
    }
}

// ── Audit Ledger Verification & Export ───────────────────────────────────────
async function loadSessionAuditTable() {
    // If no records logged, the table displays default.
}

async function verifyAuditLedger() {
    DOM.btnVerifyAudit.textContent = 'Verifying...';
    try {
        const res = await authFetch('/api/audit/verify');
        const data = await res.json();
        
        if (res.ok && data.verified) {
            alert('✓ Audit Ledger Integrity Verified: All SHA-256 hash links and HMAC signatures are valid.');
        } else {
            alert(`⛔ Audit Ledger Tamper Detected!\nErrors found:\n${data.errors.join('\n')}`);
        }
    } catch (e) {
        alert(`Verification failed: ${e.message}`);
    } finally {
        DOM.btnVerifyAudit.textContent = 'Verify Ledger Integrity';
    }
}

async function exportAuditLogs() {
    try {
        const res = await authFetch(`/api/audit/export/${state.sessionId}`);
        if (!res.ok) {
            alert('No audit logs found for this session.');
            return;
        }
        const data = await res.json();
        
        // Trigger file download in browser
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
        const dlAnchor = document.createElement('a');
        dlAnchor.setAttribute("href", dataStr);
        dlAnchor.setAttribute("download", `sentinel_audit_${state.sessionId}.json`);
        document.body.appendChild(dlAnchor);
        dlAnchor.click();
        dlAnchor.remove();
    } catch (e) {
        alert(`Export failed: ${e.message}`);
    }
}

// ── Markdown Formatter ───────────────────────────────────────────────────────
function formatMarkdown(text) {
    // Simple sanitization
    let clean = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    
    // Bold tags **text**
    clean = clean.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Line breaks
    clean = clean.replace(/\n/g, '<br>');
    
    // Bullet lists
    clean = clean.replace(/^- (.*?)(<br>|$)/gm, '<li>$1</li>');
    clean = clean.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
    
    return clean;
}

// ── Accessibility (WCAG 2.1 AA) Navigation ───────────────────────────────────
function setupAccessibility() {
    // Escape key cancels query synthesis
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            cancelQuery();
            closeModal();
        }
    });

    // Keyboard navigation focus control inside modal
    DOM.resolveModal.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            const focusables = DOM.resolveModal.querySelectorAll('button, textarea');
            const first = focusables[0];
            const last = focusables[focusables.length - 1];
            
            if (e.shiftKey && document.activeElement === first) {
                last.focus();
                e.preventDefault();
            } else if (!e.shiftKey && document.activeElement === last) {
                first.focus();
                e.preventDefault();
            }
        }
    });
}

// ── Initialization ──────────────────────────────────────────────────────────
async function init() {
    await initializeAuthToken();
    setupTabs();
    setupDragAndDrop();
    setupAccessibility();
    
    // Pollers
    pollSystemStatus();
    state.statusPoller = setInterval(pollSystemStatus, 5000);
    
    // Form triggers
    DOM.queryForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = DOM.queryInput.value;
        DOM.queryInput.value = '';
        submitQuery(text);
    });
    
    DOM.btnCancel.addEventListener('click', cancelQuery);
    DOM.btnClearSession.addEventListener('click', clearSession);
    DOM.btnRotateToken.addEventListener('click', rotateToken);
    DOM.btnVerifyAudit.addEventListener('click', verifyAuditLedger);
    DOM.btnExportAudit.addEventListener('click', exportAuditLogs);
    
    DOM.btnCloseModal.addEventListener('click', closeModal);
    DOM.resolveForm.addEventListener('submit', handleResolveSubmit);

    logger.info("Consultation Dashboard fully initialized.");
}

// Run init on DOM load
window.addEventListener('DOMContentLoaded', init);
