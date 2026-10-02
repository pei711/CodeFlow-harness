// 由 server.js 在 `/` 提供。仪表盘外壳以 JS 模块而非 .html 资源存放于此，
// 使查看器只需分发纯 JS/CSS。
module.exports = String.raw`<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Tracing Dashboard</title>
    <link rel="stylesheet" href="/app.css">
  </head>
  <body>
    <header class="app-header">
      <div class="app-brand">
        <div class="app-logo" aria-hidden="true">◆</div>
        <div class="app-brand-copy">
          <div class="app-brand-line">
            <h1>CodeFlow</h1>
            <span class="app-product">Tracing</span>
            <span class="app-product-meta">local viewer</span>
          </div>
          <p data-i18n="header.subtitle">Agent execution observability</p>
        </div>
      </div>
      <div class="app-status">
        <span class="status-pill" id="connectionStatus" data-i18n="status.disconnected">Disconnected</span>
        <span class="window-pill" id="dataWindowStatus" hidden></span>
        <span class="status-text"><span data-i18n="header.updated">Updated</span>: <strong id="lastUpdated">--:--:--</strong></span>
        <div class="lang-switch" role="group" aria-label="Language">
          <button class="lang-pill" data-lang="en" type="button">EN</button>
          <button class="lang-pill" data-lang="zh" type="button">中</button>
        </div>
        <button class="ghost-button auto-refresh-button" id="autoRefreshButton" type="button" aria-pressed="true">
          <span class="auto-refresh-dot" aria-hidden="true"></span>
          <span id="autoRefreshLabel" data-i18n="auto.on">Auto 15s</span>
        </button>
        <button class="ghost-button app-action" id="refreshButton" type="button" data-i18n="action.refresh">Refresh</button>
        <button class="ghost-button danger-button" id="shutdownButton" type="button">
          <svg aria-hidden="true" viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
            <path d="M10 2.5v7"></path>
            <path d="M5.1 5.3a7 7 0 1 0 9.8 0"></path>
          </svg>
          <span data-i18n="action.stop">Stop viewer</span>
        </button>
      </div>
    </header>
    <nav class="workspace-tabs">
      <button class="workspace-tab" data-app-view="api" type="button" data-i18n="view.api">API Calls</button>
      <button class="workspace-tab is-active" data-app-view="trace" type="button" data-i18n="view.trace">Traces</button>
    </nav>
    <div class="scene scene-trace shell" id="traceScene">
      <aside class="pane pane-sessions">
        <div class="pane-head">
          <div>
            <h1 id="listTitle" data-i18n="sessions.title">Sessions</h1>
          </div>
        </div>
        <div class="controls">
          <label class="field">
            <span>Agent</span>
            <select id="agentFilter"></select>
          </label>
          <label class="field field-search">
            <span data-i18n="field.search">Search</span>
            <input id="searchInput" type="search" data-i18n-ph="search.placeholder" placeholder="session / trace / keyword">
          </label>
          <label class="field api-only">
            <span>Provider</span>
            <select id="providerFilter"></select>
          </label>
          <label class="field api-only">
            <span>Model</span>
            <select id="modelFilter"></select>
          </label>
          <label class="field api-only">
            <span data-i18n="field.status">Status</span>
            <select id="statusFilter">
              <option value="all" data-i18n="status.all">All statuses</option>
              <option value="ok" data-i18n="status.okOnly">Success only</option>
              <option value="error" data-i18n="status.errorOnly">Failed only</option>
              <option value="unreported" data-i18n="status.unreportedOnly">Token unreported only</option>
            </select>
          </label>
        </div>
        <div class="content-search-results" id="contentSearchResults" hidden></div>
        <div class="session-list" id="sessionList" data-scroll-key="trace-session-list"></div>
      </aside>

      <main class="pane pane-traces">
        <div class="pane-head">
          <div>
            <h2 id="traceTitle" data-i18n="trace.selectSession">Select a session</h2>
          </div>
          <div class="head-meta" id="traceMeta"></div>
        </div>
        <div class="trace-list" id="traceList" data-scroll-key="trace-list"></div>
      </main>

      <section class="pane pane-details">
        <div class="pane-head">
          <div>
            <h2 id="detailsTitle" data-i18n="details.selectSpan">Select a span</h2>
          </div>
          <div class="tabs">
            <button class="tab is-active" data-tab="content" type="button" data-i18n="detailtab.content">Input / Output</button>
            <button class="tab" data-tab="metadata" type="button" data-i18n="detailtab.metadata">Metadata</button>
            <button class="tab" data-tab="raw" type="button" data-i18n="detailtab.raw">Raw</button>
          </div>
        </div>
        <div class="details-body" id="detailsBody" data-scroll-key="trace-details"></div>
      </section>
    </div>
    <main class="scene scene-api" id="apiScene"></main>

    <dialog class="shutdown-dialog" id="shutdownDialog" aria-labelledby="shutdownTitle">
      <div class="dialog-mark" aria-hidden="true">◆</div>
      <div class="dialog-copy">
        <p class="dialog-kicker" data-i18n="shutdown.kicker">Local viewer</p>
        <h2 id="shutdownTitle" data-i18n="shutdown.title">Stop the tracing viewer?</h2>
        <p data-i18n="shutdown.copy">This stops the local server and automatic refresh. Trace files stay on disk.</p>
        <p class="dialog-error" id="shutdownError" hidden></p>
      </div>
      <div class="dialog-actions">
        <button class="ghost-button" id="shutdownCancelButton" type="button" data-i18n="action.cancel">Cancel</button>
        <button class="ghost-button danger-button is-solid" id="shutdownConfirmButton" type="button" data-i18n="action.confirmStop">Stop viewer</button>
      </div>
    </dialog>

    <div class="shutdown-state" id="shutdownState" hidden>
      <div class="shutdown-state-mark" aria-hidden="true">◇</div>
      <p class="shutdown-state-kicker" data-i18n="shutdown.stoppedKicker">Viewer stopped</p>
      <h2 data-i18n="shutdown.stoppedTitle">Tracing is no longer using this process.</h2>
      <p data-i18n="shutdown.stoppedCopy">Your trace files are preserved. You can close this tab or run codeflow tracing to start again.</p>
    </div>

    <template id="emptyStateTemplate">
      <div class="empty-state">
        <p class="empty-title" data-i18n="empty.title">Nothing to show yet</p>
        <p class="empty-copy" data-i18n="empty.copy">Refresh, or send another message to try.</p>
      </div>
    </template>

    <script src="/app.js"></script>
  </body>
  </html>
`;
