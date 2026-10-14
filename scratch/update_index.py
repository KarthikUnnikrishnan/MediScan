with open('templates/scanner/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Replace mode selector (remove Auto Detect, make Medicine Strip default)
old_mode = '''          <div class="mode-selector-wrap">
            <div class="mode-selector-title">
              What are you scanning?
            </div>
            <div class="mode-selector-grid">

              <input type="radio" name="mode" id="mode_auto"
                     value="auto" class="hidden-radio" checked>
              <label for="mode_auto" class="mode-tile">
                <span class="mode-tile-icon-svg icon-auto">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 7V5a2 2 0 0 1 2-2h2"></path>
                    <path d="M17 3h2a2 2 0 0 1 2 2v2"></path>
                    <path d="M21 17v2a2 2 0 0 1-2 2h-2"></path>
                    <path d="M7 21H5a2 2 0 0 1-2-2v-2"></path>
                    <circle cx="12" cy="12" r="3.2"></circle>
                    <path d="m15.5 15.5 2 2"></path>
                  </svg>
                </span>
                <span class="mode-tile-name">Auto Detect</span>
                <span class="mode-tile-desc">
                  AI decides automatically
                </span>
              </label>

              <input type="radio" name="mode" id="mode_strip"
                     value="medicine_package" class="hidden-radio">
              <label for="mode_strip" class="mode-tile">
                <span class="mode-tile-icon-svg icon-strip">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"></path>
                    <path d="m8.5 8.5 7 7"></path>
                  </svg>
                </span>
                <span class="mode-tile-name">Medicine Strip</span>
                <span class="mode-tile-desc">
                  Box, blister pack or label
                </span>
              </label>

              <input type="radio" name="mode" id="mode_rx"
                     value="prescription" class="hidden-radio">
              <label for="mode_rx" class="mode-tile mode-tile-rx">
                <span class="mode-tile-icon-svg icon-rx">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                    <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                    <line x1="9" y1="12" x2="15" y2="12"></line>
                    <line x1="9" y1="16" x2="13" y2="16"></line>
                  </svg>
                </span>
                <span class="mode-tile-name">Prescription</span>
                <span class="mode-tile-desc">
                  Doctor's Rx — handwritten
                </span>
                <span class="mode-tile-badge">
                  Multiple drugs
                </span>
              </label>

            </div>

            <div class="mode-hint" id="modeHint">
              💡 Use <strong>Prescription</strong> mode for doctor
              handwritten notes — Auto may misread them
            </div>
          </div>'''

new_mode = '''          <div class="mode-selector-wrap">
            <div class="mode-selector-title">
              What are you scanning?
            </div>
            <div class="mode-selector-grid">

              <input type="radio" name="mode" id="mode_strip"
                     value="medicine_package" class="hidden-radio" checked>
              <label for="mode_strip" class="mode-tile">
                <span class="mode-tile-icon-svg icon-strip">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z"></path>
                    <path d="m8.5 8.5 7 7"></path>
                  </svg>
                </span>
                <span class="mode-tile-name">Medicine Strip</span>
                <span class="mode-tile-desc">
                  Box, blister pack or label
                </span>
              </label>

              <input type="radio" name="mode" id="mode_rx"
                     value="prescription" class="hidden-radio">
              <label for="mode_rx" class="mode-tile mode-tile-rx">
                <span class="mode-tile-icon-svg icon-rx">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                    <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                    <line x1="9" y1="12" x2="15" y2="12"></line>
                    <line x1="9" y1="16" x2="13" y2="16"></line>
                  </svg>
                </span>
                <span class="mode-tile-name">Prescription</span>
                <span class="mode-tile-desc">
                  Doctor's Rx — handwritten
                </span>
                <span class="mode-tile-badge">
                  Multiple drugs
                </span>
              </label>

            </div>

            <div class="mode-hint" id="modeHint">
              💡 Select <strong>Medicine Strip</strong> for packaged tablets/bottles, or <strong>Prescription</strong> for doctor handwritten notes.
            </div>
          </div>'''

# 2. Replace dropzone camera upload button
old_dropzone_inner = '''            <div class="dropzone-compact-inner">
              <div class="dropzone-camera-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                  <circle cx="12" cy="13" r="4"></circle>
                </svg>
              </div>
              <div class="dropzone-text-group">
                <div class="dropzone-prompt">Tap to photograph or upload</div>
                <div class="dropzone-format-info">JPG &bull; PNG &bull; WEBP &bull; Max 10MB</div>
              </div>
              <button type="button" class="btn-compact-camera" id="btnTriggerCamera" title="Take a live photo with camera" onclick="event.preventDefault(); event.stopPropagation();">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                  <circle cx="12" cy="13" r="4"></circle>
                </svg>
                <span>Camera</span>
              </button>
            </div>'''

new_dropzone_inner = '''            <div class="dropzone-compact-inner">
              <div class="dropzone-camera-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
              </div>
              <div class="dropzone-text-group">
                <div class="dropzone-prompt">Choose an image or drag &amp; drop</div>
                <div class="dropzone-format-info">JPG &bull; PNG &bull; WEBP &bull; Max 10MB</div>
              </div>
            </div>'''

# 3. Add confidence pill to alternatives preview in hero
old_alt_tag = '''                  {% if alt.salt %}
                  <div class="alt-salt-tag">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle; margin-right:3px;"><path d="M10 2v7.31"></path><path d="M14 9.3V2"></path><path d="M8.5 2h7"></path><path d="M14 9.3a6.5 6.5 0 1 1-4 0"></path><path d="M5.52 16h12.96"></path></svg>
                    {{ alt.salt }}
                  </div>
                  {% endif %}
                </div>'''

new_alt_tag = '''                  {% if alt.salt %}
                  <div class="alt-salt-tag">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle; margin-right:3px;"><path d="M10 2v7.31"></path><path d="M14 9.3V2"></path><path d="M8.5 2h7"></path><path d="M14 9.3a6.5 6.5 0 1 1-4 0"></path><path d="M5.52 16h12.96"></path></svg>
                    {{ alt.salt }}
                  </div>
                  {% endif %}
                  {% if alt.confidence %}
                  <div class="alt-trust-inline" style="display:flex; align-items:center; gap:6px; margin-top:5px; flex-wrap:wrap;">
                    <span class="alt-conf-pill {% if alt.confidence >= 95 %}conf-high{% elif alt.confidence >= 90 %}conf-med{% else %}conf-fair{% endif %}">
                      🛡️ {{ alt.confidence }}% Trust Confidence
                    </span>
                    <span class="alt-trust-badge">
                      {{ alt.trust_label|default:"Bioequivalent Match" }}
                    </span>
                    {% if alt.current_price_range %}
                    <span class="alt-market-price" style="margin-left:0;">
                      📊 {{ alt.current_price_range }}
                    </span>
                    {% endif %}
                  </div>
                  {% endif %}
                </div>'''

assert old_mode in html, "old_mode not found"
html = html.replace(old_mode, new_mode)

assert old_dropzone_inner in html, "old_dropzone_inner not found"
html = html.replace(old_dropzone_inner, new_dropzone_inner)

assert old_alt_tag in html, "old_alt_tag not found"
html = html.replace(old_alt_tag, new_alt_tag)

with open('templates/scanner/index.html', 'w', encoding='utf-8', newline='\n') as f:
    f.write(html)

print("Successfully updated templates/scanner/index.html")
