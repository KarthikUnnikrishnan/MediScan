import os

# 1. Update templates/scanner/result.html
with open('templates/scanner/result.html', 'r', encoding='utf-8') as f:
    res_html = f.read()

target_pill = '''        <div class="status-pills">
          <span class="status-pill">
            {% if mode == 'prescription' %}
              📋 Prescription
            {% else %}
              💊 Medicine Strip
            {% endif %}
          </span>
          <span class="status-pill">🕐 {{ timestamp|date:"j M Y, g:i A" }}</span>
          {% if total_found and total_found > 1 %}
          <span class="status-pill green">
            {{ total_found }} medicines found
          </span>
          {% endif %}
        </div>'''

replacement_pill = '''        {% if total_found and total_found > 1 %}
        <div class="status-pills">
          <span class="status-pill green">
            {{ total_found }} medicines found
          </span>
        </div>
        {% endif %}'''

if target_pill in res_html:
    res_html = res_html.replace(target_pill, replacement_pill)
    with open('templates/scanner/result.html', 'w', encoding='utf-8', newline='\n') as f:
        f.write(res_html)
    print("result.html updated: Medicine Strip and timestamp badge removed.")
else:
    print("Warning: target_pill not found in result.html")
