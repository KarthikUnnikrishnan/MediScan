/**
 * MediScan — Client Interaction Script
 * Aesthetic: Clinical-Luxury / Modern Editorial Health-Tech
 * Powers segmented mode controls, drag-and-drop dropzone, live camera modal,
 * sample demo fast-scanner, real-time loading overlay, scan history drawer,
 * and FAQ accordion.
 */

(function () {
  'use strict';

  /* ── 1. SEGMENTED MODE SELECTOR ────────────────────────────────────────── */
  function initSegmentedControl() {
    const stripTab = document.getElementById('tabStrip');
    const rxTab = document.getElementById('tabRx');
    const modeStrip = document.getElementById('modeStrip');
    const modeRx = document.getElementById('modeRx');

    if (!stripTab || !rxTab) return;

    stripTab.addEventListener('click', function () {
      stripTab.classList.add('active');
      stripTab.setAttribute('aria-selected', 'true');
      rxTab.classList.remove('active');
      rxTab.setAttribute('aria-selected', 'false');
      if (modeStrip) modeStrip.checked = true;
    });

    rxTab.addEventListener('click', function () {
      rxTab.classList.add('active');
      rxTab.setAttribute('aria-selected', 'true');
      stripTab.classList.remove('active');
      stripTab.setAttribute('aria-selected', 'false');
      if (modeRx) modeRx.checked = true;
    });
  }

  /* ── 2. DROPZONE & FILE PREVIEW ────────────────────────────────────────── */
  function initDropzone() {
    const dropzone = document.getElementById('scannerDropzone');
    const fileInput = document.getElementById('imageInput') || document.getElementById('id_image');
    const previewCard = document.getElementById('previewCard');
    const previewThumb = document.getElementById('previewThumb');
    const previewFilename = document.getElementById('previewFilename');
    const previewSize = document.getElementById('previewSize');
    const btnRemove = document.getElementById('btnRemovePreview');
    const scanBtn = document.getElementById('btnScanSubmit');

    if (!dropzone || !fileInput) return;

    // Open file browser on dropzone click (unless camera trigger is clicked)
    dropzone.addEventListener('click', function (e) {
      if (e.target.closest('#btnTriggerCamera')) {
        e.preventDefault();
        e.stopPropagation();
        return;
      }
      if (dropzone.tagName.toLowerCase() !== 'label') {
        fileInput.click();
      }
    });

    dropzone.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
      }
    });

    // Drag-over styling
    ['dragenter', 'dragover'].forEach(function (evt) {
      dropzone.addEventListener(evt, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('drag-active');
      });
    });

    ['dragleave', 'drop'].forEach(function (evt) {
      dropzone.addEventListener(evt, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('drag-active');
      });
    });

    // Drop handler
    dropzone.addEventListener('drop', function (e) {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length > 0) {
        const file = dt.files[0];
        try {
          const container = new DataTransfer();
          container.items.add(file);
          fileInput.files = container.files;
        } catch (_) {}
        displayFilePreview(file);
        fileInput.dispatchEvent(new Event('change'));
      }
    });

    // Native file input change
    fileInput.addEventListener('change', function () {
      if (fileInput.files && fileInput.files.length > 0) {
        displayFilePreview(fileInput.files[0]);
      }
    });

    // Remove preview
    if (btnRemove) {
      btnRemove.addEventListener('click', function (e) {
        e.stopPropagation();
        fileInput.value = '';
        if (previewCard) previewCard.style.display = 'none';
        if (dropzone) dropzone.style.display = 'block';
      });
    }

    function displayFilePreview(file) {
      if (!file) return;

      if (previewFilename) previewFilename.textContent = file.name;
      if (previewSize) {
        const kb = (file.size / 1024).toFixed(1);
        const sizeStr = kb > 1024 ? (kb / 1024).toFixed(2) + ' MB' : kb + ' KB';
        previewSize.textContent = sizeStr;
      }

      if (file.type.startsWith('image/') && previewThumb) {
        const reader = new FileReader();
        reader.onload = function (evt) {
          previewThumb.src = evt.target.result;
        };
        reader.readAsDataURL(file);
      }

      if (dropzone) dropzone.style.display = 'none';
      if (previewCard) previewCard.style.display = 'flex';
    }

    // Expose setter for camera and sample demos
    window.setMediScanFile = function (file) {
      try {
        const container = new DataTransfer();
        container.items.add(file);
        fileInput.files = container.files;
      } catch (_) {}
      displayFilePreview(file);
    };
  }

  /* ── 3. LIVE CAMERA SNAPSHOT MODAL ─────────────────────────────────────── */
  function initCamera() {
    const btnTrigger = document.getElementById('btnTriggerCamera');
    const modalBackdrop = document.getElementById('cameraModalBackdrop');
    const btnClose = document.getElementById('btnCloseCamera');
    const btnCloseTop = document.getElementById('btnCloseCameraTop');
    const btnCapture = document.getElementById('btnCapturePhoto');
    const videoElem = document.getElementById('cameraVideo');

    if (!btnTrigger || !modalBackdrop || !videoElem) return;

    let mediaStream = null;

    btnTrigger.addEventListener('click', async function (e) {
      e.stopPropagation();
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        videoElem.srcObject = mediaStream;
        await videoElem.play();
        modalBackdrop.classList.add('open');
      } catch (err) {
        alert('Camera could not be accessed. Please upload an image directly or grant camera permissions in your browser.');
      }
    });

    function closeCamera() {
      if (mediaStream) {
        mediaStream.getTracks().forEach(function (track) { track.stop(); });
        mediaStream = null;
      }
      modalBackdrop.classList.remove('open');
    }

    if (btnClose) btnClose.addEventListener('click', closeCamera);
    if (btnCloseTop) btnCloseTop.addEventListener('click', closeCamera);

    modalBackdrop.addEventListener('click', function (e) {
      if (e.target === modalBackdrop) closeCamera();
    });

    if (btnCapture) {
      btnCapture.addEventListener('click', function () {
        const canvas = document.createElement('canvas');
        canvas.width = videoElem.videoWidth || 640;
        canvas.height = videoElem.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElem, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(function (blob) {
          if (!blob) return;
          const snapFile = new File([blob], 'camera_scan_' + Date.now() + '.jpg', { type: 'image/jpeg' });
          if (window.setMediScanFile) {
            window.setMediScanFile(snapFile);
          }
          closeCamera();
        }, 'image/jpeg', 0.92);
      });
    }
  }

  /* ── 4. SAMPLE DEMO FAST-SCANNER ────────────────────────────────────────── */
  function initSampleDemos() {
    const sampleBtns = document.querySelectorAll('.sample-card-btn');
    if (!sampleBtns.length) return;

    sampleBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        const mode = btn.getAttribute('data-sample-mode') || 'medicine_package';
        const name = btn.getAttribute('data-sample-name') || 'Dolo 650 Tablet';

        // Update active state on sample buttons
        sampleBtns.forEach(function (b) { b.classList.remove('active-sample'); });
        btn.classList.add('active-sample');

        // 1. Switch mode
        if (mode === 'prescription') {
          const rxRadio = document.getElementById('mode_rx');
          if (rxRadio) { rxRadio.checked = true; rxRadio.dispatchEvent(new Event('change')); }
          const tabRx = document.getElementById('tabRx');
          if (tabRx) tabRx.click();
        } else {
          const stripRadio = document.getElementById('mode_strip');
          if (stripRadio) { stripRadio.checked = true; stripRadio.dispatchEvent(new Event('change')); }
          const tabStrip = document.getElementById('tabStrip');
          if (tabStrip) tabStrip.click();
        }

        // 2. Synthesize ultra-realistic medical test image on canvas
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        if (mode === 'prescription') {
          // Prescription Pad Note Simulation
          canvas.width = 720;
          canvas.height = 500;

          // Cream medical notepad background
          ctx.fillStyle = '#faf8f5';
          ctx.fillRect(0, 0, 720, 500);

          // Top Header Bar
          ctx.fillStyle = '#143d2b';
          ctx.fillRect(40, 25, 640, 4);

          ctx.fillStyle = '#143d2b';
          ctx.font = 'bold 22px "Plus Jakarta Sans", sans-serif';
          ctx.textAlign = 'left';
          ctx.fillText('METROPOLITAN CLINICAL CARE CENTER', 50, 60);

          ctx.fillStyle = '#475569';
          ctx.font = '13px "Plus Jakarta Sans", sans-serif';
          ctx.fillText('Dr. Arvind Sharma, MBBS, MD (Medicine) • Reg: MCI-49201', 50, 85);
          ctx.fillText('14 Healthcare Blvd, Ground Floor • Helpline: +91 98765 43210', 50, 105);

          // Divider
          ctx.strokeStyle = '#cde2d4';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(40, 125);
          ctx.lineTo(680, 125);
          ctx.stroke();

          // Patient metadata
          ctx.fillStyle = '#334155';
          ctx.font = '14px "Plus Jakarta Sans", sans-serif';
          ctx.fillText('Patient: Rahul Verma    Age/Sex: 38 / M    Date: 2026-10-05', 50, 155);

          // Rx Symbol
          ctx.fillStyle = '#15803d';
          ctx.font = 'bold 42px Georgia, serif';
          ctx.fillText('℞', 50, 215);

          // Prescribed Medication Line 1
          ctx.fillStyle = '#0f172a';
          ctx.font = 'bold 24px "Plus Jakarta Sans", sans-serif';
          ctx.fillText(name, 105, 215);

          ctx.fillStyle = '#475569';
          ctx.font = '15px "Plus Jakarta Sans", sans-serif';
          ctx.fillText('Sig: 1 Tablet Twice Daily (BD) after meals × 5 days', 105, 245);

          // Prescribed Medication Line 2
          ctx.fillStyle = '#0f172a';
          ctx.font = 'bold 20px "Plus Jakarta Sans", sans-serif';
          ctx.fillText('Tab Pantoprazole 40mg', 105, 295);

          ctx.fillStyle = '#475569';
          ctx.font = '15px "Plus Jakarta Sans", sans-serif';
          ctx.fillText('Sig: 1 Tablet Once Daily (OD) before breakfast × 5 days', 105, 320);

          // Advice box
          ctx.fillStyle = '#ebf5ee';
          ctx.fillRect(50, 360, 620, 48);
          ctx.fillStyle = '#1b4a35';
          ctx.font = '13px "Plus Jakarta Sans", sans-serif';
          ctx.fillText('Advice: Drink plenty of fluids. Review after 5 days if fever persists.', 65, 390);

          // Signature Line
          ctx.strokeStyle = '#94a3b8';
          ctx.beginPath();
          ctx.moveTo(490, 455);
          ctx.lineTo(670, 455);
          ctx.stroke();

          ctx.fillStyle = '#0f172a';
          ctx.font = 'italic bold 20px "Brush Script MT", cursive, sans-serif';
          ctx.fillText('Dr. A. Sharma', 520, 445);

          ctx.fillStyle = '#64748b';
          ctx.font = '11px "Plus Jakarta Sans", sans-serif';
          ctx.fillText("Physician's Signature & Stamp", 510, 472);

        } else {
          // Blister Packaging Simulation
          canvas.width = 720;
          canvas.height = 460;

          // Background
          ctx.fillStyle = '#f1f5f2';
          ctx.fillRect(0, 0, 720, 460);

          // Blister Pack Border & Surface
          const grad = ctx.createLinearGradient(40, 30, 680, 420);
          grad.addColorStop(0, '#ffffff');
          grad.addColorStop(0.3, '#f4fbf7');
          grad.addColorStop(0.7, '#e6f4ec');
          grad.addColorStop(1, '#ffffff');
          ctx.fillStyle = grad;
          ctx.strokeStyle = '#bbf7d0';
          ctx.lineWidth = 3;
          if (ctx.roundRect) {
            ctx.roundRect(40, 30, 640, 400, 16);
          } else {
            ctx.rect(40, 30, 640, 400);
          }
          ctx.fill();
          ctx.stroke();

          // Metallic blister pocket circles
          for (let i = 0; i < 5; i++) {
            const cx = 115 + i * 122;
            const cy = 110;
            const rad = ctx.createRadialGradient(cx - 8, cy - 8, 4, cx, cy, 38);
            rad.addColorStop(0, '#ffffff');
            rad.addColorStop(0.55, '#d1fae5');
            rad.addColorStop(1, '#a7f3d0');
            ctx.fillStyle = rad;
            ctx.beginPath();
            ctx.arc(cx, cy, 38, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#6ee7b7';
            ctx.lineWidth = 2;
            ctx.stroke();
          }

          // Medicine Name
          ctx.fillStyle = '#143d2b';
          ctx.font = 'bold 32px "Plus Jakarta Sans", sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(name, 360, 225);

          // Composition
          ctx.fillStyle = '#059669';
          ctx.font = '600 16px "Plus Jakarta Sans", sans-serif';
          ctx.fillText('Active Molecule • Bioequivalent Generic Formulation', 360, 260);

          // Storage info
          ctx.fillStyle = '#475569';
          ctx.font = '14px "Plus Jakarta Sans", sans-serif';
          ctx.fillText('Schedule H Prescription Drug • Store below 30°C in dry place', 360, 295);

          // Batch info
          ctx.fillStyle = '#1e293b';
          ctx.font = '12px "Courier New", monospace';
          ctx.fillText('B.No. MS-84920  MFG. 09/2026  EXP. 08/2028  M.R.P. ₹34.50', 360, 335);

          // Warning Red Bar
          ctx.fillStyle = '#ef4444';
          ctx.fillRect(110, 360, 500, 3);
          ctx.font = 'bold 11px "Plus Jakarta Sans", sans-serif';
          ctx.fillText('WARNING: To be sold by retail on the prescription of a Registered Medical Practitioner only.', 360, 380);
        }

        // 3. Convert to File and set into dropzone input
        canvas.toBlob(function (blob) {
          if (!blob) return;
          const cleanFileName = name.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase() + '.jpg';
          const sampleFile = new File([blob], cleanFileName, { type: 'image/jpeg' });
          if (window.setMediScanFile) {
            window.setMediScanFile(sampleFile);
          }

          // 4. Visual feedback: Pulse scanner terminal and highlight submit button
          const terminal = document.getElementById('scanner-terminal');
          if (terminal) {
            terminal.classList.remove('scanner-sample-loaded');
            void terminal.offsetWidth;
            terminal.classList.add('scanner-sample-loaded');
          }

          const submitBtn = document.getElementById('btnScanSubmit');
          if (submitBtn) {
            submitBtn.classList.add('btn-sample-ready');
          }

          // Smooth scroll to terminal
          if (terminal) {
            terminal.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }, 'image/jpeg', 0.95);
      });
    });
  }

  /* ── 5. FORM SUBMISSION PROGRESS OVERLAY ────────────────────────────────── */
  function initFormProgress() {
    const form = document.getElementById('scanForm');
    const overlay = document.getElementById('loadingOverlay');
    const stepTitle = document.getElementById('loadingStepTitle');
    const stepDesc = document.getElementById('loadingStepDesc');
    const progressFill = document.getElementById('loadingProgressFill');
    const fileInput = document.getElementById('imageInput') || document.getElementById('id_image');

    if (!form || !overlay) return;

    form.addEventListener('submit', function (e) {
      if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        alert('Please photograph or choose a medicine image first.');
        return;
      }

      overlay.style.display = 'flex';

      const steps = [
        { t: 'Uploading scan media…', d: 'Buffering high-resolution image for neural inspection', p: '25%' },
        { t: 'Detecting strip contours…', d: 'YOLOv8 segmenting blister packaging and text boundaries', p: '50%' },
        { t: 'Reading prescription typography…', d: 'TrOCR vision transformer recognizing printed & cursive text', p: '75%' },
        { t: 'Verifying active pharmacology…', d: 'Extracting chemical salts, calculating generic savings and interactions', p: '92%' }
      ];

      let idx = 0;
      const interval = setInterval(function () {
        idx++;
        if (idx < steps.length && stepTitle && stepDesc) {
          stepTitle.textContent = steps[idx].t;
          stepDesc.textContent = steps[idx].d;
          if (progressFill) progressFill.style.width = steps[idx].p;
        } else {
          clearInterval(interval);
        }
      }, 1200);
    });
  }

  /* ── 6. FAQ ACCORDION INTERACTION ──────────────────────────────────────── */
  function initFaqAccordion() {
    const faqCards = document.querySelectorAll('.faq-card');
    faqCards.forEach(function (card) {
      const toggleBtn = card.querySelector('.faq-toggle');
      if (!toggleBtn) return;

      toggleBtn.addEventListener('click', function () {
        const isOpen = card.classList.contains('open');
        // Close others for neat single-open accordion
        faqCards.forEach(c => {
          c.classList.remove('open');
          const b = c.querySelector('.faq-toggle');
          if (b) b.setAttribute('aria-expanded', 'false');
        });

        if (!isOpen) {
          card.classList.add('open');
          toggleBtn.setAttribute('aria-expanded', 'true');
        }
      });
    });
  }

  /* ── 7. SCROLL TO RESULTS ──────────────────────────────────────────────── */
  function checkScrollToResults() {
    const resultsSection = document.getElementById('results-section');
    if (resultsSection && resultsSection.style.display !== 'none') {
      setTimeout(function () {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 250);
    }
  }

  /* ── 8. ACTIVE NAV HIGHLIGHT & SMOOTH SCROLL ─────────────────────────── */
  function initNavScrollHighlight() {
    const sections = document.querySelectorAll('section[id]');
    const navItems = document.querySelectorAll('.nav-menu .nav-item');

    if (!sections.length || !navItems.length) return;

    window.addEventListener('scroll', function () {
      let currentSectionId = '';
      const scrollPos = window.scrollY + 140;

      sections.forEach(function (section) {
        if (section.offsetTop <= scrollPos) {
          currentSectionId = section.getAttribute('id');
        }
      });

      navItems.forEach(function (item) {
        const href = item.getAttribute('href');
        if (href && href.includes('#' + currentSectionId)) {
          navItems.forEach(i => i.classList.remove('active'));
          item.classList.add('active');
        }
      });
    }, { passive: true });

    // Smooth scroll for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        const targetId = this.getAttribute('href').substring(1);
        if (!targetId) return;
        const targetElem = document.getElementById(targetId);
        if (targetElem) {
          e.preventDefault();
          targetElem.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });
  }

  /* ── 9. ENFORCE LIGHT MODE ONLY ────────────────────────────────────────── */
  function enforceLightMode() {
    try {
      localStorage.removeItem('mediscan_theme');
      localStorage.removeItem('mediscan_history');
      document.documentElement.removeAttribute('data-theme');
    } catch (_) {}
  }

  /* ── 1b. MODE SELECTOR TILES CLICK & SYNC ────────────────────────────── */
  function initModeSelector() {
    const tiles = document.querySelectorAll('.mode-tile');
    tiles.forEach(function (tile) {
      tile.addEventListener('click', function (e) {
        const forId = tile.getAttribute('for');
        if (!forId) return;
        const radio = document.getElementById(forId);
        if (radio) {
          radio.checked = true;
          radio.dispatchEvent(new Event('change', { bubbles: true }));
          document.querySelectorAll('.mode-tile').forEach(t => t.classList.remove('auto-suggested'));
        }
      });
    });

    document.querySelectorAll('input[name="mode"]').forEach(function (radio) {
      radio.addEventListener('change', function () {
        var hint = document.getElementById('modeHint');
        if (!hint) return;
        if (this.value === 'auto') {
          hint.textContent = '💡 Use Prescription mode for doctor handwritten notes — Auto may misread them';
          hint.classList.add('visible');
        } else {
          hint.classList.remove('visible');
        }
      });
    });
  }

  /* ── DOM BOOTSTRAP ──────────────────────────────────────────────────────── */
  function boot() {
    enforceLightMode();
    initSegmentedControl();
    initModeSelector();
    initDropzone();
    initCamera();
    initSampleDemos();
    initFormProgress();
    initFaqAccordion();
    initNavScrollHighlight();
    checkScrollToResults();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();

/* ── Medicine Selector (Prescription Mode) ── */

let _currentMedIndex = 0;

function selectMedicine(medKey) {
  if (typeof SCAN_MODE === 'undefined' || SCAN_MODE !== 'prescription') return;

  // Update selector card highlighting
  document.querySelectorAll('.med-selector-card').forEach(card => {
    card.classList.remove('selected');
  });
  const activeCard = document.querySelector(
    `.med-selector-card[data-med-key="${CSS.escape(medKey)}"]`
  );
  if (activeCard) activeCard.classList.add('selected');

  // Find the index of this medicine
  const detailPanels = document.querySelectorAll('.presc-med-detail');
  const altPanels    = document.querySelectorAll('.presc-alt-panel');

  let targetIndex = 0;
  detailPanels.forEach((panel, i) => {
    if (panel.dataset.medKey === medKey) {
      targetIndex = i;
    }
  });

  // Hide all
  detailPanels.forEach(p => {
    p.style.display = 'none';
    p.classList.remove('panel-fade-in');
  });
  altPanels.forEach(p => {
    p.style.display = 'none';
    p.classList.remove('panel-fade-in');
  });

  // Show selected with animation
  const newDetail = detailPanels[targetIndex];
  const newAlt    = altPanels[targetIndex];

  if (newDetail) {
    newDetail.style.display = 'block';
    requestAnimationFrame(() => {
      newDetail.classList.add('panel-fade-in');
    });
  }
  if (newAlt) {
    newAlt.style.display = 'block';
    requestAnimationFrame(() => {
      newAlt.classList.add('panel-fade-in');
    });
  }

  _currentMedIndex = targetIndex;
}

// Keyboard navigation (left/right arrow keys)
document.addEventListener('keydown', function(e) {
  if (typeof SCAN_MODE === 'undefined' || SCAN_MODE !== 'prescription') return;
  const cards = document.querySelectorAll('.med-selector-card');
  if (!cards.length) return;

  let newIndex = _currentMedIndex;
  if (e.key === 'ArrowRight') newIndex = Math.min(_currentMedIndex + 1, cards.length - 1);
  if (e.key === 'ArrowLeft')  newIndex = Math.max(_currentMedIndex - 1, 0);
  if (newIndex !== _currentMedIndex) {
    selectMedicine(cards[newIndex].dataset.medKey);
  }
});

/* ── Auto-select prescription mode when file looks like
      a document (tall aspect ratio) ── */
function checkImageForPrescription(file) {
  var url = URL.createObjectURL(file);
  var img = new Image();
  img.onload = function() {
    var ratio = img.height / img.width;
    // Portrait + tall → likely prescription
    if (ratio > 1.2) {
      var rxRadio = document.getElementById('mode_rx');
      if (rxRadio) {
        rxRadio.checked = true;
        rxRadio.dispatchEvent(new Event('change', { bubbles: true }));
        // Trigger visual update
        document.querySelectorAll('.mode-tile')
          .forEach(function(t) { t.classList.remove('auto-suggested'); });
        var rxLabel = document.querySelector('label[for="mode_rx"]');
        if (rxLabel) {
          rxLabel.classList.add('auto-suggested');
        }
        // Show suggestion message
        var hint = document.getElementById('modeHint');
        if (hint) {
          hint.textContent =
            '📋 Portrait image detected — switched to Prescription mode. '
            + 'Change above if this is a medicine box photo.';
          hint.classList.add('visible');
        }
      }
    }
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

/* Hook into file selection */
var imageInput = document.getElementById('imageInput') || document.getElementById('id_image');
if (imageInput) {
  imageInput.addEventListener('change', function(e) {
    if (e.target.files && e.target.files[0]) {
      checkImageForPrescription(e.target.files[0]);
    }
  });
}


