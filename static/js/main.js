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
        const mode = btn.getAttribute('data-sample-mode');
        const name = btn.getAttribute('data-sample-name');

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

        // 2. Synthesize a high-contrast test medicine image on canvas
        const canvas = document.createElement('canvas');
        canvas.width = 600;
        canvas.height = 400;
        const ctx = canvas.getContext('2d');

        // Background
        ctx.fillStyle = '#f8faf7';
        ctx.fillRect(0, 0, 600, 400);

        // Strip blister simulation
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#cde2d4';
        ctx.lineWidth = 4;
        ctx.strokeRect(40, 40, 520, 320);
        ctx.fillRect(40, 40, 520, 320);

        // Blister pockets
        ctx.fillStyle = '#ebf5ee';
        for (let i = 0; i < 4; i++) {
          ctx.beginPath();
          ctx.arc(100 + i * 130, 110, 42, 0, Math.PI * 2);
          ctx.fill();
        }

        // Medicine label text
        ctx.fillStyle = '#143d2b';
        ctx.font = 'bold 30px "Plus Jakarta Sans", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(name, 300, 220);

        ctx.fillStyle = '#475569';
        ctx.font = '16px "Plus Jakarta Sans", sans-serif';
        ctx.fillText('Rx Formulation • For Medical Use Only • Batch #2026', 300, 260);

        ctx.fillStyle = '#b45309';
        ctx.font = 'bold 15px "Plus Jakarta Sans", sans-serif';
        ctx.fillText('Keep out of reach of children • Store below 30°C', 300, 290);

        // 3. Convert to File and set into input
        canvas.toBlob(function (blob) {
          if (!blob) return;
          const cleanFileName = name.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase() + '.jpg';
          const sampleFile = new File([blob], cleanFileName, { type: 'image/jpeg' });
          if (window.setMediScanFile) {
            window.setMediScanFile(sampleFile);
          }

          // 4. Smooth scroll to terminal
          const terminal = document.getElementById('scanner-terminal');
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


