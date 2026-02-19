// STATE
let soundEnabled = true;
const fallCards = {};
let fallCount = 0;

const alertBanner = document.getElementById("alertBanner");
const alertIcon = alertBanner.querySelector(".alert-icon");
const alertText = alertBanner.querySelector(".alert-text");
const videoOverlay = document.getElementById("videoOverlay");
const soundBtn = document.getElementById("soundBtn");
const eventsCount = document.getElementById("eventsCount");
const connectionStatus = document.getElementById("connectionStatus");

// SOUND
function playFallAlarm() {
  if (!soundEnabled) return;
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const now = ctx.currentTime;
    [0, 0.22, 0.44].forEach((offset) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = 880;
      osc.type = "square";
      gain.gain.setValueAtTime(0.2, now + offset);
      gain.gain.exponentialRampToValueAtTime(0.01, now + offset + 0.12);
      osc.start(now + offset);
      osc.stop(now + offset + 0.12);
    });
    setTimeout(() => ctx.close(), 2000);
  } catch (_) {}
}

function playRecoverySound() {
  if (!soundEnabled) return;
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const now = ctx.currentTime;
    [523.25, 659.25, 783.99].forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = freq;
      osc.type = "sine";
      gain.gain.setValueAtTime(0.15, now + i * 0.12);
      gain.gain.exponentialRampToValueAtTime(0.01, now + i * 0.12 + 0.3);
      osc.start(now + i * 0.12);
      osc.stop(now + i * 0.12 + 0.3);
    });
    setTimeout(() => ctx.close(), 2000);
  } catch (_) {}
}

// SOUND TOGGLE
soundBtn.addEventListener("click", function () {
  soundEnabled = !soundEnabled;
  this.querySelector("span").textContent = soundEnabled ? "Sound ON" : "Sound OFF";
});

// ALERTS
let alertTimeout = null;

function showFallAlert(personId) {
  if (alertTimeout) clearTimeout(alertTimeout);
  alertBanner.className = "alert-banner active fall";
  alertIcon.textContent = "\u26a0\ufe0f";
  alertText.textContent = `FALL DETECTED \u2014 Person ${personId}`;
  alertBanner.querySelector(".alert-dismiss").style.display = "inline-flex";
  videoOverlay.className = "feed-overlay fall-active";
  playFallAlarm();
}

function showRecoveryAlert(personId) {
  if (alertTimeout) clearTimeout(alertTimeout);
  alertBanner.className = "alert-banner active recovered";
  alertIcon.textContent = "\u2714\ufe0f";
  alertText.textContent = `Person ${personId} recovered \u2014 false alarm`;
  alertBanner.querySelector(".alert-dismiss").style.display = "inline-flex";
  videoOverlay.className = "feed-overlay";
  playRecoverySound();
  alertTimeout = setTimeout(dismissAlert, 8000);
}

function dismissAlert() {
  alertBanner.className = "alert-banner";
  videoOverlay.className = "feed-overlay";
  if (alertTimeout) { clearTimeout(alertTimeout); alertTimeout = null; }
}

window.dismissAlert = dismissAlert;

// FALL CARDS
function formatTime(ts) {
  if (!ts) return "";
  const parts = ts.split(" ");
  return parts.length > 1 ? parts[1] : ts;
}

function detailItem(label, value, opts) {
  const cls = opts && opts.fullWidth ? "detail-item full-width" : "detail-item";
  const extra = opts && opts.statusClass ? ` ${opts.statusClass}` : "";
  const valCls = opts && opts.valClass ? ` class="detail-value ${opts.valClass}"` : ' class="detail-value"';
  return `<div class="${cls}${extra ? " status-item" : ""}">
    <div class="detail-label">${label}</div>
    <div${valCls}>${value}</div>
  </div>`;
}

function fmtNum(v) {
  if (typeof v === "number") return v.toFixed(2);
  var n = parseFloat(v);
  return isNaN(n) ? v : n.toFixed(2);
}

function buildDetailsHTML(fall) {
  let html = '<div class="fall-card-divider"></div>';
  html += `<img class="fall-screenshot" src="${fall.screenshot}" alt="Fall screenshot" loading="lazy">`;

  const fd = fall.fall_data || {};
  const det = fall.detection || {};
  const foi = fall.fall_object_interaction;

  // Status
  html += '<div class="detail-grid">';
  const statusLabel = fall.status === "active" ? "Active" : "Recovered";
  const statusVal = fall.status === "active" ? "active" : "recovered";
  html += detailItem("Status", `<span id="status-${fall.id}">${statusLabel}</span>`, { statusClass: "status-item", valClass: statusVal });
  if (fall.recovered_at) html += detailItem("Recovered At", formatTime(fall.recovered_at));
  html += "</div>";

  // Detection & Kinematics
  html += '<div class="detail-section"><div class="detail-section-title">Detection &amp; Kinematics</div>';
  html += '<div class="detail-grid">';
  if (det.angle != null) html += detailItem("Torso Angle", `${det.angle}&deg;`);
  if (det.box_ratio != null) html += detailItem("Box Ratio", det.box_ratio);
  if (det.head_position_pct != null) html += detailItem("Head Position", `${det.head_position_pct}%`);
  if (fd.head_speed != null) html += detailItem("Head Speed", fmtNum(fd.head_speed) + " px/s");
  if (fd.head_accel != null) html += detailItem("Head Accel", fmtNum(fd.head_accel) + " px/s\u00b2");
  if (fd.shoulder_speed != null) html += detailItem("Shoulder Speed", fmtNum(fd.shoulder_speed) + " px/s");
  html += "</div></div>";

  // Fall Timing
  html += '<div class="detail-section"><div class="detail-section-title">Fall Timing</div>';
  html += '<div class="detail-grid">';
  if (fd.fall_frame != null) html += detailItem("Fall Frame", fd.fall_frame);
  if (fd.fall_id != null) html += detailItem("Fall ID", fd.fall_id);
  if (fd.person_id != null) html += detailItem("Person ID", fd.person_id);
  if (fd.fall_start) {
    var fs = Array.isArray(fd.fall_start) ? fd.fall_start : [fd.fall_start];
    html += detailItem("Fall Start", "Frame " + fs[0]);
  }
  if (fd.fall_end) {
    var fe = Array.isArray(fd.fall_end) ? fd.fall_end : [fd.fall_end];
    html += detailItem("Fall End", "Frame " + fe[0]);
  }
  if (fd.fall_start && fd.fall_end) {
    var startArr = Array.isArray(fd.fall_start) ? fd.fall_start : [fd.fall_start];
    var endArr = Array.isArray(fd.fall_end) ? fd.fall_end : [fd.fall_end];
    if (startArr[1] != null && endArr[1] != null) {
      var duration = Math.abs(Number(endArr[1]) - Number(startArr[1]));
      html += detailItem("Duration", duration.toFixed(2) + "s");
    }
  }
  html += "</div></div>";

  // Object Interaction (only after enrichment)
  if (foi) {
    html += '<div class="detail-section"><div class="detail-section-title">Fall Context</div>';
    html += '<div class="detail-grid">';
    if (foi.pre_fall_posture) html += detailItem("Pre-fall Posture", foi.pre_fall_posture);
    if (foi.fall_time_sec != null) html += detailItem("Fall Time", `${foi.fall_time_sec}s`);

    if (foi.was_sitting_on && foi.was_sitting_on.length > 0) {
      const sitStr = foi.was_sitting_on.map(function (s) {
        return typeof s === "string" ? s : s.object + " (" + s.seconds_before_fall + "s before)";
      }).join(", ");
      html += detailItem("Was Sitting On", sitStr, { fullWidth: true });
    }

    if (foi.first_floor_contact) {
      if (typeof foi.first_floor_contact === "string") {
        html += detailItem("Floor Contact", foi.first_floor_contact);
      } else {
        var ffc = foi.first_floor_contact;
        html += detailItem("Floor Contact", ffc.body_part || "unknown");
        if (ffc.all_parts && ffc.all_parts.length > 0) html += detailItem("All Parts", ffc.all_parts.join(", "), { fullWidth: true });
        if (ffc.flat_fall != null) html += detailItem("Flat Fall", ffc.flat_fall ? "Yes" : "No");
        if (ffc.frame != null) html += detailItem("Contact Frame", ffc.frame);
        if (ffc.time_sec != null) html += detailItem("Contact Time", ffc.time_sec + "s");
      }
    }
    html += "</div>";

    // Object interactions
    if (foi.object_interactions && foi.object_interactions.length > 0) {
      html += '<div class="detail-subsection"><div class="detail-subsection-title">Object Interactions</div>';
      html += '<div class="interaction-list">';
      foi.object_interactions.forEach(function (inter) {
        var riskCls = (inter.injury_risk || "").toLowerCase();
        html += '<div class="interaction-row">';
        html += '<span class="inter-object">' + inter.object + '</span>';
        html += '<span class="inter-badge risk-' + riskCls + '">' + (inter.injury_risk || "N/A") + '</span>';
        if (inter.touching) html += '<span class="inter-badge inter-touching">touching</span>';
        if (inter.proximity_px != null) html += '<span class="inter-proximity">' + inter.proximity_px + 'px</span>';
        if (inter.body_parts && inter.body_parts.length > 0) html += '<span class="inter-parts">' + inter.body_parts.join(", ") + '</span>';
        html += '</div>';
      });
      html += "</div></div>";
    }

    // Body part touches by phase
    if (foi.body_part_touches) {
      var bpt = foi.body_part_touches;
      var phases = [
        { key: "while_falling", label: "While Falling" },
        { key: "during_fall", label: "During Fall" },
        { key: "after_fall", label: "After Fall" }
      ];
      var hasAny = phases.some(function (p) { return bpt[p.key] && bpt[p.key].length > 0; });
      if (hasAny) {
        html += '<div class="detail-subsection"><div class="detail-subsection-title">Body Part Touches</div>';
        phases.forEach(function (phase) {
          var items = bpt[phase.key];
          if (!items || items.length === 0) return;
          html += '<div class="phase-group"><div class="phase-label">' + phase.label + '</div>';
          items.forEach(function (t) {
            var riskCls = (t.risk || "").toLowerCase();
            html += '<div class="interaction-row">';
            html += '<span class="inter-object">' + t.object + '</span>';
            html += '<span class="inter-badge risk-' + riskCls + '">' + (t.risk || "") + '</span>';
            if (t.body_parts && t.body_parts.length > 0) html += '<span class="inter-parts">' + t.body_parts.join(", ") + '</span>';
            html += '</div>';
          });
          html += "</div>";
        });
        html += "</div>";
      }
    }

    // Summary
    if (foi.summary) {
      var s = foi.summary;
      html += '<div class="detail-subsection"><div class="detail-subsection-title">Summary</div>';
      html += '<div class="detail-grid">';
      if (s.pre_fall_posture) html += detailItem("Posture", s.pre_fall_posture);
      if (s.was_sitting_on && s.was_sitting_on.length > 0) html += detailItem("Sitting On", s.was_sitting_on.join(", "));
      if (s.first_floor_contact) html += detailItem("Floor Contact", s.first_floor_contact);
      html += "</div></div>";
    }

    html += "</div>";
  }

  // Frames Data
  if (fd.frames && fd.frames.length > 0) {
    html += '<div class="detail-section"><div class="detail-section-title">Frames Data <span class="frame-count">(' + fd.frames.length + ' frames)</span></div>';
    html += '<div class="frames-table-wrap"><table class="frames-table"><thead><tr>';
    html += "<th>Frame</th><th>Time</th><th>Angle</th><th>\u0394Angle</th><th>Head Y</th><th>Shoulder Y</th><th>\u0394Head</th><th>Horiz</th><th>Vert</th><th>Ongoing</th>";
    html += "</tr></thead><tbody>";
    fd.frames.forEach(function (fr) {
      html += "<tr>";
      html += "<td>" + (fr.frame_id != null ? fr.frame_id : "-") + "</td>";
      html += "<td>" + (fr.timestamp != null ? Number(fr.timestamp).toFixed(2) : "-") + "</td>";
      html += "<td>" + (fr.angle != null ? Number(fr.angle).toFixed(1) : "-") + "</td>";
      html += "<td>" + (fr.angle_change != null ? Number(fr.angle_change).toFixed(1) : "-") + "</td>";
      html += "<td>" + (fr.head_y != null ? Number(fr.head_y).toFixed(1) : "-") + "</td>";
      html += "<td>" + (fr.shoulder_y != null ? Number(fr.shoulder_y).toFixed(1) : "-") + "</td>";
      html += "<td>" + (fr.head_change != null ? Number(fr.head_change).toFixed(1) : "-") + "</td>";
      html += "<td>" + (fr.horizontal || "-") + "</td>";
      html += "<td>" + (fr.vertical || "-") + "</td>";
      html += "<td>" + (fr.fall_ongoing || "-") + "</td>";
      html += "</tr>";
    });
    html += "</tbody></table></div></div>";
  }

  // Video Clip
  if (fall.clip) {
    html += '<video class="fall-clip" controls src="' + fall.clip + '"></video>';
  } else {
    html += '<p class="clip-pending" id="clip-' + fall.id + '">Recording clip...</p>';
  }

  return html;
}

function createFallCard(fall) {
  var noEvents = document.getElementById("noEvents");
  if (noEvents) noEvents.style.display = "none";

  fallCount++;
  eventsCount.textContent = fallCount;

  var card = document.createElement("div");
  card.className = "fall-card status-" + fall.status;
  card.id = "fall-" + fall.id;

  var time = formatTime(fall.timestamp);
  var badgeCls = fall.status === "active" ? "badge-active" : "badge-recovered";
  var badgeLabel = fall.status === "active" ? "Active" : "Recovered";
  var personLabel = fall.person_id != null ? "Person " + fall.person_id : "";

  card.innerHTML =
    '<div class="fall-card-header">' +
      '<div>' +
        '<div class="fall-card-title">#' + fall.id + ' Fall Detected</div>' +
        '<div class="fall-card-meta">' +
          '<span class="fall-card-time">' + time + '</span>' +
          (personLabel ? '<span class="fall-card-person">' + personLabel + '</span>' : "") +
        '</div>' +
      '</div>' +
      '<div class="fall-card-right">' +
        '<span class="badge ' + badgeCls + '">' + badgeLabel + '</span>' +
        '<span class="chevron">\u25bc</span>' +
      '</div>' +
    '</div>' +
    '<div class="fall-card-body">' +
      '<div class="fall-card-details">' + buildDetailsHTML(fall) + '</div>' +
    '</div>';

  card.querySelector(".fall-card-header").addEventListener("click", function () {
    card.classList.toggle("expanded");
  });

  var eventList = document.getElementById("eventList");
  eventList.prepend(card);
  fallCards[fall.id] = { card: card, data: fall };
}

function updateFallCard(fall) {
  var entry = fallCards[fall.id];
  if (!entry) return;
  var card = entry.card;
  entry.data = fall;

  var wasExpanded = card.classList.contains("expanded");
  card.className = "fall-card status-" + fall.status + (wasExpanded ? " expanded" : "");

  var badge = card.querySelector(".badge");
  if (badge) {
    badge.className = "badge " + (fall.status === "active" ? "badge-active" : "badge-recovered");
    badge.textContent = fall.status === "active" ? "Active" : "Recovered";
  }

  var body = card.querySelector(".fall-card-details");
  if (body) body.innerHTML = buildDetailsHTML(fall);

  if (fall.clip) {
    var pending = card.querySelector("#clip-" + fall.id);
    if (pending) {
      pending.outerHTML = '<video class="fall-clip" controls src="' + fall.clip + '"></video>';
    }
  }
}

function enrichFallCard(fall) {
  var entry = fallCards[fall.id];
  if (!entry) return;
  entry.data = fall;
  var body = entry.card.querySelector(".fall-card-details");
  if (body) body.innerHTML = buildDetailsHTML(fall);
}

// SSE
var es = new EventSource("/events");

es.onopen = function () {
  connectionStatus.textContent = "Connected";
  connectionStatus.classList.add("connected");
};

es.onmessage = function (e) {
  try {
    var data = JSON.parse(e.data);

    if (data.type === "fall_detected") {
      createFallCard(data.fall);
      showFallAlert(data.fall.person_id);
    } else if (data.type === "fall_history") {
      createFallCard(data.fall);
    } else if (data.type === "fall_recovered") {
      updateFallCard(data.fall);
      showRecoveryAlert(data.fall.person_id);
    } else if (data.type === "fall_enriched") {
      enrichFallCard(data.fall);
    }
  } catch (err) {
    console.error("SSE parse error:", err);
  }
};

es.onerror = function () {
  connectionStatus.textContent = "Reconnecting...";
  connectionStatus.classList.remove("connected");
};
