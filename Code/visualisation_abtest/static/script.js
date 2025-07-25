document.addEventListener('DOMContentLoaded', function() {
  const segmentSelect = document.getElementById('segment-select');

  // Fetch available segments and populate the dropdown
  fetch('/segments')
    .then(response => {
      if (!response.ok) throw new Error(`Failed to load segments: ${response.status}`);
      return response.json();
    })
    .then(data => {
      const segments = data.segments;
      if (segments && segments.length > 0) {
        segments.forEach(seg => {
          const option = document.createElement('option');
          option.value = seg;
          option.text = seg;
          segmentSelect.appendChild(option);
        });
        // Load the first segment by default
        loadSegment(segments[0]);
      }
    })
    .catch(error => console.error('Error fetching segments:', error));

  // When user selects a new segment, load it
  segmentSelect.addEventListener('change', function() {
    loadSegment(this.value);
  });

  // Helper: Clear a plot container and show a spinner
  function resetPlotContainer(divId) {
    const container = document.getElementById(divId);
    container.innerHTML = `
      <div class="spinner-container">
        <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
          <span class="visually-hidden">Loading...</span>
        </div>
      </div>
    `;
  }

  // Helper: Display an error message inside a plot container
  function showError(divId, msg) {
    const container = document.getElementById(divId);
    container.innerHTML = `<pre class="text-danger">${msg}</pre>`;
  }

  // Helper: Remove spinner (after successful plot)
  function removeSpinner(divId) {
    const container = document.getElementById(divId);
    const spinner = container.querySelector('.spinner-container');
    if (spinner) spinner.remove();
  }

  // Create plots for a given audio type and remove spinner once done
  function createPlots(audioType, data, audioElem) {
    const divPrefix = audioType + '-';
    const plotTypes = ['waveform', 'spectrogram', 'pitch'];
    let autoscaleLock = false;

    // Waveform
    const wfTrace = {
      x: data.waveform.time,
      y: data.waveform.audio,
      type: 'scatter', mode: 'lines', name: 'Waveform',
      line: { width: 1 }
    };
    const wfLayout = {
      title: 'Waveform',
      xaxis: { title: 'Time (s)', showgrid: false, autorange: true },
      yaxis: { title: 'Amplitude', showgrid: false },
      margin: { t: 50 }, shapes: [], annotations: []
    };
    Plotly.newPlot(divPrefix + 'waveform', [wfTrace], wfLayout)
      .then(() => removeSpinner(divPrefix + 'waveform'));

    // Spectrogram
    const specTrace = {
      x: data.spectrogram.time,
      y: data.spectrogram.freqs,
      z: data.spectrogram.D,
      type: 'heatmap', showscale: false
    };
    const specLayout = {
      title: 'Spectrogram',
      xaxis: { title: 'Time (s)', showgrid: false, autorange: true },
      yaxis: { title: 'Frequency (Hz)', showgrid: false },
      margin: { t: 50 }, shapes: [], annotations: []
    };
    Plotly.newPlot(divPrefix + 'spectrogram', [specTrace], specLayout)
      .then(() => removeSpinner(divPrefix + 'spectrogram'));

    // Pitch
    const pitchTrace = {
      x: data.pitch.time,
      y: data.pitch.f0,
      type: 'scatter', mode: 'lines', name: 'Pitch',
      line: { width: 1 }
    };
    const pitchLayout = {
      title: 'Pitch Contour',
      xaxis: { title: 'Time (s)', showgrid: false, autorange: true },
      yaxis: { title: 'Frequency (Hz)', showgrid: false },
      margin: { t: 50 }, shapes: [], annotations: []
    };
    Plotly.newPlot(divPrefix + 'pitch', [pitchTrace], pitchLayout)
      .then(() => removeSpinner(divPrefix + 'pitch'));

    // Word intervals on all plots
    data.intervals.forEach(([start, end, label]) => {
      if (!label) return;
      plotTypes.forEach(pt => {
        const plotDiv = document.getElementById(divPrefix + pt);
        const shapes = plotDiv.layout.shapes ? plotDiv.layout.shapes.slice() : [];
        const lineColor = (pt === 'spectrogram') ? 'white' : 'black';
        shapes.push(
          { type: 'line', x0: start, x1: start, yref: 'paper', y0: 0, y1: 1,
            line: { color: lineColor, dash: 'dash', width: 1 } },
          { type: 'line', x0: end,   x1: end,   yref: 'paper', y0: 0, y1: 1,
            line: { color: lineColor, dash: 'dash', width: 1 } }
        );
        const ann = {
          x: (start + end)/2, y: 1, yref: 'paper', text: label,
          showarrow: false, font: { size: 10 },
          bgcolor: 'rgba(255,255,255,0.7)', bordercolor: lineColor,
          borderpad: 4
        };
        const annotations = plotDiv.layout.annotations ? plotDiv.layout.annotations.slice() : [];
        annotations.push(ann);
        Plotly.relayout(divPrefix + pt, { shapes, annotations });
      });
    });

    // Dynamic playback line
    audioElem.addEventListener('timeupdate', function() {
      const t = audioElem.currentTime;
      const line = { type: 'line', x0: t, x1: t, yref: 'paper', y0: 0, y1: 1,
                     line: { color: 'red', dash: 'dot', width: 2 } };
      plotTypes.forEach(pt => {
        const pd = document.getElementById(divPrefix + pt);
        const old = pd.layout.shapes
          ? pd.layout.shapes.filter(s => !(s.line && s.line.color === 'red'))
          : [];
        Plotly.relayout(divPrefix + pt, { shapes: [...old, line] });
      });
    });

    // Sync zoom/pan across plots
    plotTypes.forEach(pt => {
      const pd = document.getElementById(divPrefix + pt);
      pd.on('plotly_relayout', evt => {
        if (evt['xaxis.autorange'] && !autoscaleLock) {
          autoscaleLock = true;
          plotTypes.forEach(op => Plotly.relayout(divPrefix + op, { 'xaxis.autorange': true }));
          setTimeout(() => { autoscaleLock = false; }, 100);
        } else if (evt['xaxis.range[0]'] != null && evt['xaxis.range[1]'] != null) {
          const xmin = evt['xaxis.range[0]'], xmax = evt['xaxis.range[1]'];
          plotTypes.forEach(op => {
            if (op !== pt) Plotly.relayout(divPrefix + op, { 'xaxis.range': [xmin, xmax] });
          });
        }
      });
    });
  }

  // Load & render a single plot type (/plot_data) with error handling
  function loadPlot(audioType, segment) {
    ['waveform','spectrogram','pitch'].forEach(pt =>
      resetPlotContainer(`${audioType}-${pt}`)
    );

    fetch(`/plot_data/${audioType}/${segment}`)
      .then(response => {
        if (!response.ok) {
          return response.text().then(txt => {
            ['waveform','spectrogram','pitch'].forEach(pt =>
              showError(`${audioType}-${pt}`, `Error ${response.status}: ${txt}`)
            );
            throw new Error(`Plot data fetch failed: ${response.status}`);
          });
        }
        return response.json();
      })
      .then(data => {
        const audioElem = document.getElementById(`${audioType}-audio`);
        createPlots(audioType, data, audioElem);
      })
      .catch(err => console.error(err));
  }

  // Switch segment, set audio src, and trigger plots
  function loadSegment(segment) {
    const impAudio = document.getElementById('improved-audio');
    const ttsAudio = document.getElementById('tts-audio');
    impAudio.src = `/audio/improved/${segment}`;
    ttsAudio.src = `/audio/tts/${segment}`;
    impAudio.load();
    ttsAudio.load();
    loadPlot('improved', segment);
    loadPlot('tts', segment);
  }
});
