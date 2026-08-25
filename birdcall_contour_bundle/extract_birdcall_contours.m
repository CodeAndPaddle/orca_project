function results = extract_birdcall_contours(wavFile, intervals, cfg)
%EXTRACT_BIRDCALL_CONTOURS Standalone bird-call contour extractor.
%
% results = extract_birdcall_contours(wavFile, intervals, cfg)
%
% Inputs
%   wavFile   Full path to one audio recording.
%   intervals Numeric N-by-2 [startSec endSec], a table containing
%             tStartSec/tEndSec, or [] for automatic detection.
%   cfg       Optional struct. Start from birdcall_contour_default_config().
%
% Outputs
%   results.contours            Contour struct array.
%   results.summaryTable        One row per accepted contour.
%   results.pointTable          One row per contour time-frequency point.
%   results.intervalStatusTable One row per requested/detected interval.

% The implementation is self-contained. It uses MATLAB built-in
% spectrogram(), zero-phase bandpass filtering, robust enhancement, and a
% constrained Viterbi/dynamic-programming ridge tracker.

if nargin < 2
    intervals = [];
end

defaults = birdcall_contour_default_config();
if nargin < 3 || isempty(cfg)
    cfg = defaults;
else
    cfg = merge_structs(defaults, cfg);
end

wavFile = char(string(wavFile));
if ~isfile(wavFile)
    error('birdcontour:InputNotFound', 'Recording not found: %s', wavFile);
end

info = audioinfo(wavFile);
fs = info.SampleRate;
if fs <= cfg.audio.minSampleRateHz
    error('birdcontour:SampleRateTooLow', ...
        'Sample rate %.1f Hz is too low for the configured %.1f-%.1f Hz band.', ...
        fs, cfg.filter.bandHz(1), cfg.filter.bandHz(2));
end

[wavDir, fileStem, ext] = fileparts(wavFile);
fileName = [fileStem ext];

if isempty(cfg.output.dir)
    cfg.output.dir = fullfile(wavDir, [fileStem '_birdcall_contours']);
end
if isempty(cfg.output.prefix)
    cfg.output.prefix = fileStem;
end
cfg.output.prefix = sanitize_name(cfg.output.prefix);

safe_mkdir(cfg.output.dir);
figuresDir = fullfile(cfg.output.dir, 'figures');
if cfg.figures.savePng || cfg.figures.saveFig
    safe_mkdir(figuresDir);
end

if isempty(intervals)
    intervalTable = auto_detect_intervals(wavFile, info, cfg);
    intervalMode = 'auto';
else
    intervalTable = normalize_intervals(intervals, info.Duration);
    intervalMode = 'provided';
end

nIntervals = height(intervalTable);
status = strings(nIntervals, 1);
errorMessage = strings(nIntervals, 1);
contourIndex = NaN(nIntervals, 1);
contours = empty_contours();
figureCount = 0;

if cfg.debug.verbose
    fprintf('birdcontour: file=%s\n', wavFile);
    fprintf('birdcontour: mode=%s, intervals=%d, fs=%.1f Hz\n', ...
        intervalMode, nIntervals, fs);
    fprintf('birdcontour: output=%s\n', cfg.output.dir);
end

for iInterval = 1:nIntervals
    tStartSec = intervalTable.tStartSec(iInterval);
    tEndSec = intervalTable.tEndSec(iInterval);

    if (tEndSec - tStartSec) > cfg.interval.warnDurationSec && cfg.debug.verbose
        warning('birdcontour:LongInterval', ...
            'Interval %d is %.3f s long. One Viterbi ridge will be returned.', ...
            iInterval, tEndSec - tStartSec);
    end

    try
        [ridge, diagnostic] = extract_one_interval( ...
            wavFile, info, tStartSec, tEndSec, cfg);

        if ridge.valid
            c = make_contour(fileName, wavFile, intervalTable(iInterval, :), ...
                ridge, numel(contours) + 1);
            contours = append_contours(contours, c);
            contourIndex(iInterval) = numel(contours);
            status(iInterval) = "accepted";
        else
            status(iInterval) = "rejected_low_confidence";
        end

        shouldSave = ridge.valid || cfg.figures.saveRejected;
        if shouldSave && figureCount < cfg.figures.maxFigures && ...
                (cfg.figures.savePng || cfg.figures.saveFig)
            save_interval_figure(diagnostic, ridge, intervalTable(iInterval, :), ...
                figuresDir, cfg.output.prefix, cfg);
            figureCount = figureCount + 1;
        end
    catch ME
        status(iInterval) = "error";
        errorMessage(iInterval) = string(ME.message);
        warning('birdcontour:IntervalFailed', ...
            'Interval %d [%.6f, %.6f] failed: %s', ...
            iInterval, tStartSec, tEndSec, ME.message);
    end
end

intervalStatusTable = intervalTable;
intervalStatusTable.status = status;
intervalStatusTable.contourIndex = contourIndex;
intervalStatusTable.errorMessage = errorMessage;

summaryTable = make_summary_table(contours);
pointTable = make_point_table(contours);

summaryCsvPath = fullfile(cfg.output.dir, ...
    [cfg.output.prefix '_contour_summary.csv']);
pointsCsvPath = fullfile(cfg.output.dir, ...
    [cfg.output.prefix '_contour_points.csv']);
intervalCsvPath = fullfile(cfg.output.dir, ...
    [cfg.output.prefix '_interval_status.csv']);
matPath = fullfile(cfg.output.dir, ...
    [cfg.output.prefix '_contours.mat']);

if cfg.output.writeCsv
    writetable(summaryTable, summaryCsvPath);
    writetable(pointTable, pointsCsvPath);
    writetable(intervalStatusTable, intervalCsvPath);
end

results = struct();
results.wavFile = wavFile;
results.fileName = fileName;
results.fs = fs;
results.durationSec = info.Duration;
results.intervalMode = intervalMode;
results.cfg = cfg;
results.intervals = intervalTable;
results.intervalStatusTable = intervalStatusTable;
results.contours = contours;
results.summaryTable = summaryTable;
results.pointTable = pointTable;
results.outputDir = cfg.output.dir;
results.summaryCsvPath = summaryCsvPath;
results.pointsCsvPath = pointsCsvPath;
results.intervalCsvPath = intervalCsvPath;
results.matPath = matPath;

if cfg.output.saveMat
    try
        save(matPath, 'results', '-v7.3');
    catch
        save(matPath, 'results');
    end
end

if cfg.debug.verbose
    fprintf('birdcontour: accepted=%d/%d, points=%d\n', ...
        height(summaryTable), nIntervals, height(pointTable));
end

end

function cfg = merge_structs(defaults, overrides)

cfg = defaults;
if isempty(overrides)
    return;
end

names = fieldnames(overrides);
for i = 1:numel(names)
    name = names{i};
    value = overrides.(name);
    if isfield(cfg, name) && isstruct(cfg.(name)) && isstruct(value)
        cfg.(name) = merge_structs(cfg.(name), value);
    else
        cfg.(name) = value;
    end
end

end

function intervalTable = normalize_intervals(intervals, recordingDurationSec)

if isnumeric(intervals)
    if size(intervals, 2) ~= 2
        error('birdcontour:InvalidIntervals', ...
            'Numeric intervals must be N-by-2 [tStartSec tEndSec].');
    end
    tStartSec = double(intervals(:, 1));
    tEndSec = double(intervals(:, 2));
    n = size(intervals, 1);
    source = repmat("provided", n, 1);
    detectionScore = NaN(n, 1);
    entropyScore = NaN(n, 1);
    ridgeScore = NaN(n, 1);
    energyScore = NaN(n, 1);
elseif istable(intervals)
    names = intervals.Properties.VariableNames;
    iStart = find_name(names, {'tStartSec', 'startSec', 'startTimeSec', 'start'});
    iEnd = find_name(names, {'tEndSec', 'endSec', 'endTimeSec', 'end'});
    if isempty(iStart) || isempty(iEnd)
        if width(intervals) < 2
            error('birdcontour:InvalidIntervalTable', ...
                'Interval table needs tStartSec/tEndSec or at least two columns.');
        end
        iStart = 1;
        iEnd = 2;
    end
    tStartSec = double(intervals{:, iStart});
    tEndSec = double(intervals{:, iEnd});
    n = height(intervals);
    source = get_string_variable(intervals, 'source', repmat("provided", n, 1));
    detectionScore = get_numeric_variable(intervals, 'detectionScore', NaN(n, 1));
    entropyScore = get_numeric_variable(intervals, 'entropyScore', NaN(n, 1));
    ridgeScore = get_numeric_variable(intervals, 'ridgeScore', NaN(n, 1));
    energyScore = get_numeric_variable(intervals, 'energyScore', NaN(n, 1));
else
    error('birdcontour:InvalidIntervals', ...
        'intervals must be numeric N-by-2, a table, or empty.');
end

tStartSec = max(0, tStartSec(:));
tEndSec = min(recordingDurationSec, tEndSec(:));
valid = isfinite(tStartSec) & isfinite(tEndSec) & tEndSec > tStartSec;

if any(~valid)
    warning('birdcontour:InvalidIntervalsRemoved', ...
        'Removed %d invalid or out-of-range intervals.', nnz(~valid));
end

tStartSec = tStartSec(valid);
tEndSec = tEndSec(valid);
source = source(valid);
detectionScore = detectionScore(valid);
entropyScore = entropyScore(valid);
ridgeScore = ridgeScore(valid);
energyScore = energyScore(valid);

[tStartSec, order] = sort(tStartSec);
tEndSec = tEndSec(order);
source = source(order);
detectionScore = detectionScore(order);
entropyScore = entropyScore(order);
ridgeScore = ridgeScore(order);
energyScore = energyScore(order);

intervalIndex = (1:numel(tStartSec)).';
durationSec = tEndSec - tStartSec;
intervalTable = table(intervalIndex, tStartSec, tEndSec, durationSec, source, ...
    detectionScore, entropyScore, ridgeScore, energyScore);

end

function idx = find_name(names, candidates)

idx = [];
for i = 1:numel(candidates)
    j = find(strcmpi(names, candidates{i}), 1);
    if ~isempty(j)
        idx = j;
        return;
    end
end

end

function value = get_numeric_variable(T, name, fallback)

j = find(strcmpi(T.Properties.VariableNames, name), 1);
if isempty(j)
    value = fallback;
else
    value = double(T{:, j});
end
value = value(:);

end

function value = get_string_variable(T, name, fallback)

j = find(strcmpi(T.Properties.VariableNames, name), 1);
if isempty(j)
    value = fallback;
else
    value = string(T{:, j});
end
value = value(:);

end

function intervalTable = auto_detect_intervals(wavFile, info, cfg)

segmentSec = cfg.detect.scanSegmentSec;
hopSec = cfg.detect.scanHopSec;
startsSec = 0:hopSec:info.Duration;
startsSec = startsSec(startsSec < info.Duration);

if cfg.detect.keepLastPartial
    actualDuration = info.Duration - startsSec;
    keep = actualDuration >= cfg.detect.minLastDurationSec | ...
        (startsSec + segmentSec <= info.Duration + 1e-9);
else
    keep = startsSec + segmentSec <= info.Duration + 1e-9;
end
startsSec = startsSec(keep);

if isfinite(cfg.detect.maxScanSegments)
    startsSec = startsSec(1:min(numel(startsSec), cfg.detect.maxScanSegments));
end

candidateRows = repmat(empty_candidate(), 0, 1);
for iSegment = 1:numel(startsSec)
    [y, t0AbsSec] = read_fixed_segment( ...
        wavFile, info, startsSec(iSegment), segmentSec, cfg);
    [Pdb, Ppow, F, T] = compute_spectrogram(y, info.SampleRate, t0AbsSec, cfg);
    [E, ~] = enhance_spectrogram(Pdb, F, T, cfg);
    candidates = detect_candidates(E, Ppow, F, T, y, info.SampleRate, ...
        t0AbsSec, cfg);
    candidateRows = append_candidates(candidateRows, candidates);

    if cfg.debug.verbose && (mod(iSegment, 50) == 0 || iSegment == numel(startsSec))
        fprintf('birdcontour: detection scan %d/%d segments\n', ...
            iSegment, numel(startsSec));
    end
end

candidateRows = deduplicate_candidates(candidateRows, cfg.detect.dedupToleranceSec);

if isempty(candidateRows)
    intervalTable = empty_interval_table();
    return;
end

n = numel(candidateRows);
tStartSec = zeros(n, 1);
tEndSec = zeros(n, 1);
detectionScore = zeros(n, 1);
entropyScore = zeros(n, 1);
ridgeScore = zeros(n, 1);
energyScore = zeros(n, 1);
for i = 1:n
    tStartSec(i) = candidateRows(i).tStartSec;
    tEndSec(i) = candidateRows(i).tEndSec;
    detectionScore(i) = candidateRows(i).detectionScore;
    entropyScore(i) = candidateRows(i).entropyScore;
    ridgeScore(i) = candidateRows(i).ridgeScore;
    energyScore(i) = candidateRows(i).energyScore;
end

intervalIndex = (1:n).';
durationSec = tEndSec - tStartSec;
source = repmat("auto", n, 1);
intervalTable = table(intervalIndex, tStartSec, tEndSec, durationSec, ...
    source, detectionScore, entropyScore, ridgeScore, energyScore);

end

function T = empty_interval_table()

T = table('Size', [0 9], ...
    'VariableTypes', {'double', 'double', 'double', 'double', 'string', ...
    'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'intervalIndex', 'tStartSec', 'tEndSec', ...
    'durationSec', 'source', 'detectionScore', 'entropyScore', ...
    'ridgeScore', 'energyScore'});

end

function [y, t0AbsSec] = read_fixed_segment(wavFile, info, startSec, lengthSec, cfg)

fs = info.SampleRate;
targetSamples = max(1, round(lengthSec * fs));
segmentStartSample = floor(startSec * fs) + 1;
segmentEndSample = min(info.TotalSamples, segmentStartSample + targetSamples - 1);
padSamples = round(cfg.interval.readPadSec * fs);
readStart = max(1, segmentStartSample - padSamples);
readEnd = min(info.TotalSamples, segmentEndSample + padSamples);

x = audioread(wavFile, [readStart readEnd]);
x = preprocess_audio(x, cfg);
yPadded = bandpass_audio(x, fs, cfg);

cropStart = segmentStartSample - readStart + 1;
cropEnd = segmentEndSample - readStart + 1;
y = yPadded(cropStart:cropEnd);
if numel(y) < targetSamples
    y(numel(y)+1:targetSamples, 1) = 0;
elseif numel(y) > targetSamples
    y = y(1:targetSamples);
end

t0AbsSec = (segmentStartSample - 1) / fs;

end

function [ridge, diagnostic] = extract_one_interval(wavFile, info, tStartSec, tEndSec, cfg)

fs = info.SampleRate;
readStartSec = max(0, tStartSec - cfg.interval.readPadSec);
readEndSec = min(info.Duration, tEndSec + cfg.interval.readPadSec);
readStartSample = max(1, floor(readStartSec * fs) + 1);
readEndSample = min(info.TotalSamples, ceil(readEndSec * fs));

x = audioread(wavFile, [readStartSample readEndSample]);
x = preprocess_audio(x, cfg);
y = bandpass_audio(x, fs, cfg);
t0AbsSec = (readStartSample - 1) / fs;

[Pdb, ~, F, T, stftInfo] = compute_spectrogram(y, fs, t0AbsSec, cfg);
[E, Pvis, enhanceInfo] = enhance_spectrogram(Pdb, F, T, cfg);
ridge = track_ridge(E, F, T, tStartSec, tEndSec, cfg);

diagnostic = struct();
diagnostic.y = y;
diagnostic.fs = fs;
diagnostic.t0AbsSec = t0AbsSec;
diagnostic.F = F;
diagnostic.T = T;
diagnostic.Pdb = Pdb;
diagnostic.Pvis = Pvis;
diagnostic.E = E;
diagnostic.stftInfo = stftInfo;
diagnostic.enhanceInfo = enhanceInfo;

end

function x = preprocess_audio(x, cfg)

x = double(x);
x(~isfinite(x)) = 0;
if size(x, 2) > 1
    if strcmpi(cfg.audio.channelMode, 'first')
        x = x(:, 1);
    else
        x = mean(x, 2);
    end
end
x = x(:);
if ~isempty(x)
    x = x - mean(x);
end
x(~isfinite(x)) = 0;

end

function y = bandpass_audio(x, fs, cfg)

bandHz = sort(double(cfg.filter.bandHz(:).'));
nyquist = fs / 2;
bandHz(1) = max(1, bandHz(1));
bandHz(2) = min(0.99 * nyquist, bandHz(2));
if bandHz(1) >= bandHz(2)
    error('birdcontour:InvalidBand', ...
        'Invalid band %.1f-%.1f Hz for fs %.1f Hz.', ...
        bandHz(1), bandHz(2), fs);
end

[b, a] = butter(cfg.filter.order, bandHz / nyquist, 'bandpass');
try
    y = filtfilt(b, a, x);
catch ME
    error('birdcontour:FilterFailed', ...
        'Zero-phase bandpass filtering failed: %s', ME.message);
end
y = y(:);
y(~isfinite(y)) = 0;

end

function [Pdb, Ppow, F, TAbs, stftInfo] = compute_spectrogram(y, fs, t0AbsSec, cfg)

nwin = cfg.stft.winSamples;
if isempty(nwin) || nwin <= 0
    nwin = round(cfg.stft.winSec * fs);
end
nwin = max(128, round(nwin));
if numel(y) < nwin
    y(numel(y)+1:nwin, 1) = 0;
end

noverlap = round(cfg.stft.overlapFrac * nwin);
noverlap = min(max(0, noverlap), nwin - 1);
hop = nwin - noverlap;

if isempty(cfg.stft.nfft) || cfg.stft.nfft <= 0
    nfft = 2^nextpow2(cfg.stft.nfftFactor * nwin);
else
    nfft = round(cfg.stft.nfft);
end
nfft = max([nwin, cfg.stft.minNfft, nfft]);
nfft = min(cfg.stft.maxNfft, nfft);
nfft = max(nwin, nfft);

if exist('hann', 'file')
    window = hann(nwin, 'periodic');
else
    window = hanning(nwin, 'periodic');
end

% Required built-in MATLAB spectrogram implementation.
[S, F, TRel] = spectrogram(y, window, noverlap, nfft, fs);
TAbs = t0AbsSec + TRel(:).';
F = F(:);

keep = F >= cfg.stft.detectBandHz(1) & F <= cfg.stft.detectBandHz(2);
F = F(keep);
S = S(keep, :);
if isempty(F)
    error('birdcontour:EmptyFrequencyBand', ...
        'No STFT bins remain in %.1f-%.1f Hz.', ...
        cfg.stft.detectBandHz(1), cfg.stft.detectBandHz(2));
end

Ppow = abs(S).^2;
Pdb = 10 * log10(Ppow + eps);

stftInfo = struct();
stftInfo.nwin = nwin;
stftInfo.noverlap = noverlap;
stftInfo.hop = hop;
stftInfo.nfft = nfft;
stftInfo.winSec = nwin / fs;
stftInfo.hopSec = hop / fs;
stftInfo.dfHz = fs / nfft;
stftInfo.trueFreqResolutionHz = fs / nwin;

end

function [E, Pvis, info] = enhance_spectrogram(Pdb, F, T, cfg)

P = double(Pdb);
finiteValues = P(isfinite(P));
if isempty(finiteValues)
    P(:) = 0;
else
    P(~isfinite(P)) = median(finiteValues);
end

if exist('medfilt2', 'file')
    P = medfilt2(P, [3 3], 'symmetric');
end

dt = safe_spacing(T, 1);
df = safe_spacing(F, 1);
timeBgFrames = odd_number(cfg.enhance.timeBgSec / max(dt, eps));
freqBgBins = odd_number(cfg.enhance.freqBgHz / max(df, eps));

try
    timeFloor = movmedian(P, timeBgFrames, 2, 'omitnan');
catch
    timeFloor = movmedian(P, timeBgFrames, 2);
end
try
    freqFloor = movmedian(P, freqBgBins, 1, 'omitnan');
catch
    freqFloor = movmedian(P, freqBgBins, 1);
end

Et = P - timeFloor;
Ef = P - freqFloor;

colMed = median(P, 1);
colMad = 1.4826 * median(abs(bsxfun(@minus, P, colMed)), 1);
Zc = bsxfun(@rdivide, bsxfun(@minus, P, colMed), ...
    max(colMad, cfg.enhance.minMadDb));

rowMed = median(Et, 2);
rowMad = 1.4826 * median(abs(bsxfun(@minus, Et, rowMed)), 2);
Zt = bsxfun(@rdivide, bsxfun(@minus, Et, rowMed), ...
    max(rowMad, cfg.enhance.minMadDb));

freqMed = median(Ef(:));
freqMad = 1.4826 * median(abs(Ef(:) - freqMed));
Zf = (Ef - freqMed) ./ max(freqMad, cfg.enhance.minMadDb);

E = cfg.enhance.wTime * Zt + ...
    cfg.enhance.wFreq * Zf + ...
    cfg.enhance.wCol * Zc;
E(~isfinite(E)) = 0;

if cfg.enhance.gaussSize > 1
    E = gaussian_smooth(E, cfg.enhance.gaussSize, cfg.enhance.gaussSigma);
end

Pvis = P;
info = struct('timeBgFrames', timeBgFrames, ...
    'freqBgBins', freqBgBins, 'dtSec', dt, 'dfHz', df);

end

function Y = gaussian_smooth(X, sizeValue, sigma)

sizeValue = odd_number(sizeValue);
radius = floor(sizeValue / 2);
[xx, yy] = meshgrid(-radius:radius, -radius:radius);
kernel = exp(-(xx.^2 + yy.^2) / (2 * max(sigma, eps)^2));
kernel = kernel / sum(kernel(:));
Y = conv2(X, kernel, 'same');

end

function candidates = detect_candidates(E, Ppow, F, T, y, fs, t0AbsSec, cfg)

candidates = repmat(empty_candidate(), 0, 1);
if isempty(E) || numel(T) < 2
    return;
end

Ppow = double(Ppow);
Ppow(~isfinite(Ppow) | Ppow < 0) = 0;
columnSum = sum(Ppow, 1);
columnSum(columnSum <= 0) = eps;
Pnorm = bsxfun(@rdivide, Ppow, columnSum);

entropyScore = zeros(1, numel(T));
if cfg.detect.useEntropy
    H = -sum(Pnorm .* log2(Pnorm + eps), 1);
    Hnorm = H / max(log2(numel(F)), eps);
    rawEntropy = median(Hnorm) - Hnorm;
    entropyScore = robust_zscore(rawEntropy);
end

ridgeScore = zeros(1, numel(T));
if cfg.detect.useRidge
    ridgeScore = max(E, [], 1) - median(E, 1);
    dt = safe_spacing(T, cfg.detect.scoreSmoothSec);
    nSmooth = odd_number(cfg.detect.scoreSmoothSec / max(dt, eps));
    ridgeScore = smooth_vector(ridgeScore, nSmooth, 'median');
    ridgeScore = smooth_vector(ridgeScore, nSmooth, 'mean');
end

energyScore = zeros(1, numel(T));
if cfg.detect.useEnergy
    energyScore = waveform_energy_score(y, fs, T, t0AbsSec, cfg);
end

score = cfg.detect.wEntropy * entropyScore + ...
    cfg.detect.wRidge * ridgeScore + ...
    cfg.detect.wEnergy * energyScore;
score(~isfinite(score)) = 0;

active = score >= cfg.detect.scoreThreshold;
if cfg.detect.useEnergy && cfg.detect.useEnergyOrGate
    active = active | (energyScore >= cfg.detect.energyThreshold & ...
        ridgeScore >= cfg.detect.energyMinRidgeScore);
end

dt = safe_spacing(T, cfg.detect.scoreSmoothSec);
maxGapFrames = round(cfg.detect.maxGapSec / max(dt, eps));
active = fill_short_gaps(active, maxGapFrames);
runs = logical_runs(active);

for iRun = 1:size(runs, 1)
    i1 = runs(iRun, 1);
    i2 = runs(iRun, 2);
    durationSec = T(i2) - T(i1) + dt;
    if durationSec < cfg.detect.minDurationSec || ...
            durationSec > cfg.detect.maxDurationSec
        continue;
    end

    c = empty_candidate();
    c.tStartSec = T(i1);
    c.tEndSec = T(i2);
    c.detectionScore = mean(score(i1:i2));
    c.entropyScore = mean(entropyScore(i1:i2));
    c.ridgeScore = mean(ridgeScore(i1:i2));
    c.energyScore = mean(energyScore(i1:i2));
    candidates(end + 1, 1) = c; %#ok<AGROW>
end

end

function score = waveform_energy_score(y, fs, T, t0AbsSec, cfg)

windowSamples = max(3, round(cfg.detect.energyWinSec * fs));
backgroundSamples = max(windowSamples + 2, ...
    round(cfg.detect.energyBgSec * fs));
backgroundSamples = odd_number(backgroundSamples);

powerDb = 10 * log10(movmean(y .^ 2, windowSamples) + eps);
try
    backgroundDb = movmedian(powerDb, backgroundSamples, 'omitnan');
catch
    backgroundDb = movmedian(powerDb, backgroundSamples);
end
localEnergyDb = powerDb - backgroundDb;
z = robust_zscore(localEnergyDb);

sampleTime = t0AbsSec + (0:numel(y)-1).' / fs;
score = interp1(sampleTime, z(:), T(:), 'linear', 0).';

dt = safe_spacing(T, cfg.detect.energySmoothSec);
nSmooth = odd_number(cfg.detect.energySmoothSec / max(dt, eps));
score = smooth_vector(score, nSmooth, 'median');
score = smooth_vector(score, nSmooth, 'mean');
score(~isfinite(score)) = 0;

end

function candidates = deduplicate_candidates(candidates, toleranceSec)

if numel(candidates) < 2
    return;
end

[~, order] = sort([candidates.tStartSec]);
candidates = candidates(order);
kept = repmat(empty_candidate(), 0, 1);

for i = 1:numel(candidates)
    current = candidates(i);
    if isempty(kept)
        kept = current;
        continue;
    end

    previous = kept(end);
    overlap = min(previous.tEndSec, current.tEndSec) - ...
        max(previous.tStartSec, current.tStartSec);
    closeStart = abs(previous.tStartSec - current.tStartSec) <= toleranceSec;
    if overlap > 0 || closeStart
        if current.detectionScore > previous.detectionScore
            kept(end) = current;
        end
    else
        kept(end + 1, 1) = current; %#ok<AGROW>
    end
end

candidates = kept;

end

function c = empty_candidate()

c = struct('tStartSec', NaN, 'tEndSec', NaN, ...
    'detectionScore', NaN, 'entropyScore', NaN, ...
    'ridgeScore', NaN, 'energyScore', NaN);

end

function out = append_candidates(a, b)

if isempty(a)
    out = b(:);
elseif isempty(b)
    out = a(:);
else
    out = [a(:); b(:)];
end

end

function ridge = track_ridge(E, F, T, tStartSec, tEndSec, cfg)

ridge = empty_ridge();
t1 = max(T(1), tStartSec - cfg.interval.trackPadSec);
t2 = min(T(end), tEndSec + cfg.interval.trackPadSec);
timeMask = T >= t1 & T <= t2;
if nnz(timeMask) < 2
    return;
end

Ec = E(:, timeMask);
Tc = T(timeMask);
[rows, totalScore] = viterbi_dp(Ec, F, Tc, cfg);
if isempty(rows)
    return;
end

nTime = numel(Tc);
pathScores = Ec(sub2ind(size(Ec), rows(:).', 1:nTime));
rawFreqHz = F(rows);
freqHz = refine_subbin(Ec, F, rows);

dt = safe_spacing(Tc, 1);
medianFrames = odd_number(cfg.viterbi.smoothMedianSec / max(dt, eps));
meanFrames = odd_number(cfg.viterbi.smoothMeanSec / max(dt, eps));
freqHz = smooth_vector(freqHz, medianFrames, 'median');
freqHz = smooth_vector(freqHz, meanFrames, 'mean');

keep = Tc >= tStartSec & Tc <= tEndSec;
if nnz(keep) < 2
    return;
end

ridge.timeSecAbs = Tc(keep);
ridge.freqHz = freqHz(keep);
ridge.rawFreqHz = rawFreqHz(keep);
ridge.pathScores = pathScores(keep);
ridge.confidence = safe_mean(ridge.pathScores);
ridge.scoreMean = ridge.confidence;
ridge.scoreMax = max(ridge.pathScores);
ridge.totalDpScore = totalScore;
ridge.numFrames = nnz(keep);
ridge.valid = isfinite(ridge.confidence) && ...
    ridge.confidence >= cfg.viterbi.minConfidence;

end

function ridge = empty_ridge()

ridge = struct('valid', false, 'timeSecAbs', [], 'freqHz', [], ...
    'rawFreqHz', [], 'pathScores', [], 'confidence', NaN, ...
    'scoreMean', NaN, 'scoreMax', NaN, 'totalDpScore', NaN, ...
    'numFrames', 0);

end

function [pathRows, finalScore] = viterbi_dp(Ec, F, T, cfg)

[nFreq, nTime] = size(Ec);
pathRows = [];
finalScore = NaN;
if nFreq == 0 || nTime == 0
    return;
end

emission = double(Ec);
finiteValues = emission(isfinite(emission));
if isempty(finiteValues)
    return;
end
emission(~isfinite(emission)) = min(finiteValues) - 10;

if nTime == 1
    [finalScore, row] = max(emission(:, 1));
    pathRows = row;
    return;
end

df = safe_spacing(F, 1);
dt = safe_spacing(T, 1);
maxJump = ceil(cfg.viterbi.maxSlopeHzPerSec * dt / max(df, eps));
maxJump = max(1, min(nFreq - 1, maxJump));

scorePrevious = emission(:, 1);
backPointer = zeros(nFreq, nTime, 'uint16');

for iTime = 2:nTime
    bestTransition = -inf(nFreq, 1);
    bestPrevious = zeros(nFreq, 1, 'uint16');

    for jump = -maxJump:maxJump
        penalty = cfg.viterbi.jumpPenaltyLinear * abs(jump) + ...
            cfg.viterbi.jumpPenaltyQuadratic * jump^2;
        transition = -inf(nFreq, 1);

        if jump < 0
            currentRows = 1:(nFreq + jump);
            previousRows = currentRows - jump;
        elseif jump > 0
            currentRows = (1 + jump):nFreq;
            previousRows = currentRows - jump;
        else
            currentRows = 1:nFreq;
            previousRows = currentRows;
        end

        transition(currentRows) = scorePrevious(previousRows) - penalty;
        better = transition > bestTransition;
        bestTransition(better) = transition(better);
        previousVector = zeros(nFreq, 1, 'uint16');
        previousVector(currentRows) = uint16(previousRows);
        bestPrevious(better) = previousVector(better);
    end

    scorePrevious = emission(:, iTime) + bestTransition;
    backPointer(:, iTime) = bestPrevious;
end

[finalScore, row] = max(scorePrevious);
pathRows = zeros(1, nTime);
pathRows(end) = row;
for iTime = nTime:-1:2
    row = double(backPointer(row, iTime));
    if row < 1
        row = pathRows(iTime);
    end
    pathRows(iTime - 1) = row;
end

pathRows = max(1, min(nFreq, round(pathRows)));

end

function freqHz = refine_subbin(Ec, F, rows)

rows = rows(:).';
freqHz = F(rows).';
df = safe_spacing(F, 0);
if df <= 0
    return;
end

for iTime = 1:numel(rows)
    row = rows(iTime);
    if row <= 1 || row >= numel(F)
        continue;
    end
    yMinus = Ec(row - 1, iTime);
    yZero = Ec(row, iTime);
    yPlus = Ec(row + 1, iTime);
    denominator = yMinus - 2 * yZero + yPlus;
    if isfinite(denominator) && abs(denominator) > eps
        delta = 0.5 * (yMinus - yPlus) / denominator;
        delta = max(-0.75, min(0.75, delta));
        freqHz(iTime) = F(row) + delta * df;
    end
end

end

function contour = make_contour(fileName, filePath, intervalRow, ridge, contourIndex)

freqHz = ridge.freqHz(:).';
timeSec = ridge.timeSecAbs(:).';

contour = empty_contours();
contour(1).fileName = fileName;
contour(1).filePath = filePath;
contour(1).intervalIndex = intervalRow.intervalIndex;
contour(1).contourIndex = contourIndex;
contour(1).source = char(intervalRow.source);
contour(1).requestedStartSec = intervalRow.tStartSec;
contour(1).requestedEndSec = intervalRow.tEndSec;
contour(1).detectionScore = intervalRow.detectionScore;
contour(1).entropyScore = intervalRow.entropyScore;
contour(1).ridgeScore = intervalRow.ridgeScore;
contour(1).energyScore = intervalRow.energyScore;
contour(1).timeSecAbs = timeSec;
contour(1).freqHz = freqHz;
contour(1).rawFreqHz = ridge.rawFreqHz(:).';
contour(1).pathScores = ridge.pathScores(:).';
contour(1).tStartSec = timeSec(1);
contour(1).tEndSec = timeSec(end);
contour(1).durationSec = max(0, timeSec(end) - timeSec(1));
contour(1).fStartHz = freqHz(1);
contour(1).fEndHz = freqHz(end);
contour(1).fMinHz = min(freqHz);
contour(1).fMaxHz = max(freqHz);
contour(1).fMedianHz = median(freqHz);
contour(1).bandwidthHz = max(freqHz) - min(freqHz);
contour(1).confidence = ridge.confidence;
contour(1).scoreMean = ridge.scoreMean;
contour(1).scoreMax = ridge.scoreMax;
contour(1).method = ...
    'bandpass_builtin_spectrogram_robust_constrained_viterbi';

end

function contours = empty_contours()

contours = struct( ...
    'fileName', {}, 'filePath', {}, 'intervalIndex', {}, ...
    'contourIndex', {}, 'source', {}, 'requestedStartSec', {}, ...
    'requestedEndSec', {}, 'detectionScore', {}, 'entropyScore', {}, ...
    'ridgeScore', {}, 'energyScore', {}, 'timeSecAbs', {}, ...
    'freqHz', {}, 'rawFreqHz', {}, 'pathScores', {}, ...
    'tStartSec', {}, 'tEndSec', {}, 'durationSec', {}, ...
    'fStartHz', {}, 'fEndHz', {}, 'fMinHz', {}, 'fMaxHz', {}, ...
    'fMedianHz', {}, 'bandwidthHz', {}, 'confidence', {}, ...
    'scoreMean', {}, 'scoreMax', {}, 'method', {});

end

function out = append_contours(a, b)

if isempty(a)
    out = b(:);
elseif isempty(b)
    out = a(:);
else
    out = [a(:); b(:)];
end

end

function T = make_summary_table(contours)

names = {'fileName', 'intervalIndex', 'contourIndex', 'source', ...
    'requestedStartSec', 'requestedEndSec', 'tStartSec', 'tEndSec', ...
    'durationSec', 'fStartHz', 'fEndHz', 'fMinHz', 'fMaxHz', ...
    'fMedianHz', 'bandwidthHz', 'confidence', 'detectionScore', ...
    'entropyScore', 'ridgeScore', 'energyScore', 'scoreMean', ...
    'scoreMax', 'method'};
types = {'string', 'double', 'double', 'string', ...
    'double', 'double', 'double', 'double', 'double', 'double', ...
    'double', 'double', 'double', 'double', 'double', 'double', ...
    'double', 'double', 'double', 'double', 'double', 'double', 'string'};

if isempty(contours)
    T = table('Size', [0 numel(names)], ...
        'VariableTypes', types, 'VariableNames', names);
    return;
end

n = numel(contours);
fileName = strings(n, 1);
intervalIndex = zeros(n, 1);
contourIndex = zeros(n, 1);
source = strings(n, 1);
requestedStartSec = zeros(n, 1);
requestedEndSec = zeros(n, 1);
tStartSec = zeros(n, 1);
tEndSec = zeros(n, 1);
durationSec = zeros(n, 1);
fStartHz = zeros(n, 1);
fEndHz = zeros(n, 1);
fMinHz = zeros(n, 1);
fMaxHz = zeros(n, 1);
fMedianHz = zeros(n, 1);
bandwidthHz = zeros(n, 1);
confidence = zeros(n, 1);
detectionScore = NaN(n, 1);
entropyScore = NaN(n, 1);
ridgeScore = NaN(n, 1);
energyScore = NaN(n, 1);
scoreMean = zeros(n, 1);
scoreMax = zeros(n, 1);
method = strings(n, 1);

for i = 1:n
    c = contours(i);
    fileName(i) = string(c.fileName);
    intervalIndex(i) = c.intervalIndex;
    contourIndex(i) = c.contourIndex;
    source(i) = string(c.source);
    requestedStartSec(i) = c.requestedStartSec;
    requestedEndSec(i) = c.requestedEndSec;
    tStartSec(i) = c.tStartSec;
    tEndSec(i) = c.tEndSec;
    durationSec(i) = c.durationSec;
    fStartHz(i) = c.fStartHz;
    fEndHz(i) = c.fEndHz;
    fMinHz(i) = c.fMinHz;
    fMaxHz(i) = c.fMaxHz;
    fMedianHz(i) = c.fMedianHz;
    bandwidthHz(i) = c.bandwidthHz;
    confidence(i) = c.confidence;
    detectionScore(i) = c.detectionScore;
    entropyScore(i) = c.entropyScore;
    ridgeScore(i) = c.ridgeScore;
    energyScore(i) = c.energyScore;
    scoreMean(i) = c.scoreMean;
    scoreMax(i) = c.scoreMax;
    method(i) = string(c.method);
end

T = table(fileName, intervalIndex, contourIndex, source, ...
    requestedStartSec, requestedEndSec, tStartSec, tEndSec, durationSec, ...
    fStartHz, fEndHz, fMinHz, fMaxHz, fMedianHz, bandwidthHz, ...
    confidence, detectionScore, entropyScore, ridgeScore, energyScore, ...
    scoreMean, scoreMax, method);

end

function T = make_point_table(contours)

names = {'fileName', 'intervalIndex', 'contourIndex', 'pointIndex', ...
    'timeSec', 'freqHz', 'rawFreqHz', 'pathScore', 'confidence'};
types = {'string', 'double', 'double', 'double', 'double', ...
    'double', 'double', 'double', 'double'};

if isempty(contours)
    T = table('Size', [0 numel(names)], ...
        'VariableTypes', types, 'VariableNames', names);
    return;
end

counts = arrayfun(@(c) numel(c.timeSecAbs), contours);
nPoints = sum(counts);
fileName = strings(nPoints, 1);
intervalIndex = zeros(nPoints, 1);
contourIndex = zeros(nPoints, 1);
pointIndex = zeros(nPoints, 1);
timeSec = zeros(nPoints, 1);
freqHz = zeros(nPoints, 1);
rawFreqHz = zeros(nPoints, 1);
pathScore = zeros(nPoints, 1);
confidence = zeros(nPoints, 1);

k = 0;
for i = 1:numel(contours)
    c = contours(i);
    n = numel(c.timeSecAbs);
    idx = k + (1:n);
    fileName(idx) = string(c.fileName);
    intervalIndex(idx) = c.intervalIndex;
    contourIndex(idx) = c.contourIndex;
    pointIndex(idx) = (1:n).';
    timeSec(idx) = c.timeSecAbs(:);
    freqHz(idx) = c.freqHz(:);
    rawFreqHz(idx) = c.rawFreqHz(:);
    pathScore(idx) = c.pathScores(:);
    confidence(idx) = c.confidence;
    k = k + n;
end

T = table(fileName, intervalIndex, contourIndex, pointIndex, ...
    timeSec, freqHz, rawFreqHz, pathScore, confidence);

end

function save_interval_figure(diagnostic, ridge, intervalRow, figuresDir, prefix, cfg)

intervalIndex = intervalRow.intervalIndex;
baseName = sprintf('%s_interval%04d', prefix, intervalIndex);
pngPath = fullfile(figuresDir, [baseName '.png']);
figPath = fullfile(figuresDir, [baseName '.fig']);

figureHandle = figure('Visible', cfg.figures.visible, 'Color', 'w', ...
    'Position', [100 100 1400 950]);
cleanup = onCleanup(@() close_if_valid(figureHandle)); %#ok<NASGU>

tWave = diagnostic.t0AbsSec + (0:numel(diagnostic.y)-1) / diagnostic.fs;
x1 = max(tWave(1), intervalRow.tStartSec - cfg.figures.contextSec);
x2 = min(tWave(end), intervalRow.tEndSec + cfg.figures.contextSec);

subplot(3, 1, 1);
plot(tWave, diagnostic.y, 'k-', 'LineWidth', 0.6);
xlim([x1 x2]);
hold on;
xline(intervalRow.tStartSec, '--b');
xline(intervalRow.tEndSec, '--b');
hold off;
grid on;
ylabel('Amplitude');
title(sprintf('Interval %d: %.6f-%.6f s', intervalIndex, ...
    intervalRow.tStartSec, intervalRow.tEndSec));

subplot(3, 1, 2);
imagesc(diagnostic.T, diagnostic.F / 1000, diagnostic.Pdb);
axis xy;
xlim([x1 x2]);
ylim(cfg.figures.plotBandHz / 1000);
maxDb = max(diagnostic.Pdb(:));
if isfinite(maxDb)
    clim([maxDb - cfg.figures.dynamicRangeDb, maxDb]);
end
colorbar;
ylabel('Frequency (kHz)');
title('Built-in spectrogram power (dB)');

subplot(3, 1, 3);
imagesc(diagnostic.T, diagnostic.F / 1000, diagnostic.E);
axis xy;
xlim([x1 x2]);
ylim(cfg.figures.plotBandHz / 1000);
hold on;
if ridge.valid
    plot(ridge.timeSecAbs, ridge.freqHz / 1000, ...
        'w-', 'LineWidth', 2.0);
    plot(ridge.timeSecAbs, ridge.freqHz / 1000, ...
        'k-', 'LineWidth', 0.7);
end
hold off;
colorbar;
xlabel('Time from recording start (s)');
ylabel('Frequency (kHz)');
title(sprintf('Enhanced TF map and contour; confidence=%.3f', ...
    ridge.confidence));

colormap(figureHandle, parula(256));

if cfg.figures.savePng
    print(figureHandle, pngPath, '-dpng', ...
        sprintf('-r%d', cfg.figures.resolutionDpi));
end
if cfg.figures.saveFig
    savefig(figureHandle, figPath);
end

end

function close_if_valid(handle)

if isgraphics(handle)
    close(handle);
end

end

function y = robust_zscore(x)

x = double(x);
medianValue = median(x(isfinite(x)));
if isempty(medianValue) || ~isfinite(medianValue)
    y = zeros(size(x));
    return;
end
madValue = 1.4826 * median(abs(x(isfinite(x)) - medianValue));
y = (x - medianValue) ./ max(madValue, eps);
y(~isfinite(y)) = 0;

end

function y = smooth_vector(x, windowLength, method)

windowLength = max(1, round(windowLength));
if windowLength <= 1 || numel(x) <= 1
    y = x;
elseif strcmpi(method, 'median')
    try
        y = movmedian(x, windowLength, 'omitnan');
    catch
        y = movmedian(x, windowLength);
    end
else
    try
        y = movmean(x, windowLength, 'omitnan');
    catch
        y = movmean(x, windowLength);
    end
end

end

function active = fill_short_gaps(active, maxGapFrames)

active = logical(active(:).');
if maxGapFrames <= 0 || ~any(active)
    return;
end

gaps = logical_runs(~active);
for i = 1:size(gaps, 1)
    first = gaps(i, 1);
    last = gaps(i, 2);
    bounded = first > 1 && last < numel(active) && ...
        active(first - 1) && active(last + 1);
    if bounded && (last - first + 1) <= maxGapFrames
        active(first:last) = true;
    end
end

end

function runs = logical_runs(x)

x = logical(x(:).');
edges = diff([false x false]);
starts = find(edges == 1);
ends = find(edges == -1) - 1;
runs = [starts(:) ends(:)];

end

function value = odd_number(value)

value = max(1, round(value));
if mod(value, 2) == 0
    value = value + 1;
end

end

function spacing = safe_spacing(x, fallback)

if numel(x) > 1
    spacing = median(diff(x));
else
    spacing = fallback;
end
if ~isfinite(spacing) || spacing <= 0
    spacing = fallback;
end

end

function value = safe_mean(x)

x = x(isfinite(x));
if isempty(x)
    value = NaN;
else
    value = mean(x);
end

end

function safe_mkdir(folder)

if ~exist(folder, 'dir')
    mkdir(folder);
end

end

function name = sanitize_name(name)

name = char(string(name));
name = regexprep(name, '[^A-Za-z0-9_.-]+', '_');

end
