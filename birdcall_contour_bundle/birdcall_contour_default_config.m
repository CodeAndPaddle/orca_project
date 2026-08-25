function cfg = birdcall_contour_default_config()
%BIRDCALL_CONTOUR_DEFAULT_CONFIG Configuration for the generated nonlinear chirp.

cfg.output.dir = '';
cfg.output.prefix = 'synthetic_dolphin';
cfg.output.writeCsv = true;
cfg.output.saveMat = true;

cfg.audio.channelMode = 'mean';

% Must exceed twice the highest analyzed frequency.
% The generated audio uses 48 kHz.
cfg.audio.minSampleRateHz = 24000;

% Generated contour spans 4–10 kHz; margins accommodate STFT spreading.
cfg.filter.bandHz = [3500 10500];
cfg.filter.order = 6;

% At 48 kHz:
%   window = 5.33 ms
%   hop    = about 0.80 ms with 85% overlap
cfg.stft.winSamples = 256;
cfg.stft.winSec = [];
cfg.stft.overlapFrac = 0.85;
cfg.stft.nfft = [];
cfg.stft.nfftFactor = 8;
cfg.stft.minNfft = 4096;
cfg.stft.maxNfft = 16384;
cfg.stft.detectBandHz = [3500 10500];

% Background scales suit the one-second synthetic whistle.
cfg.enhance.timeBgSec = 0.10;
cfg.enhance.freqBgHz = 500;
cfg.enhance.minMadDb = 0.75;
% Column contrast dominates for the continuous one-second synthetic whistle.
cfg.enhance.wTime = 0.20;
cfg.enhance.wFreq = 0.10;
cfg.enhance.wCol = 0.70;
cfg.enhance.gaussSize = 0;
cfg.enhance.gaussSigma = 0.75;

% Suitable when the input file contains only the short synthetic signal.
cfg.detect.scanSegmentSec = 0.25;
cfg.detect.scanHopSec = 0.10;
cfg.detect.keepLastPartial = true;
cfg.detect.minLastDurationSec = 0.10;
cfg.detect.maxScanSegments = inf;

% Avoid smoothing away the nonlinear curvature.
cfg.detect.scoreSmoothSec = 0.006;

cfg.detect.useEntropy = true;
cfg.detect.useRidge = true;
cfg.detect.useEnergy = true;
cfg.detect.wEntropy = 0.20;
cfg.detect.wRidge = 0.40;
cfg.detect.wEnergy = 0.40;

% Start with these values and lower them only if the call is missed.
cfg.detect.scoreThreshold = 2.0;
cfg.detect.energyWinSec = 0.008;
cfg.detect.energyBgSec = 0.10;
cfg.detect.energySmoothSec = 0.006;
cfg.detect.energyThreshold = 2.2;
cfg.detect.energyMinRidgeScore = 0.5;
cfg.detect.useEnergyOrGate = true;

cfg.detect.maxGapSec = 0.015;
cfg.detect.minDurationSec = 0.12;
cfg.detect.maxDurationSec = 0.30;
cfg.detect.dedupToleranceSec = 0.020;

cfg.interval.readPadSec = 0.05;
cfg.interval.trackPadSec = 0.015;
cfg.interval.warnDurationSec = 1.25;

% The generated contour reaches approximately 121 kHz/s.
% Allow some margin above that theoretical maximum.
cfg.viterbi.maxSlopeHzPerSec = 150000;

% Slightly relaxed penalties help follow the steep end of the contour.
cfg.viterbi.jumpPenaltyLinear = 0.010;
cfg.viterbi.jumpPenaltyQuadratic = 0.0005;
cfg.viterbi.minConfidence = 1.2;

% The original 12–15 ms smoothing could flatten the curved trajectory.
cfg.viterbi.smoothMedianSec = 0.004;
cfg.viterbi.smoothMeanSec = 0.005;

cfg.figures.savePng = true;
cfg.figures.saveFig = false;
cfg.figures.saveRejected = false;
cfg.figures.maxFigures = 100;
cfg.figures.contextSec = 0.10;
cfg.figures.plotBandHz = [3000 11000];
cfg.figures.dynamicRangeDb = 55;
cfg.figures.resolutionDpi = 250;
cfg.figures.visible = 'off';

cfg.debug.verbose = true;
end
