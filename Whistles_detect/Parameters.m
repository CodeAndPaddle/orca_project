
    %% Parameters
    noise_time=0.2;         %Length of noise at start of sample (in seconds)
    prebuff = 1;            %Length of buffer added prior to detection (in seconds)
    postbuff = 4;           %Length of buffer added after detection (in seconds)
    %% NLMS and other parameters
    N=60;                   % Length of NLMS impulse responce (in bins - not time)
    bufsize=1000;           % Length of refference buffer for NLMS (in bins - not time)
    mu=0.1;                 % Step size of NLMS filter
    two_filters = false;    % Flag for use of consecutive NLMS filters (default is false)
    WinLen=300;             % Window size for energy detector (in bins - not time)
    Pfa = 1e-3;             % Pfa threshold for energy detector
    
    %% default constants for Entropy and Viterbi
    min_window = 0.2 ;              % in sec
    extra_time_window = 0.5 ;       % in sec
    th_c = 0.02 ;                   % [];   % if empty, correlator is adaptive
    max_trans = 8;                  % TBD
    % p_fa = 0.2;                     % 0.1; % probability of noise after Entropy detector
    corr_wind_size = floor(Fs/5e3); % sliding window size for correlation
    detection_wind_size = 5757;     % detector window size
    max_window_size = 5;            % in sec; eliminates extreme false detections
    NumPoss=5;                      % number of viterbi paths - not used in this version
    th_v=0.90;                      % confidence present it is a whistle in viterbi algo
