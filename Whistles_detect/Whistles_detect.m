function [Res] = Whistles_detect(sig, mode, T, Fs)
% WHISTLES_DETECT(SIG, mode, T, Fs, Pfa) - runs an NLMS filter on the input
% SIG (after a BPF, for marine noise), then uses an energy detector on the
% output. The detections are used to run segments of the original signal
% (without the marine noise) through an entropy detector and Viterbi
% algorithm for feature extraction. The resulting struct RES, detected
% segments, a flag indicating if the segment was in the time interval T
% specifying a tagged whistle, and the extracted features.
%   INPUT - 
%       SIG -   the input signal
%       MODE -  the mode of operation - if set to 'train' SEG will contain
%               only segments that intersect with the time interval T.
%       T -     Vector desiganting a tagged interval
%       Fs -    Sampling frequency of the signal
% 
%   OUTPUT - 
%       RES -   A struct withfields:
%               RES.SEG -   The detected segment
%               RES.FEAT -  The extarcted features as a cell vector with
%               entries in this order:
%                   [Start_Frequency	End_Frequency	Duration	
%                       Min_Frequency	Max_Frequency	n_Inflection_Point	
%                       n_of_steps	Start_Time	End_Time   SNR]
%               RES.DET -   A flag indicating if the detected segment
%                           intersects the tagged time interval T.

    run('Parameters.m')      %get runtime parameters
    mu_temp = 0.1;
    %% Simple BP filter
    Rx = dolphin_filt(sig(:,1),Fs);


    %% Filter the signal
    h_prev = zeros(N,1);h_prev(1)=1;  %%%%initialize NLMS impulse response - Ilan
    h_curr = h_prev;
    Rx_res = NLMS_filter(Rx,bufsize,mu_temp,N,h_prev,h_curr,two_filters);
    % normalize both vectors
    Rx = Rx/rms(Rx);
    Rx_res = Rx_res/rms(Rx_res);
                    
    Detect = energy_detector6(Pfa,Rx,WinLen, noise_time,Fs, 6, 0.6);
    %Detect = energy_detector6(Pfa,Rx_res,WinLen, noise_time,Fs, 6, 0.6);

    start_blk = zeros(size(Detect));
    end_blk = zeros(size(Detect));
    Detect = [zeros(1,length(Pfa));Detect(1:end-1,:);zeros(1,length(Pfa))];
    arg = Detect(1:end-1,:) - Detect(2:end,:);
    start_blk(arg==-1) = 1;
    end_blk(arg==1) = 1;
    Res=[];
    det=[];
    for pfa_ind = 1:length(Pfa)
        block = find(start_blk(:,pfa_ind));
        start_ind = max((noise_time - prebuff)*Fs + block*WinLen,1);
        block = find(end_blk(:,pfa_ind));
        end_ind = min((noise_time + postbuff)*Fs + block*WinLen, length(Rx));
        
        for det_num = 1:length(start_ind)
            seg = Rx(start_ind(det_num):end_ind(det_num));
            if length(seg)>(prebuff + postbuff)
                [whistle_times_mat, features] = dolphin_project( seg , Fs,...
                    min_window ,...% in sec
                    extra_time_window,...% in sec
                    th_c,...%[];%   if empty , correlator is adaptive
                    max_trans, ...,% Viterbi frequency bins
                    Pfa(pfa_ind), ...% probability of noise after Entropy detector
                    corr_wind_size ,...% sliding window size for correlation
                    detection_wind_size,...% detector window size
                    max_window_size,... %in sec; eliminates extreme false detections
                    NumPoss,...% number of viterbi paths - not used in this version
                    th_v,...% % confidence present it is a whistle in viterbi algo
                    []); % results directory
                
                whistle_times = whistle_times_mat + start_ind(det_num)/Fs;
                if ~isempty(whistle_times)
                    [N,~] = size(whistle_times);
                    for ii = 1:N
                        Res(length(Res)+1).seg = Rx(whistle_times(1)*Fs:whistle_times(2)*Fs);
                        Res(length(Res)).feat = features(ii + 1,:);
                        if any(T)
                            Res(length(Res)).det = whistle_times(1,ii)<T(2) & whistle_times(ii,2) > T(1);
                            det = [det,Res(length(Res)).det];
                        end
                    end
                end
            end
        end
        
    end
    if strcmp(mode,'train')
        Res = Res(det==1);
    end
    
% end