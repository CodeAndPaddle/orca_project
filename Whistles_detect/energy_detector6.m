function [Detect] = energy_detector6(Pfa,sig,WinLen,...
    noise_time, Fs, MinLen, MinRatio)
%
%function [noise_flag_ind] = energy_detector(Pfa,sig,WinLen,noise_time,Fs)
%
%   This function detects the presence of a signal in a sliding window (of
%   size WinLen) along the signal. The windowed signal is mean squared to 
%   determine its instantanious power. This instantanious reading is then
%   normalized according to chi squared distribution, using the noise mean
%   and variance (this noise is taken from the beginning of the total
%   signal (defined by noise_time). The normalized value - d - is then
%   compared to d_t, the detection threshold (derived from Pfa), to be
%   input into a result vector - noise_flag_ind, that indicates for every
%   time instance the detection of a signal according to these parameters
%   and model.
%
%   Input:
%           Pfa -           A value corresponding to the probability of
%                           false alarm. This value is used to derive the
%                           detection threshold via the inverse normal
%                           distribution function.
%           sig -           Signal on which the detector is run
%           WinLen -        Length of window for detection
%           noise_time -    Time (in seconds) from start of signal that is
%                           assumed to be noise. Used to determine noise
%                           mean and variance for power centering and
%                           normalization.
%           Fs -            Sampling frequency (used to convert noise_time
%                           from seconds to cell number)
%
%   output:
%          noise_flag_ind - A vector indicating the detection result for
%                           each windowed part of the signal.
%
%   Created by Eitan Ovrutski, October 2018
delta_time=2*WinLen/Fs;
%calculate energy detection threshold
%d_t = sqrt(2)*erfcinv(2*Pf);  %=norminv(pf)
d_t =-norminv(Pfa);
%d_t is the threshold for the neyman pearson test
%{

    dNoise = zeros(1, size(DenoisedValMat,2)-WinLen);

    for ind = WinLen+1: size(DenoisedValMat,2)

        dNoise(ind-WinLen) = sum((DenoisedValMat(ind-WinLen+1:ind)).^2) / WinLen;

    end
%}
%dNoise: section of noise to estimate parameters

%%%%%%%%%
NoiseVec=sig(1:noise_time*Fs);        %Get noise from start of signal
NumBlocks = floor(length(NoiseVec)/WinLen);

CurrentWinEn = [];
for ind = 0: NumBlocks-1
    CurrentWin = NoiseVec(ind*WinLen+1: (ind+1)*WinLen);
    CurrentWinEn = [CurrentWinEn, sum(CurrentWin.^2)/WinLen];
end
n_std = std(CurrentWinEn,1);
n_mean = mean(CurrentWinEn);
n_std=std(NoiseVec,1);
n_mean=mean(NoiseVec);
RxSig = sig(noise_time*Fs+1: end);
NumBlocks = floor(length(RxSig)/WinLen);
CurrentWinEs = [];
for ind = 0: NumBlocks-1
    CurrentWin = RxSig(ind*WinLen+1: (ind+1)*WinLen);
    CurrentWin=(CurrentWin-n_mean)/n_std;
    CurrentWinEs = [CurrentWinEs, sum(CurrentWin.^2)/WinLen];
end
% CurrentWinEs = (CurrentWinEs - n_mean) / n_std; 
Detect = CurrentWinEs' > d_t;
noise_flag_ind = zeros(length(Pfa),1);
detect_flag_ind = zeros(length(Pfa),1);


for Pfa_ind=1:length(Pfa)
    loc = find(Detect(:,Pfa_ind));

    if nargin > 5
        loc = FilterLoc(loc, Detect(:,Pfa_ind), MinLen, MinRatio);
    end
    temp = zeros(length(Detect(:,Pfa_ind)),1);
    temp(loc) = 1;
    Detect(:,Pfa_ind) = temp;
%     if any(loc)
%         if any((loc(1)-1)*WinLen + 1 < (StartPos - noise_time - delta_time)*Fs | (loc(end)-1)*WinLen > (EndPos-noise_time+delta_time)*Fs)
%             noise_flag_ind(Pfa_ind) = 1;
%         end
%         if any((loc-1)*WinLen + 1 > (StartPos - noise_time)*Fs & (loc-1)*WinLen < (EndPos-noise_time)*Fs)
% %         if any(find((loc-1)*WinLen + 1 > (StartPos - noise_time)*Fs & (loc-1)*WinLen < (EndPos-noise_time)*Fs))
%             detect_flag_ind(Pfa_ind) = 1;
%         end
%     end
end


% t = 1: length(CurrentWinEs);
% figure;
% subplot(2,1,1);
% plot(RxSig);
% subplot(2,1,2);
% plot(t, CurrentWinEs);
% hold on;
% plot(t(loc),CurrentWinEs(loc));

return

function loc = FilterLoc(InitLoc, Detect, MinLen, MinRatio)

    Detect_filt = zeros(size(Detect));
    for loc_ind = 1:length(InitLoc)
        curr_ind = InitLoc(loc_ind);
        curr_win = Detect(max(1, curr_ind - MinLen/2):(min(length(Detect), curr_ind + MinLen/2 - 1)));
        Detect_filt(curr_ind)= mean(curr_win) > MinRatio;
    end
    loc=find(Detect_filt);

return