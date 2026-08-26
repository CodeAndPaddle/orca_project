function [sig_out] = dolphin_filt(sig,Fs)



% BandPass Filter :
% Dolphin whistles are typically found in range ~= 5-24K Herz
% Under 5K we generally get man made noise -such as motors at very high
% energy
% 
maxfreq=2*24000/Fs;
minfreq = 2*5000/Fs;

if (maxfreq>=1)
    maxfreq=0.99;
end
BPF = fir1(100,[minfreq maxfreq]);
sig_out = filter(BPF,1,sig);
% %BPF Constants
% maxfreq=2*24000/Fs;
% minfreq=2*5000/Fs;
% %BPF
% if (maxfreq>1)
%     maxfreq=0.99;
% end
% 
% sig_out=bandpass(sig,[minfreq maxfreq]);

end

