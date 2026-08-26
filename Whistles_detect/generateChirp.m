function [t,chirp_wave] = generateChirp(Fs,freq_start,freq_end,duration)
% utility function to create chirps
ts =(1/Fs);
t = 0:ts:duration;
try
    chirp_wave = chirp(t,freq_start,duration,freq_end)';
catch
    disp('err');
end

