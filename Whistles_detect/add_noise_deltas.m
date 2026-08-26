function [sig_out] = add_noise_deltas(sig,Fs,deltas_len,delta_locations)
% Adds unbounded spectraly noise in sepecified legngth

sig_dur = length(sig)-Fs;
num_delta = floor(abs(rand(1))*20*length(sig)/Fs);
delta_locations = floor(rand(num_delta,1)*sig_dur);

sig_out=zeros(size(sig));
sig_out = sig + sig_out;
% parfor i=1:1:num_delta
sig_out(delta_locations-floor(deltas_len*Fs)/2:delta_locations+floor(deltas_len*Fs)/2)...
    = max(sig)*10;
% end

end