function [sig_out] = take_spectrogram_peaks(sig_in, Fs)

s=spectrogram(sig_in,256,120,1024,Fs, 'yaxis');

% take all the high signals in each time slot
peaks=zeros(size(s));
for i=1:size(s,2)
    [pks,locs,w,p] = findpeaks(abs(s(:,i)));
    peaks(locs,i)=abs(s(locs,i));
end
sig_out=abs(peaks);

end

