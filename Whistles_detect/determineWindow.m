function [Entropy_sig_out,start_vec,end_vec] = ...
    determineWindow(extra_time_window,Entropy_sig_out,start_vec,end_vec,Fs)
% Extends whistle to 'max_window' size.
% 
% extra_time_window - window size
% Entropy_sig_out - vector needed adjustments
% Start_vec - indexes of whistles
% End_vec - indexes of whistles
% Fs - sampling frequency
Start_vec_sec = start_vec/Fs;
End_vec_sec = end_vec/Fs;

Start_vec_sec = Start_vec_sec - ...
    extra_time_window/2;
End_vec_sec = End_vec_sec +...
    extra_time_window/2;
T =1/Fs;
Start_vec_sec(Start_vec_sec<0)=T;   %Avoid negative times 

for k = 1:length(Start_vec_sec)
    if End_vec_sec(k) < length(Entropy_sig_out)/Fs
        Entropy_sig_out(floor(Start_vec_sec(k)*Fs):floor(End_vec_sec(k)*Fs))=1;
    else
        Entropy_sig_out(floor(Start_vec_sec(k)*Fs):end)=1;
        End_vec_sec(k) = length(Entropy_sig_out)/Fs; %should work - Ilan
    end
end


end

