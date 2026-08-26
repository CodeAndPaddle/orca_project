function [start_time, end_time,Tagged_whistles_features,sig] = chirpSignal(Fs,sig_duration,chirp_duration,num_of_chirps)
%UNTITLED4 Summary of this function goes here
%   This function is designed for simulations providing data to be able to
%   test for lond periods of time on chirps with thier spectral
%   information

% num_of_chirps =floor(2+abs(randn(1))*3);
% sig = noise_amp *randn((sig_duration)*Fs,1);
sig = zeros((sig_duration)*Fs,1);

for i=1:1:num_of_chirps
    % Set up chirp:
    freq_start(i)=5e3 + abs(rand)*19e3;
    freq_end(i)=5e3 + abs(rand)*19e3;
    % set up chirp and signal length
    rand_loc = floor(Fs*(2+i*(sig_duration-5)/num_of_chirps +abs(rand)/2)); % allow a 3 sec of noise to start with
    % create recording: 10 sec , 1 sec with chirp
    [ ~, chirp_wave] = generateChirp(Fs,freq_start,freq_end,chirp_duration+rand*0.2);
    % adding noise
    
    start_time(i)=rand_loc;
    end_time(i)=start_time(i)+length(chirp_wave);
    sig(start_time(i):end_time(i)-1) = sig(start_time(i):end_time(i)-1) + chirp_wave;
    
    start_time(i)=start_time(i)/Fs;
    end_time(i)=end_time(i)/Fs;
   
end
Tagged_whistles_features=zeros(num_of_chirps,11);
Tagged_whistles_features(:,3) = start_time';
Tagged_whistles_features(:,4) = end_time';
Tagged_whistles_features(:,5)= freq_start';
Tagged_whistles_features(:,6)= freq_end';
Tagged_whistles_features(:,7)= min([freq_start,freq_start])';
Tagged_whistles_features(:,8)= max([freq_start,freq_start])';
Tagged_whistles_features(:,9)= 0;


end

