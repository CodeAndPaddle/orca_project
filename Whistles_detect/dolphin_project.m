function [whistle_times_mat, features] = dolphin_project( sig_filtered , Fs,...
    min_window ,...% in sec
    extra_time_window,...% in sec
    th_c,...%[];%   if empty , correlator is adaptive
    max_trans, ...,% Viterbi frequency bins
    p_fa, ...% probability of noise after Entropy detector
    corr_wind_size ,...% sliding window size for correlation
    detection_wind_size,...% detector window size
    max_window_size,... %in sec; eliminates extreme false detections
    NumPoss,...% number of viterbi paths - not used in this version
    th_v,...% % confidence present it is a whistle in viterbi algo
    res_dir) % results directory

%% set up logger:
Logger = log4m.getLogger('logfile.txt');
Logger.info('Status ','Started script');

% control levels logged to command window
Logger.setCommandWindowLevel(Logger.INFO);

% L.setLogLevel(L.ERROR);  % control levels logged to file
%% Get file
%{
test_mode=0;
if (test_mode==0)
    Logger.info('Status ','Prompt for selecting file');
    [filename,pathname]= uigetfile('*.wav');
    [ sig , Fs ] = audioread([pathname filename]);
    Logger.info('Status ',['Opened file: ',filename] );

%}
%% synthetic chirp and noises
%{
Fs = 96e3;
num_of_chirps = 3;
sig_duration =20;
noise_amp = 0.2;
chirp_duration = 1; % +- 0.5 sec
sig = chirpSignal(Fs,sig_duration,chirp_duration,num_of_chirps,noise_amp);
Logger.info('Status ',['Created chirps:'...
    'length =', num2str(chirp_duration),'[sec]'] );
deltas_len = 0.0005;%[sec]
sig = add_noise_deltas(sig,Fs,deltas_len);
%}
%% constants for debug runs
%{
    min_window = 0.1 ;% in sec
    extra_time_window = 0.5 ;% in sec
    th_c =0.02 ;%[];%   if empty , correlator is adaptive
    max_trans = 10;% TBD
    p_fa=0.1; %probability of false alarm
    corr_wind_size = floor(Fs/5e3);% sliding window size for correlation
    detection_wind_size = floor(Fs/10);% detector window size
    max_window_size = 7; %in sec; eliminates extreme false detections
    NumPoss=5;% number of viterbi paths
    th_v=0.75;% % confidence present it is a whistle in viterbi algo
% end
%}




%% Num of channels:
channel2 = false;
if any(size(sig_filtered)==2)
    channel2 = true;
    Logger.info('Status ','2 channels detected');
else
    Logger.info('Status ','only 1 channel detected');
end


%% Entropy Detector - initial detection of whistles
Logger.info('Status ','Start Entropy detector');

[Entropy_sig_out1,start_vec,end_vec,Ent_hist]= ...
    Entropy_Detector(sig_filtered(:,1), Fs, p_fa,max_window_size);
if channel2
    [Entropy_sig_out2,start_vec2,end_vec2,~]= ...
        Entropy_Detector(sig_filtered(:,2), Fs, p_fa,max_window_size);
    Entropy_sig_out=or(Entropy_sig_out1 , Entropy_sig_out2);
    if length(start_vec2)>length(start_vec) %review - perhaps change nature of start and end detection
        start_vec=start_vec2;
        end_vec = end_vec2;
    end
else
    Entropy_sig_out = Entropy_sig_out1;
end
Logger.info('Status ','Finished Entropy detector');

%% extend signal for correlation detector - extend detected windows to insure inclusion of entire whistle
Logger.info('Status ','Start "whistle scaling"');
[Entropy_sig_out,start_vec,end_vec] =....
    determineWindow(extra_time_window,Entropy_sig_out,start_vec,...
    end_vec,Fs);
Logger.info('Status ','Finished "whistle scaling"');

% % takes only one channel? :(

sig_entropy_output = Entropy_sig_out.*sig_filtered(:,1); %use only first channel - possible to extend for both

%% present results -Debug
%
%{
Logger.debug('Plot: ','Ploting whistles ');

%if (test_mode==0)
    figure();
    subplot(3,1,1);
    spectrogram(sig_filtered(:,1),256,120,1024,Fs, 'yaxis');
    title('Input Signal');
    subplot(3,1,2);
    spectrogram(sig_entropy_output,256,120,1024,Fs, 'yaxis');
    title('Output Signal after entropy detector');
    subplot(3,1,3);
    plot(Ent_hist);
end
%}

%% Convert to Cells
% convert segments to cells - break up original audio into individual
% whistle vectors

whistles = vecs2cell(sig_entropy_output,start_vec,end_vec);

%% Correlation detector - detect start and end of whistles more accurately
Logger.info('Status ','Start correlation detector');
[whistles_corr,start_vec_corr1,end_vec_corr1]= ...
    Correlation_Detector(whistles, Fs,th_c,min_window,...
    corr_wind_size,detection_wind_size,start_vec,end_vec);
Logger.info('Status ','Finished correlation detector');


%% present results - Debug
%{
if ~test_mode

    Logger.debug('Plot: ','Ploting whistles ');
    ln=length(whistles_corr);
    if ~ln
        ln=2;
    end

    figure();
    for i = 1:1:ln
        subplot(1,ln,i);
        spectrogram(whistles_corr{i},256,120,1024,Fs, 'yaxis');
        title('Output Signal after correlator');
    end
end
%}
%% Remove short whistles
Logger.info('Status ','Remove short whistles before tracing');
delete_vec = find(cellfun(@length,whistles_corr)<1000);
whistles_corr(delete_vec)=[];
start_vec_corr1(delete_vec)=[];
end_vec_corr1(delete_vec)=[];



%% viterbi:
Logger.info('Status ','Start Viterbi algorithm');
[traced,whistles_vit,start_vec_vit,end_vec_vit] =...
    traceWhistles(whistles_corr,max_trans,Fs,NumPoss,th_v,...
    start_vec_corr1,end_vec_corr1);
Logger.info('Status ','Finished Viterbi algorithm');

%% present results after viterbi: -Debug

%{
%NOTE: the axes' scale changes according to how many whsitles there are
%Therefore, you have to adapt the constant in the second plot accordingly
Logger.info('Status ','Plot Viterbi tracing outputs');
if (test_mode==0)
    figure()
    for i= 1:length(whistles_vit)
        figure
        emis = spectrogram(whistles_vit{i},256,120,1024,Fs, 'yaxis');
        subplot(2,length(whistles_vit),i)
        spectrogram(whistles_vit{i},256,120,1024,Fs, 'yaxis')

        % Scale time and frequency of Viterbi results to spectrogram
        ax=gca;
        if (ax.XLim(2)>10)
            time_scale=1000;%scale to miliseconds
        else
            time_scale=1;%scale to seconds
        end

        if (ax.YLim(2)>1000)
            freq_scale=1;%scale to KHz
        else
            freq_scale=1/1000;%scale to Hz
        end
        title(['Whistle number: ' num2str(i)])
        f = 0:(Fs/2)/size(emis,1):Fs/2;
        subplot(2,length(whistles_vit),i+length(whistles_vit))
        spectrogram(whistles_vit{i},256,120,1024,Fs, 'yaxis')
        hold on
        t = 0:length(whistles_vit{i})/Fs/length(traced{i}):...
            length(whistles_vit{i})/Fs;
        plot(time_scale.*t(1:end-1),medfilt1(traced{i}.*freq_scale,20),'r');
        hold off
        title(['viterbi most likely path: ' num2str(i)])

    end
    savefig([res_dir,'viterbi_output'])
end
%}

%% Log detection:
% logDetection(whistles_vit,Fs,start_vec_vit,end_vec_vit,res_dir);


%% Feature Extraction
freq_scale=1;
time_scale=1;

whistle_times_cell = [start_vec_vit,end_vec_vit]/Fs;
features = extractFeatures(whistles_vit,Fs,traced,freq_scale,...
    time_scale, whistle_times_cell);
Tb = cell2table(features(2:end, :));
Tb.Properties.VariableNames = (features(1,:));
% writetable(Tb,[res_dir, 'features.csv']);
whistle_times_mat=whistle_times_cell;   %what for?

% save([res_dir,'features.mat'], 'features');
% end