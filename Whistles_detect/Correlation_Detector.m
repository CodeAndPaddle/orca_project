function [whistles_out,start_vec,end_vec] = ...
             Correlation_Detector(whistles, Fs,th_c,min_win ,...
             corr_wind,thresh_wind,orgn_start,orgn_end)
         
         %{
Dolphin whistles are easily spotted on spectrograms for being continuous
in frequency and exist for a few hundred msec.
We therfore expect high correlation when correlating with a fragment of
time earlier.
 Inputs: 
    whistles - cells detected
    Fs - Sample rate
    th_c - threshold (typically 0.01-0.8)
    min_win - minimum length of whistle [secs]
    corr_wind - sliding window size in samples
    thresh_wind - sliding window for detection in samples
    orgn_start - indexs of starting times
    orgn_end - indexs of ending times
outputs:
    whistles_out - cells containing whistle recordings 
    start_vec - updated vector
    end_vec - updated vector
%}
     whistles_out = cell(0);
     start_vec=[];
     end_vec=[];
     for i = 1:length(whistles)         
         [tmp,start_whistle,end_whistle] = ...
                Correlation_Detector_help(whistles{i},Fs,th_c,...
                min_win, corr_wind,thresh_wind);
            if ~isempty(tmp)
                whistles_out{end+1} = tmp;
                start_vec =[start_vec ;orgn_start(i)+start_whistle];
                end_vec = [end_vec ;orgn_start(i)+end_whistle];       
            end
     end
end




function [whistle_detected,start_whistle,end_whistle] = ...
    Correlation_Detector_help(recording,Fs,Th_c,min_win,...
    corr_wind,thresh_wind)

%% Sliding window setup:
% set sliding windows length:
win_len = corr_wind;% floor(Fs/5e3);% Fs * (time_in_sec)
LW = length(recording)/win_len;
Ld = win_len*ceil(LW);

recMat = reshape(cat(1,recording,zeros(Ld-size(recording,1),1)), ...
        [],floor(Ld/win_len));

%% Correlate sliding windows:

% feed correlator:
% output_corr=[];
% for l=1:1:length(recording)-win_len-1
%     NormCorr = NormCorrVer0(recording(l:l+win_len),recording(l+1:l+1+win_len));
%     output_corr = [output_corr ,abs(NormCorr)] ;
% end
l=1:length(recMat)-1;
NormCorr = arrayfun(@(l) NormCorrVer0(recMat(:,l),...
    recMat(:,l+1)),l,'UniformOutput',0);
output_corr=abs(cell2mat(NormCorr));    %why use cell2mat instead of flagging UniformOutput as 1?
%{
if isempty(Th_c)
    num_samples =floor(Fs/500);
    output_corr_smoothed=conv(output_corr,ones(1,num_samples)/num_samples,'same');
else
    output_corr_smoothed = output_corr;
end
%}
%%  correlation detector:
% Init detector:
seg_len = thresh_wind;%floor(Fs/6);
detected_whistle=false;

% Assume first segment is noise:
start_whistle=[];
end_whistle=[];
first_seg = output_corr(1:min(seg_len,length(output_corr)));
E_corr_noise= mean(first_seg);
std_corr_noise=std(first_seg);

ind=1;
E_corr_hist=zeros(1,length(output_corr));
E_corr_hist(ind)=E_corr_noise;
ind=ind+1;


% if isempty(Th_c)
%     norm_noise_segment=(first_seg-E_corr_noise)/std_corr_noise;
%     sorted_norm_noise_segment=sort(norm_noise_segment, 'descend');
% 
%     value_of_p_fa=sorted_norm_noise_segment(find...
%         (qfunc(sorted_norm_noise_segment)<=p_fa,1));
%     if ~(isempty(value_of_p_fa))
%         Th_c= norm_noise_segment(find(norm_noise_segment==value_of_p_fa,1));
%     end
% end
% moving window:
for i = seg_len:seg_len:length(output_corr)-seg_len %perhaps + 1
    out_corr_normed = ...
        (output_corr(i:i+seg_len)-E_corr_noise)/std_corr_noise;
    E_corr = mean(out_corr_normed);
%     std_corr = std(out_corr_normed);
    
    E_corr_hist(ind)=E_corr_noise;
    ind=ind+1;
    
    if ( E_corr > Th_c)
        if ~detected_whistle %if first detection
            start_whistle = i;
            detected_whistle=true;
        end
    else
        if detected_whistle   %if stopped detecting
            end_whistle = i;
            if (end_whistle - start_whistle)/Fs < min_win
                break % discover only one whistle! at minimal size
            else
                i=i+seg_len;    %WHY?
                end_whistle= i;
            end
        else %No detection yet - update noise stats
            noise_seg = output_corr(i-seg_len+1:i);
            E_corr_noise= mean(noise_seg);
            std_corr_noise=std(noise_seg);
        end
    end
end

if ~isempty(start_whistle) && isempty(end_whistle)
    end_whistle = i;
    diff = length(output_corr) - seg_len - i;
    if diff < seg_len
       end_whistle = i + diff;
    end
end
whistle_detected = recording((start_whistle):(end_whistle));

end

