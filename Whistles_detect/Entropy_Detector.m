function [sig_out,Start_vec,End_vec, Entropy_history] = ...
    Entropy_Detector(sig, Fs, p_fa,max_window_size)
% Overview: 
%   Function runs through predetermined windows ( varibale: intreval )
%   Calculating spectral entropy
%   Then assumes beginning of the recording is noise, and begins searching
%   for whistles adapting to noise as signal progresses
%   max_window_size eliminates long false detections
% Outputs:
%     sig_out - 0/1 vector indicating possible index whistle exists
%     Start_vec - first index of possible whistles
%     End_vec - last index of possible whistles
%     Entropy_history - Spectral Entropy log 
%
% Comments :
%   indexes can be easily translated to sec by dividing by Fs
%% Constants

%Signal Processing Constants
sig_out = zeros(size(sig));

%how many samples we take for Adapting (assume there's only noise)
adapt_length = 0.3*Fs;  % originaly 3*Fs, but 3 seconds is alot for the tagged whistles;

%how many samples we consider each iteration
interval = floor(Fs/10);


%% adapt to noise

LW = length(sig)/interval;
Ld = interval*ceil(LW);

sigMat = reshape(cat(1,sig,zeros(ceil(Ld-length(sig)),1)), [],Ld/interval);


l = 1 : Ld/interval;

penMat = arrayfun(@(l) pentropy(sigMat(:,l),Fs),l,'UniformOutput',0);
penMat = (cell2mat(penMat));
pentropy_length = size(penMat,1);

first_noise_cols = adapt_length/interval;
pen_vec = reshape(penMat,1,[]);
noise_segment = pen_vec(1:first_noise_cols*pentropy_length);
E_SE_Noise = mean(noise_segment);
Sigma_SE_Noise = std(noise_segment);

% norm_noise_segment=(noise_segment-E_SE_Noise)/Sigma_SE_Noise;
% sorted_norm_noise_segment=sort(norm_noise_segment, 'descend');

% value_of_p_fa=sorted_norm_noise_segment(find...
%     (qfunc(sorted_norm_noise_segment)>=1-p_fa,1));
% if (isempty(value_of_p_fa))
%     th_e=-Sigma_SE_Noise;
% else
%     th_e=norm_noise_segment(find(norm_noise_segment==value_of_p_fa,1));
% end
th_e = qfuncinv(1-p_fa);

%% Process
Start_vec = [];
End_vec = [];
detected_whistle = false;
k=(first_noise_cols-2)*pentropy_length; %maybe +1 - Ilan
for i = adapt_length : interval:length(sig)
    
    seg_se = pen_vec(k:k+pentropy_length);
    k = k + pentropy_length;
    %Normalized segment entropy
    seg_se_Norm = (seg_se-E_SE_Noise) / Sigma_SE_Noise;
    E_seg_se = mean(seg_se_Norm);
%     Sigma_seg_se = std(seg_se_Norm);      %redundant
    
    
    % compare location of signal input to known noise
    if ( E_seg_se <= th_e )
        if ~detected_whistle
            Start_vec = [Start_vec i-3*interval];
            detected_whistle = true;
        end
        % take 3 intervals back because always misses start
        sig_out(i-3*interval:i) = 1;      %add 3 intervals in front of first detection
        % check if the whistle is too long - if so it's probably 
        % falsly detected
        if(max_window_size*Fs < i - Start_vec(end))
            End_vec = [End_vec i-interval];
            detected_whistle = false;
            [E_SE_Noise,Sigma_SE_Noise] = ...
                adaptNoiseParams(seg_se,noise_segment);
        end
    else % not whistle
        if (detected_whistle) %whistle just ended
            detected_whistle = false;
            %sig_out(i-interval:i)=1;
            End_vec = [End_vec i-interval];
            
        else %no whistle, adapt noise
            detected_whistle = false;
            sig_out(i-interval:i) = 0;
           
            [E_SE_Noise,Sigma_SE_Noise] = ...
                adaptNoiseParams(seg_se,noise_segment);
        end
    end
end
if (length(End_vec) == length(Start_vec) - 1)   %add end to whistle if at end of file (make sure same number of starts and ends)
    End_vec = [End_vec length(sig)];
end
Entropy_history = pen_vec;
end

function [E_SE_Noise,Sigma_SE_Noise] = ...
    adaptNoiseParams(seg_se,noise_segment)
        noise_segment(1:length(seg_se))=seg_se;
        noise_segment = circshift(noise_segment,-length(seg_se));
        E_SE_Noise = mean(noise_segment);
        Sigma_SE_Noise = std(noise_segment);
end

