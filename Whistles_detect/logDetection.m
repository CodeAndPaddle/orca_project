function [detection_log] = logDetection(whistles,Fs,start_vec,end_vec,res_dir)
% save a log file of the start and end times of detected whistles
    detection_log=cell(length(whistles)+1,2);
    detection_log(1,:)=  {'Start time: sec', 'End time: sec'};

    % Start time in seconds
    detection_log(2:end,1)=num2cell(start_vec/Fs);
    % End time in seconds
    detection_log(2:end,2)=num2cell(end_vec/Fs);

    save([res_dir,'detection_log.mat'], 'detection_log');
end

