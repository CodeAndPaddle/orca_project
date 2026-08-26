function [Features] = extractFeatures(whistles,Fs,traces,...
    freq_scale,time_scale, whistle_times_cell)

Features_num=10;
Features=cell(length(whistles)+1,Features_num);
Features(1,:)=  {'Start_Frequency', 'End_Frequency', 'Duration',...
    'Min_Frequency', 'Max_Frequency', 'n_Inflection_Point','n_of_steps',...
    'Start_Time', 'End_Time', 'SNR'};


for i=1:length(whistles)
    
    %Start Frequency
    Features(i+1,1)={freq_scale*traces{i}(1)};
    
    %End Frequency
    Features(i+1,2)={freq_scale*traces{i}(end)};
    
    %Duration in miliseconds
    Features(i+1,3)={time_scale*length(whistles{i})/Fs};
    
    %Min Frequency
    Features(i+1,4)={min(min(traces{i}))};
    
    %Max Frequency
    Features(i+1,5)={max(max(traces{i}))};
    
    %# Inflection Point
    medianed_whistle=medfilt1(traces{i},20);
    Derivative=diff(medianed_whistle);
    Derivative(Derivative>=0)=1;
    Derivative(Derivative<0)=-1;
    Derivative=diff(Derivative);
    Features(i+1,6)={sum(Derivative~=0)};
    
    %# of steps
    diffs=diff(medianed_whistle);
    diffs(abs(diffs)<=50)=1;
    diffs(abs(diffs)>50)=0;
    step_num=0;
    j=1;
    while j<length(diffs)
        start_ind=j-1+find(diffs(j:end),1);
        if (~isempty(start_ind))
            end_ind=start_ind-2+find(~diffs(start_ind:end),1);
            if (isempty(end_ind))
                end_ind=length(diffs);
            end
            step_num=step_num+(end_ind-start_ind>30);
            j=end_ind+1;
        else
            j=length(diffs);
        end
    end
    
    if (isempty(step_num))
            step_num=0;
    end
    
    Features(i+1,7)={step_num};
    
    %Start time
    Features(i+1,8)={whistle_times_cell(i,1)};
    
    %End time
    Features(i+1,9)={whistle_times_cell(i,2)};
    
    %SNR
%     Features(i+1,10)={extract_whistle_SNR(cell2mat(whistles(i))...
%         ,cell2mat(traces(i)),Fs)};
    
end

end

