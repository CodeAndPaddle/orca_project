function [cmp_table] = Features_Compare(Tagged,Extracted)
% This function is used to compare Tagged features to the features which
% were extracted by our system. Only TP detected whistles should be
% inserted here


cmp_table={'StartTime_diff','EndTime_diff','startFreq_KHz_diff','endFreq_KHz_diff','minFreq_KHz_diff','maxFreq_KHz_diff','x_InflectionPoints_diff'};

%% Constants
Freq_error=1; %in KHz
Duration_Err=0.5;%in seconds
Inflection_error=2; %Error in # inflection points


%% Calculation

TP=cell(size(Extracted)); %True Positive per feature
TP(1,1:end)=Extracted(1,1:end);
FN=TP; %False Negative per feature

for i=2:size(Extracted,1)
        %Start Time
    cmp_table(i,1) = { abs(cell2mat(Extracted(i,8))-...
            (cell2mat(Tagged(i,3))))};
            %End Time
    cmp_table(i,2) = { abs(cell2mat(Extracted(i,9))-...
            (cell2mat(Tagged(i,4))))};
    
    %Start Freq
    cmp_table(i,3) = { abs(cell2mat(Extracted(i,1))-...
            (cell2mat(Tagged(i,5))))/1e3};
%     if abs(cell2mat(Extracted(i,1))*1e-3-...
%             str2double(cell2mat(Tagged(i,5)))) > Freq_error
%         TP(i,1)={0};
%         FN(i,1)={1};
% %     else
%         TP(i,1)={1};
%         FN(i,1)={0};
%     end
    
    %End Freq
    cmp_table(i,4) = { abs(cell2mat(Extracted(i,2))-...
            (cell2mat(Tagged(i,6))))/1e3};
%     if abs(cell2mat(Extracted(i,2))*1e-3-...
%             str2double(cell2mat(Tagged(i,6)))) > Freq_error
%         TP(i,2)={0};
%         FN(i,2)={1};
%     else
%         TP(i,2)={1};
%         FN(i,2)={0};
%     end
%     
    %Duration
%     Tagged_Duration=time2double(cell2mat(Tagged(i,4))) - ...
%         time2double(cell2mat(Tagged(i,3)));
%     Extracted_Duration=abs(cell2mat(Extracted(i,3))-...
%         str2double(cell2mat(Tagged(i,6))));
%     if abs(Extracted_Duration-...
%             Tagged_Duration) > Duration_Err
%         TP(i,3)={0};
%         FN(i,3)={1};
%     else
%         TP(i,3)={1};
%         FN(i,3)={0};
%     end
%     
    %Min Freq
    cmp_table(i,5) = {abs(cell2mat(Extracted(i,4))-...
            (cell2mat(Tagged(i,7))))/1e3};
%     if abs(cell2mat(Extracted(i,4))*1e-3-...
%             str2double(cell2mat(Tagged(i,7)))) > Freq_error
%         TP(i,4)={0};
%         FN(i,4)={1};
%     else
%         TP(i,4)={1};
%         FN(i,4)={0};
%     end
    
    
    %Max Freq
    cmp_table(i,6) = {abs(cell2mat(Extracted(i,5))-...
            (cell2mat(Tagged(i,8))))/1e3 };
%     if abs(cell2mat(Extracted(i,5))*1e-3-...
%             str2double(cell2mat(Tagged(i,8)))) > Freq_error
%         TP(i,5)={0};
%         FN(i,5)={1};
%     else
%         TP(i,5)={1};
%         FN(i,5)={0};
%     end
    
    %#Inflection points
     cmp_table(i,7) = {abs(cell2mat(Extracted(i,6))-...
             (cell2mat(Tagged(i,9))))};
%     if abs(cell2mat(Extracted(i,6))-...
%             str2double(cell2mat(Tagged(i,9)))) > Inflection_error
%         TP(i,6)={0};
%         FN(i,6)={1};
%     else
%         TP(i,6)={1};
%         FN(i,6)={0};
%     end
end

end

