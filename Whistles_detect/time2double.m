function [double_time] = time2double(time_in)
    ind=find(time_in==':',1);
    double_time=60*str2double(time_in(1:ind-1))+...
        str2double(time_in(ind+1:end));
end

