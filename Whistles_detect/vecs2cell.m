function [cell_output] = vecs2cell(full,start_vec,end_vec)
% utility function to turn a full recording to cells according to start
% vec and end vec - easier to deal with and rejects long periods of noise
    cell_output = cell(length(start_vec),1);
    for i = 1:length(start_vec)
        cell_output(i) = num2cell(full(...
            floor(start_vec(i)):ceil(end_vec(i))),1);
    end

end