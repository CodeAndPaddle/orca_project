function [FoundFlag, ValidLoc] = DetectTarget2(X, Th)

%first row of ProbMat correspongs to the found path

% Th = K/10;
% NumPoss = size(X,1)-1;
    DiffVec=abs(X(1,:)-X(2:end,:)); %calculate difference of first row from other rows
    MeanDiffVec=mean(DiffVec,2);    %take the mean of the difference

% DiffVec = zeros(size(X));
% MeanDiffVec = zeros(1, NumPoss);
% 
% for ind = 1: NumPoss
%     CurrentDiff = abs(X(1,:) - X(ind+1,:));
%     DiffVec(ind, :) = CurrentDiff; 
%     MeanDiffVec(ind) = mean(CurrentDiff);
% end
% MeanDiffVec
    if any(MeanDiffVec > Th)
        FoundFlag = 0;
        ValidLoc = [];
    else
        FoundFlag = 1;
        ValidLoc = find(DiffVec(1,:) <= Th);
    end

end