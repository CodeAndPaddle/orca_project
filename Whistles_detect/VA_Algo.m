function [X, ProbMat] = VA_Algo(B, A, N, K, NumPoss)

%N: number of observations (time)
%K: number of states (frequency)
%NumPoss: number of paths to check
%A - transition matrix. Matrix of probabilities to pass between two states  
%B - emission matrix. in this case - the normalized spectrogram

% Y = 1 : N;  %why is Y even necessarry here? seems very redundant
% Pi =(1/K)*ones(1,K);%initial probability - redundant Ilan

X = zeros(1,N);

T1 = zeros(N,K);% T1(i,j)  stores the probability of the most likely path so far X ^ = ( x ^ 1 , x ^ 2 , … , x ^ i ) with x^i = s^j  that generates Y = ( y 1 , y 2 , … , y i )
T2 = zeros(N,K);% T2(i,j)  stores x^i-1 of the most likely path so far X ^ = ( x ^ 1 , x ^ 2 , … , x ^( i -1),x^i = s^j )for each 2<=i<=T

%the next loop seems redundant
    T1(1,:) = B(1,:);  %why not just devide by K, whats the point of Pi, (then there is normaliztion, so even deviding by K is redundant)?
%     T2(1,:) = 0;        %Already zero

for ii = 2 : N
    %T1(i-1,:) = T1(i-1,:)/sum(T1(i-1,:));%normalize to avoid numeric precision
    T1(ii-1,:) = T1(ii-1,:)/max(T1(ii-1,:));%normalize to avoid numeric precision 
%     [T1(ii,:),T2(ii,:)]=max(diag(T1(ii-1,:))*A*diag(B(ii,:))); %this replaces the loop
    
     for jj=1:K
        tempVec = T1(ii-1,:).* (A(:,jj)'*B(ii,jj));
        %tempVec = bsxfun(@times,T1(ii-1,:),A(:,jj)'*B(ii,jj));
        [val, pos] = max(tempVec);
        T1(ii,jj) = val;
        T2(ii,jj) = pos;
        [T1(ii,jj),T2(ii,jj)] = max(tempVec);
    end 
%     disp([num2str(ii),'/',num2str(N)]);


end

%     [val, pos] = sort(T1(N,:), 'descend');
%     if PosInd == 1
%         Z(PosInd, N) = pos(1);
%     else
%         loc = round(rand(1)*size(B,2));
%         Z(PosInd, N) = pos(loc);
%     end
%     X(PosInd, N) = Z(PosInd, N);

%the next 7 lines do the same thing as the loop that follows.
% T1=T1';
% ProbMat = T1(randi(K,NumPoss,N) + repmat((0:K:(N-1)*K),NumPoss,1));
% [ProbMat(1,N),X(N)] = max(T1(N,:));
% for ind=N:-1:2
%         X(ind-1) = T2(ind,X(ind));
%         ProbMat(1,ind-1) = T1(X(ind),ind);
% end


ProbMat = zeros(NumPoss,N);
for PosInd = 1: NumPoss
    Z = zeros(1,N);         %Why use this at all?
    if PosInd == 1
        [MaxVal,Z(N)] = max(T1(N,:));
        X(N)=Z(N);
    else
        loc = randi(K);     %max([1, round(rand(1)*size(B,2))]); %Why not just use K? why random? Why max? will this ever be under 1 after rounding?
        MaxVal = T1(N,loc);
    end
    ProbMat(PosInd,N) = MaxVal;
    
    for ind=N:-1:2
        if PosInd == 1
            Z(ind-1) = T2(ind,Z(ind));
            X(ind-1) = Z(ind-1);
            ProbMat(PosInd,ind-1) = T1(ind,Z(ind));
        else
            loc = randi(K);     %max([1, round(rand(1)*size(B,2))]);
            ProbMat(PosInd,ind-1) = T1(ind,loc);
        end
    end
end

