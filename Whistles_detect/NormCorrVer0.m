function[Corr] = NormCorrVer0(Data,Ref, FftFlag, NormFlag, StartIndx, StopIndx)
% function[Corr] = NormCorrVer0(Data,Ref, FftFlag, NormFlag ,StartIndx, StopIndx)
%
%  Description: This function perform normalized correlation between ...
%               a refference signal and data signal. It can calculate the ...
%               correlation output in time domain or frequency domain.
%               It also can return normalized correlation.
%
% Input parameters:
%             Data - The data signal.
%             Ref - The refference signal.
%             FftFlag -  A flag determines whether to use time domain
%                        implementation  or frequency domain implementation.
%             NormFlag - A flag determines whether to normalize the correlation output
%             StartIndx - Start index of the result relative to the case where
%                         the result is of length = length(Data)+RefL - 1
%             StopIndx - Stop index of the result relative to the case where
%                         the result is of length = length(Data)+RefL - 1
%
%  Output parameters:
%            Corr - The correlation result.
%
%
%
%    Written by : Kobi Bucris 15.7.07
%    Updated by : Kobi Bucris 19.2.08
%    Updated by : Shlomi Museri 6.9.2011
%

%   Set default values to StartIndx and StopIndx
if nargin < 3
    FftFlag = 1 ;
end
if nargin < 4
    NormFlag = 0;
end
if nargin < 5
    StartIndx = length(Ref) ;
end
if nargin < 6
    StopIndx = length(Data) + length(Ref) -1 ;
end

% Assure row vectors
Data = Data(:).';
Ref = Ref(:).';

K = sqrt( Ref * Ref');      % power of the reference;
RefL = length(Ref);         % The length of the reference
DataLen = length(Data);     % The length of the data

if FftFlag
    % method 1 - fft implementation
    pad = 2 ^ nextpow2(RefL + DataLen  - 1);% Padding to a power of 2 for the fft function
    Corr= ifft(fft(Data,pad).*fft(conj(fliplr(Ref)),pad)); % Correlation in frequency domain
    % figure;plot(abs(Corr))
    Corr = Corr( StartIndx : StopIndx );  % cutting the fft implementation to the supposed length.
    % Normalize the correlation
    if NormFlag
        Data = [zeros(1,RefL-1) Data zeros(1,RefL)]; % zero padding to perform normalization
        SumX2 = sum(abs(Data(StartIndx: StartIndx + RefL - 1)).^2); % the energy of each point
        for ind = StartIndx:StopIndx
            Corr(ind -  StartIndx + 1) = Corr(ind -  StartIndx + 1) / sqrt( SumX2) ;
            SumX2 = SumX2 + abs(Data(ind + RefL )).^2 - abs(Data(ind)).^2;
        end
    end
else
    % method 2 - time implementation
    Data = [zeros(1,RefL-1) Data zeros(1,RefL)];% zero padding to perform normalization
    Corr = zeros(1,StopIndx - StartIndx + 1); % Initialization of Corr
    SumX2 = sum(abs(Data(StartIndx: StartIndx + RefL - 1)).^2);% the energy of each point
    for ind = StartIndx:StopIndx
        Corr(ind -  StartIndx + 1) = Data(ind:ind  + RefL - 1)*Ref';% Compute the correlation
        % Normalize the correlation
        if NormFlag
            Corr(ind -  StartIndx + 1) = Corr(ind -  StartIndx + 1) / sqrt( SumX2) ;
            SumX2 = SumX2 + abs(Data(ind + RefL )).^2 - abs(Data(ind)).^2;
        end
    end
end
% Normalize the correlation
if NormFlag
    Corr = Corr/K;
end
