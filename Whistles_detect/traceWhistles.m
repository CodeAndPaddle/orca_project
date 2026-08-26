function [traced,whistles_out,start_vec,end_vec] = ...
    traceWhistles(whistles,max_trans,Fs,NumPoss,th_v,start_vec,end_vec)
% Description:
%   This function traces spectrograms using hmm viterbi function
%   Validates whistles by getting a number of viterbi traces and comparing
%   the distance between them, assuming they must be close for real whistles
%
% inputs:
%     whistles - cell array containing defferent length signals
%     max_trans - defines the amount of frequencies a signal can hop in 
%                 the transition matrix
%     Fs - signals sampling frequency
%     NumPoss - number of possibilities checked by viterbi
% returns:
%     traced - A cell array containing different length tracings in.
%                 Amplitude is already adjusted to fit frequencies and

traced = cell(length(whistles),1);
whistles_out = cell(length(whistles),1);
% Create matrixes for hmmviterbi func
% emission matrix:
Logger = log4m.getLogger('logfile.txt');
Logger.debug('Viterbi debug: ','creating emission matrix');
emis = cellfun(@(x) ...
    abs(spectrogram(x,256,120,1024,Fs, 'yaxis'))...
    ,whistles,'UniformOutput',false);
%normalize spcetrograms
for ii = 1:length(whistles)
    emis{ii}(emis{ii}==0)=eps;        % before normalization swap 0s with epsilon
    emis{ii}=emis{ii}./sum(emis{ii});  % normalize emission mat.
end
delete_vec=[];
Logger.debug('Viterbi debug: ','Try hmmviterbi function');
for ii = 1 : length(whistles)
    freq_samples = size(emis{ii},1);
    time_samples = size(emis{ii},2);
    Logger.debug('Viterbi debug: ','creating transition matrix');
    
    % transition :
    tmp = repmat(1:freq_samples,freq_samples,1);
    trans = abs(tmp-tmp')+1<max_trans+1;
    trans(:,1)=1;trans(:,end)=1;
    trans = trans./sum(trans);% normalize transition mat.
    
    
%     trans = ones(freq_samples)*eps;
%     for kk = 1 : freq_samples
%         for jj = 2 : freq_samples
%             if  (abs(kk-jj) < max_trans) && (freq_samples>kk)&&(kk>0) && ...
%                     freq_samples>jj &&jj>0
%                 trans(kk,jj) = 10e3;
%             end
%         end
%     end
%     trans = trans./sum(trans);% normalize transition mat.
%     sequence = 1:1:time_samples;% there is only one sequence possible 
%                and it follows time
%     estimatesStates = ...
%         hmmviterbi(sequence,trans,emis{i});

    [X, ProbMat] = VA_Algo(emis{ii}',trans , time_samples,...
        freq_samples, NumPoss);
%     figure();
%     subplot(3,1,1)
%     spectrogram(whistles{i},256,120,1024,Fs, 'yaxis');
%     subplot(3,1,2)
%     plot((ProbMat(1,:))')
%     hold on
%     plot(((ProbMat(2,:)))')
%     plot(ones(1,length((ProbMat(1,:))))*mean((ProbMat(1,:))))
%     plot(ones(1,length((ProbMat(2,:))))*mean((ProbMat(2,:))))
%     subplot(3,1,3)
%     plot(X)
    [FoundFlag, ~] = DetectTarget2(ProbMat, th_v);
    if ~FoundFlag
        traced(ii) = num2cell(X(1,:)'.*(Fs/2)/freq_samples,1);
        whistles_out{ii} = whistles{ii};
    else
        delete_vec =[delete_vec; ii];
    end
    
end
traced(delete_vec)=[];
whistles_out(delete_vec)=[];
start_vec(delete_vec)=[];
end_vec(delete_vec)=[];
Logger.info('Status ','Finished Viterbi algo');

% debug plots:
% figure
% for i= 1:1:length(whistles)
%         freq_samples = size(emis{i},1);
%     time_samples = size(emis{i},2);
%     subplot(length(whistles),length(whistles),i)
%     surf(emis{i},'EdgeColor','flat')
%     view(2)
%     ylim([0 freq_samples])
%     xlim([0 time_samples])
%     title(['Whistle number: ' num2str(i)])
%     subplot(length(whistles),length(whistles),(i+length(whistles)))
%     plot(estimatesStates)
%     ylim([0 freq_samples])
%     xlim([0 time_samples])
%     title(['viterbi most likely path: ' num2str(i)])
% end
end

