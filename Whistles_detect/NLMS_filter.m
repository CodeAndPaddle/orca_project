function [res_signal,h_prev,h_curr] = NLMS_filter(Rx_signal,bufsize,mu,N,h_prev,h_curr,two_filters)
%
%function [res_signal,h_prev,h_curr] = NLMS_filter(Rx_signal,bufsize,mu,N,h_prev,h_curr,two_filters)
%
%   This function executes a autocorrelation NLMS filter, with parameters
%   mu (step size), bufsize (buffer size) and N (impulse responce length)
%   with previous filters allready input (in order to be able to filter
%   adjacent sections of a signal with the adaptive filters starting
%   position for the next section as the end position for the previous
%   section).
%   Input:
%           Pfa -           A value corresponding to the probability of
%                           false alarm. This value is used to derive the
%                           detection threshold via the inverse normal
%                           distribution function.
%           sig -           Signal on which the detector is run
%           WinLen -        Length of window for detection
%           noise_time -    Time (in seconds) from start of signal that is
%                           assumed to be noise. Used to determine noise
%                           mean and variance for power centering and
%                           normalization.
%           Fs -            Sampling frequency (used to convert noise_time
%                           from seconds to cell number)
%
%   output:
%          noise_flag_ind - A vector indicating the detection result for
%                           each windowed part of the signal.
%
%   Created by Eitan Ovrutski, October 2018



    num_buf=ceil(length(Rx_signal)/bufsize);        %%%% Changed from floor to include last partial buffer of section
    res_signal=Rx_signal;

    %% start filtering                
    prev_buff=[zeros(N-1,1);Rx_signal(1:bufsize)];
    for ll=1:(num_buf-1)
        ind=ll*bufsize + 1:min((ll+1)*bufsize,size(Rx_signal));
        curr_buff=Rx_signal(ind);

        [ h_prev,r_err,~ ] = simple_NLMS(prev_buff,curr_buff,h_prev,mu,N);  
         %recycling the filter's result
         %second filter
        if two_filters
            [ h_curr,r_err,~ ] = simple_NLMS([prev_buff(1:N-1);r_err],curr_buff,h_curr,mu,N);
        end
        res_signal(ind)=r_err;
        prev_buff=[prev_buff(end-N+2:end);r_err];
    end
end