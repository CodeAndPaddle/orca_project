function [ h_curr,r_err,err ] = simple_NLMS(ref_signal,buff_signal,h_prev,mu,N)
%
%function [ h_prev,r_err,err ] = simple_NLMS(ref_signal,buff_signal,h_prev,mu,N)
% 
%   This function performes othe NLMS Filter algorithm, and returns the 
%   updated filter impulse response at the end of the algorithm and the filtered signal. 
% 
%   Input:
%       ref_signal -      Reference signal, near signal symbol
%       buff_signal -     A buffer of the recieved signal
%       delay_vec -       A binary vector that indicates the channel response
%                         estimation for the current buffer.
%       h_prev -          The channel response of previous buffer
%       mu -              step size for the NLMS < 2/(3*trace(x'*x))
%     
%   Output:
%       h_curr -          The channel response of the current buffer
%       err -             Error from Filter - corresponds to part of signal
%                         with no correlation
%       r_err -           The filtered buffer signal
% 	
%   Created by Eitan Ovrutski, October 2018

buff_size=length(buff_signal); 
err=zeros(buff_size,1);
r_err=err;
% padd=buff_size-length(ref_signal);  %%%%seems redundant - consider removing along with use of this variable - Ilan
% ref_signal=[ ref_signal ; zeros(padd,1)];    %changed from [ 0; ref_signal ; zeros(padd,1)] for removal of redundant 0 - Ilan
% ref_signal=[0; ref_signal; zeros(padd,1)];
beta=1e-3;
    for n=1:buff_size             %-padd)
        x = ref_signal(n+N-1:-1:n); %removed max(1,n+1:-1:n-N+2) restriction because n>=N - Ilan
%         x = ref_signal(max(1,n+1:-1:n-N+2));
        % filter ref with estimated impulse response theta
        r_err(n)=h_prev.'*x;
        e = buff_signal(n)-r_err(n);
        % update estimated impulse response
        h_prev = h_prev + (mu * x * e) / (beta+ x.'*x); %
        err(n)=e;
    end
    h_curr=h_prev;          %update current impulse response
end