function paramlist = func_getparameterlist(param)
% paramlist = func_getparameterlist(param)
% Input:  param - parameter (optional)
% Output: paramlist - list of parameters; or
%         index of the parameter in the list
% Notes:  Dual purpose function
%
% Author: Yen-Liang Shue, Speech Processing and Auditory Perception Laboratory, UCLA
% Modifications: Soo Jin Park
% Copyright UCLA SPAPL 2009-2015


paramlist = {'F0 (Straight)', ...
             'F0 (Snack)', ...
             'F0 (Praat)', ...
             'F0 (SHR)', ...
             'F0 (Other)', ...
             'Formants (Snack)', ...
             'Formants (Praat)', ...
             'Formants (Other)', ...
             'H1, H2, H4', ...
             'A1, A2, A3', ...
             '2K', ...
             '5K', ...
             'H1*-H2*, H2*-H4*', ...
             'H1*-A1*, H1*-A2*, H1*-A3*', ...
             'H4*-2K*', ...
             '2K*-5K', ...
             'Energy', ...
             'CPP', ...
             'Harmonic to Noise Ratios - HNR', ...
             'Subharmonic to Harmonic Ratio - SHR', ...
             'epoch, excitation strength (epoch, SoE)',...
             };


% user is asking for index to a param
if (nargin == 1)
    for k=1:length(paramlist)
        if (strcmp(paramlist{k}, param))
            paramlist = k;
            return;
        end
    end
    paramlist = -1;  % param not found in list
end
