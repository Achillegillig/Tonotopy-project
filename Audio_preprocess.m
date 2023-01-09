clear
close all
%%% Script to preprocess audio files: 
%%% (1) Cuts a sequence of stimulus of given name into multiple files
%%% each containing a single stimulus. 
%%% Also, transforms the audio to mono if the input is stereo.

%add path to files
folder = "C:\Users\achil\Documents\Universite\M2_CNN_local\Research_Project\Stimuli\Domenico";
addpath(folder);

%retrieve stimulus names
DirList = dir(fullfile(folder, '*.wav'));
stim_names = {DirList.name};


stim = {};
stim_id = {};
FSstim = nan(1);

FS = 10e3;

c = 0;
p = 0;
for l = 1:length(stim_names)
   
        [temp_stim, FSstim] = audioread(stim_names{l});

        %  Convert from stereo to mono if necessary
    if size(temp_stim,2) == 2
            temp_stim = mean(temp_stim,2);
    end
        % %% resample stimuli if necessary
    if FSstim ~= FS
        temp_stim = resample(temp_stim, FS, FSstim);
        FSstim = FS;
    end
 
    if strcmp(stim_names(l), 'protocol0_sequence_mouse_tone.wav')
        seq_start = 20000; %start of the sequence in bins
        stim_dur = 5000; %duration of stim in bins
        ITI = 10000; %inter-trial interval
        i = seq_start;
        n_stim = 14;
        n_ITI = n_stim-1;

        for n = 1:n_stim+n_ITI-1

            if (mod(n,2) == 0) == 0
                p = p+1;
                if p < 10
                    filename = [stim_names{1} '_stim_0' num2str(p) '.wav'];
                else
                    filename = [stim_names{1} '_stim_' num2str(p) '.wav'];
                end
               
                j = i + stim_dur;
                audiowrite(filename, temp_stim(i:j-1), FSstim);
%                 stim_id{l+c} = stim_names(l);
                c = c+1;
                i = j;
            else
                j = i + ITI;
                i = j;

            end
        
%             else
%                 j = i + stim_dur;
%             end

        end
    else
        break %add conditions for other file names
    end
end
