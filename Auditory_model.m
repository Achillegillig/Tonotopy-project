clear
close all
% add path to models: laptop
addpath("C:\Users\achil\Documents\Universite\M2_CNN_local\Research_Project\amtoolbox-full-1.2.0\");

%on home computer
% % % addpath("G:\Project_local\amtoolbox-full-1.2.0");

amt_start;
tic;
%add path to files
folder = "C:\Users\achil\Documents\Universite\M2_CNN_local\Research_Project\Stimuli\Domenico\Cut_stim";
addpath(folder);

%retrieve stimulus names
DirList = dir(fullfile(folder, '*.wav'));
stim_names = {DirList.name};

%define the different loudnesses to test
% L = [60 80]; %Filipchuck et al
L = [-10 0 10 20 30 40 50 60 70 80];


FS = 10e3; %chosen frequency of sampling for stimuli

%%%%%%% Step 1: import files and transform them to matching dB SPLs
stim = {};
stim_id = {};
FSstim = {};

% c = 0;
for l = 1:length(stim_names)
    
    [temp_stim, FSstim{l}] = audioread(stim_names{l});

        %  Convert from stereo to mono if necessary
    if size(temp_stim,2) == 2
            temp_stim = mean(temp_stim,2);
    end
        % %% resample stimuli if necessary
    if FSstim{l} ~= 10e3
        temp_stim = resample(temp_stim, FS, FSstim{l});
        FSstim{l} = FS;
    end

    stim{l} = temp_stim;
    stim_id{l} = stim_names(l);
end

for l = 1:size(stim,2)
    temp_stim = stim{l};
    for b = 1:length(L)
        stim{l}(:,b) = scaletodbspl(temp_stim, L(b));
    end
end


fc_flag = 'all';
numH = 13; % Nr. of high-spontaneous   rate neurons to be simulated (default=13)
numM =  3; % Nr. of middle-spontaneous rate (default=3)
numL =  3; % Nr. of low-spontaneous    rate (default=3)

%% run the verhulst2018 model
ic_out = cell(length(L),length(stim));
for l = 1:length(stim)
    fprintf('computing IHC-AN simulation for stim %i of %i \n' , l, size(stim,2))
    tic;
    input_stim = stim{l};
    input_FSstim = FS;
    out = verhulst2018(input_stim,input_FSstim,fc_flag, ...
       'v','oae','anfH','anfM','anfL',... % model flags, specify the required output 
       'numH',numH,'numM',numM,'numL',numL); % number of AN fibres
    for b = 1:length(L)
        ic_out{b,l} = out(b).ic;
    end
    toc
end


%% save & export
cf = out.cf;
writematrix(cf);
writetable(cell2table(stim_names), "stim_names")
stim_id = repmat(stim_id, size(L,2), 1);

for i = 1:size(stim_names)
    for b = 1:size(L,2)
        new_stim = [];
        new_stim = cell2mat(ic_out(strcmp(horzcat(stim_id{b,:}), stim_names{i})'));
        filename = sprintf('ic_out_stim_0%i_%idB', i, L(b));
        writematrix(new_stim, filename);
    end
end
       
for l = 1:length(stim)
    for b = 1:length(L)
        filename = sprintf('ic_out_stim_0%i_%idB', l, L(b));
        writematrix(ic_out{b,l}, filename);
    end
end



